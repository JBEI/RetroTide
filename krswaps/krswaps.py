"""
Modules for running RetroTide with the KR Swaps stereochemistry correction algorithm
"""
import json
from collections import namedtuple, OrderedDict
from typing import Tuple
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem, Draw
from mapchiral.mapchiral import encode, jaccard_similarity
import pandas as pd
import bcs
from retrotide import structureDB, designPKS, compareToTarget

# Helper functions
def mcs_search(mol_1: Chem.Mol, mol_2: Chem.Mol, chiral: bool) -> tuple[int, str]:
    """
    Performs a maximum common substructure search between two molecules

    Args:
        mol_1, mol2 (Chem.Mol): RDKit mol objects to compare
        chiral (bool): determines whether stereochemistry is considered

    Returns:
        atom_ct (int): MCS atom count
        smarts (str): MCS SMARTS string
    """
    mcs = rdFMCS.FindMCS(
        [mol_1, mol_2],
        timeout=10,
        matchValences=True,
        matchChiralTag=chiral,
        bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=False
        )
    atom_ct = mcs.numAtoms
    smarts = Chem.MolFromSmarts(mcs.smartsString)
    return atom_ct, smarts

def substructure_search(mol: Chem.Mol, pattern: str) -> tuple:
    """
    Identify substructure matches for a given molecule

    Args:
        mol (Chem.Mol): query molecule
        pattern (str): SMARTS search pattern

    Returns:
        matches (tuple): atom indices of substructure matches
    """
    pattern_mol = Chem.MolFromSmarts(pattern)
    matches = mol.GetSubstructMatches(pattern_mol)
    return matches

def lactone_size(mol: Chem.Mol, pattern_matches: tuple) -> tuple[int, tuple]:
    """
    Identify the size of the largest lactone in the target molecule

    Args:
        mol (Chem.Mol): query molecule
        pattern_matches (tuple): atom indices of ester substructure matches

    Returns:
        lactone_size (int): number of bonds in the largest lactone ring
        ring (tuple): atom indices of the lactone ring
    """
    ring_info = mol.GetRingInfo()
    lactone_sizes = []
    if ring_info.AtomRings():
        for ring in ring_info.AtomRings():
            if any(idx in ring for match in pattern_matches for idx in match):
                lactone_sizes.append((len(ring) - 1))
                ring = ring
            else:
                lactone_sizes.append(int(0))
    if lactone_sizes:
        return max(lactone_sizes), ring
    return 0, None

def get_mod_number(module_label: str) -> int:
    """
    Parses a module label to return a module number

    Args:
        module_label (str): module label (e.g. 'LM', 'M1')

    Returns:
        module_number (int): module number (e.g. 0, 1)
    """
    if module_label == 'LM':
        return 0
    return int(module_label.lstrip('M'))
    
# Obtain RetroTide results functions
def canonicalize_smiles(molecule_smi: str, stereo: str) -> str:
    """
    Standardizes a SMILES string and returns its canonical form with the
    specified isomeric information.

    Args:
        molecule_smi (str): The SMILES string of the target molecule
        stereo (str): The type of stereochemistry to specify

    Returns:
        target (str): The canonicalized target SMILES string
    """
    mol = Chem.MolFromSmiles(molecule_smi)
    if stereo == 'R/S':
        target = Chem.MolToSmiles(mol, isomericSmiles=True).replace('/', '').replace('\\', '')
    elif stereo == 'E/Z':
        target = Chem.MolToSmiles(mol, isomericSmiles=True).replace('@', '').replace('@@', '')
    elif stereo == 'none':
        target = Chem.MolToSmiles(mol, isomericSmiles=False)
    elif stereo == 'all':
        target = Chem.MolToSmiles(mol, isomericSmiles=True) 
    else:
        raise ValueError("Invalid stereo option. Choose from 'R/S', 'E/Z', 'none', or 'all'.")
    return target

def initial_pks(target_smi: str) -> tuple[list, Chem.Mol]:
    """
    Computes a PKS design to synthesize the 2D structure of the target molecule

    Args:
        target (str): The SMILES string of the target molecule

    Returns:
        pks_design (list): Initial RetroTide proposed PKS design
        mol (Chem.Mol): The computed PKS product of the PKS design
    """
    designs = designPKS(Chem.MolFromSmiles(target_smi),
                        maxDesignsPerRound = 500,
                        similarity = 'mcs_without_stereo')
    pks_design = designs[-1][0][0].modules
    mol = designs[-1][0][0].computeProduct(structureDB)
    return pks_design, mol

def get_design_features(pks_design: list) -> dict:
    """
    Constructs a dictionary containing information about the PKS design.

    Args:
        pks_design (list): Initial RetroTide design

    Returns:
        pks_features (dict): Dictionary with information about the intial PKS design,
        including module number, substrate, KR, DH, and ER subtypes.
    """
    pks_features = {
        'Module': [],
        'Substrate': [],
        'KR Type': [],
        'DH Type': [],
        'ER Type': [],
    }
    for idx, module in enumerate(pks_design):
        pks_features['Module'].append(idx)
        pks_features['Substrate'].append(module.domains[bcs.AT].substrate)
        if bcs.KR in module.domains:
            pks_features['KR Type'].append(module.domains[bcs.KR].type)
            if bcs.DH in module.domains:
                pks_features['DH Type'].append(module.domains[bcs.DH].type)
            else:
                pks_features['DH Type'].append('None')
            if bcs.ER in module.domains:
                pks_features['ER Type'].append(module.domains[bcs.ER].type)
            else:
                pks_features['ER Type'].append('None')
        else:
            pks_features['KR Type'].append('None')
            pks_features['DH Type'].append('None')
            pks_features['ER Type'].append('None')
    return pks_features

def te_offload(pks_product: Chem.Mol, mol: Chem.Mol, release_mechanism: str) -> tuple:
    """
    Models the TE mediated offloading of a PKS product from the ACP domain.
    For cyclization, ensures the correct lactone size is formed.

    Args:
        pks_product (Chem.Mol): The PKS bound product
        mol (Chem.Mol): The target molecule
        release_mechanism (str): 'thiolysis' or 'cyclization'

    Returns:
        unbound_mol (Chem.Mol, ): The unbound PKS product
    """
    Chem.SanitizeMol(pks_product)
    thioester_pattern = '[C:1](=[O:2])[S:3]'
    thioester_matches = substructure_search(pks_product, thioester_pattern)
    if thioester_matches:
        c_idx, o_idx, s_idx = thioester_matches[0]
        carbonyl_c_idx = thioester_matches[0][0]
    if not thioester_matches:
        raise ValueError('No thioester found. Cannot offload the PKS bound product.')

    if release_mechanism == 'thiolysis':
        editable_product = Chem.EditableMol(pks_product)
        editable_product.RemoveBond(c_idx, s_idx)
        editable_product.RemoveAtom(s_idx)
        acid_o_idx = editable_product.AddAtom(Chem.Atom('O'))
        editable_product.AddBond(c_idx, acid_o_idx, Chem.rdchem.BondType.SINGLE)
        unbound_mol = editable_product.GetMol()
        Chem.SanitizeMol(unbound_mol)
        return (unbound_mol,)

    if release_mechanism == 'cyclization':
        ester_pattern = '[C:1](=[O:2])[O:3][C:4]'
        ester_matches = substructure_search(mol, ester_pattern)
        target_size = lactone_size(mol, ester_matches)[0]
        if target_size != 0:
            hydroxyl_pattern = '[OH1]'
            hydroxyl_matches = substructure_search(pks_product, hydroxyl_pattern)
            if not hydroxyl_matches:
                print("Warning: No hydroxyl groups found. Cannot cyclize")
                return(pks_product,)
            for hydroxyl in hydroxyl_matches:
                hydroxyl_o_idx = hydroxyl[0]
                distance = len(Chem.GetShortestPath(
                    pks_product, carbonyl_c_idx, hydroxyl_o_idx)) - 1
                if distance == target_size:
                    editable_product = Chem.EditableMol(pks_product)
                    editable_product.RemoveBond(carbonyl_c_idx, s_idx)
                    editable_product.AddBond(carbonyl_c_idx, hydroxyl_o_idx,
                                             Chem.rdchem.BondType.SINGLE)
                    editable_product.RemoveAtom(s_idx)
                    unbound_mol = editable_product.GetMol()
                    Chem.SanitizeMol(unbound_mol)
                    return (unbound_mol,)
            raise ValueError(f"No hydroxyl group found at the target distance {target_size}. Cannot cyclize")
        raise ValueError("No valid lactone found in target molecule. Cannot cyclize.")
    if release_mechanism not in ['thiolysis', 'cyclization']:
        raise ValueError("Invalid release mechanism. Choose 'thiolysis' or 'cyclization'.")

def get_retro_tide_results(target: str, stereo: str, offload_mech: str) -> tuple[Chem.Mol, dict]:
    """
    Get the initial RetroTide results, offload and module map the corresponding
    PKS product, and extract design features from the PKS design.

    Args:
        target (str): The SMILES string of the target molecule
        stereo (str): The type of stereochemistry to specify
        offload_mech (str): 'thiolysis' or 'cyclization'

    Returns:
        unbound_mol (Chem.Mol): The unbound PKS product
        pks_features (dict): PKS design informaiton
    """
    target_smi = canonicalize_smiles(target, stereo)
    pks_design = initial_pks(target_smi)[0]
    mapped_product = module_map_product(pks_design)
    unbound_mol = te_offload(mapped_product, Chem.MolFromSmiles(target_smi), offload_mech)[0]
    pks_features = get_design_features(pks_design)
    return unbound_mol, pks_features

# PKS Precursor extraction functions
def pks_precursor_atoms(unbound_mol: Chem.Mol, mol: Chem.Mol) -> set:
    """
    Identify the PKS precursor if the entire target molecule is not synthesized via PKSs

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product
        mol (Chem.Mol): The target molecule

    Returns:
        precursor_atoms (set): Target molecule atom indices present in
                               the PKS product
    """
    mcs_atoms, mcs_smarts = mcs_search(unbound_mol, mol, False)
    if mcs_atoms > 0:
        precursor_atoms = set(mol.GetSubstructMatch(mcs_smarts))
    else:
        precursor_atoms = set()
    return precursor_atoms

def extract_precursor_mol(mol: Chem.Mol, precursor_atoms: set) -> Chem.Mol:
    """
    Remove target atoms and bonds not present in the PKS precursor

    Args:
        mol (Chem.Mol): The target molecule
        precursor_atoms (set): PKS precursor atom indices

    Returns:
        precursor_mol (Chem.Mol): The extracted PKS precursor molecule
    """
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetIntProp("original_idx", atom.GetIdx())
    editable_target = Chem.EditableMol(mol_copy)
    break_bonds = []
    for bond in mol_copy.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if (idx1 in precursor_atoms) != (idx2 in precursor_atoms):
            break_bonds.append((idx1, idx2))
    for idx1, idx2 in break_bonds:
        editable_target.RemoveBond(idx1, idx2)
    original_atoms = set(range(mol.GetNumAtoms()))
    remove_atoms = sorted(original_atoms - precursor_atoms, reverse=True)
    for atom_idx in remove_atoms:
        editable_target.RemoveAtom(atom_idx)
    precursor_mol = editable_target.GetMol()
    return precursor_mol

def get_pks_target(unbound_mol: Chem.Mol, mol: Chem.Mol) -> tuple[Chem.Mol, float]:
    """
    Sets the target molecule as the PKS precursor if the 2D MCS similarity between
    the RetroTide PKS product and the full target molecule is less than 100%.

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product
        mol (Chem.Mol): The target molecule

    Returns:
        precursor_mol (Chem.Mol): The PKS precursor target if needed
        mcs_score (float): The 2D MCS similarity score
    """
    mcs_score = compareToTarget(unbound_mol, mol, 'mcs_without_stereo')
    if mcs_score < 1.0:
        print(f'The PKS product only matches {mcs_score*100:.1f}% of the 2D target molecule.')
        print('Extracting the PKS precursor from the target molecule.')
        precursor_atoms = pks_precursor_atoms(unbound_mol, mol)
        precursor_mol = extract_precursor_mol(mol, precursor_atoms)
        return precursor_mol, mcs_score
    return mol, mcs_score

# Atom and module mapping functions
def module_map_product(pks_design: list) -> Chem.Mol:
    """
    Maps atoms of the PKS product to each module in the PKS design.

    Args:
        pks_design (list): The RetroTide proposed PKS design

    Returns:
        mapped_product (Chem.Mol): An atom-module mapped PKS product
    """
    cluster = bcs.Cluster(modules = pks_design)
    mapped_product = bcs.Cluster.create_atom_maps(cluster, structureDB)[-1]
    return mapped_product

def module_map(unbound_mol: Chem.Mol) -> pd.DataFrame:
    """
    Stores information about each atom in the PKS product

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product
    
    Returns:
        module_map_df (pd.DataFrame): DataFrame containing the PKS atom type,
                                      atom index, and PKS module index
    """
    module_map_df = pd.DataFrame(columns=['Atom Type', 'Product Atom Idx', 'Module Idx'])
    for i, atom in enumerate(unbound_mol.GetAtoms()):
        if not atom.HasProp('atomLabel'):
            module = '-'
        else:
            module = atom.GetProp('atomLabel')
        atom_entry = pd.DataFrame({
            'Atom Type': [atom.GetSymbol()],
            'Product Atom Idx': [atom.GetIdx()],
            'Module Idx': [module]
        },
        index = [i])
        module_map_df = pd.concat([module_map_df, atom_entry])
    return module_map_df

def atom_map(unbound_mol: Chem.Mol, mol: Chem.Mol) -> pd.DataFrame:
    """"
    Uses maximum common substructure search to map atoms of the offloaded
    PKS product to the target molecule

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product
        target_mol (Chem.Mol): The target molecule

    Returns:
        mcs_mapped_atoms_df (pd.DataFrame): DataFrame containing atom type, product
                                            atom index, and target atom index
    """
    atom_map_df = pd.DataFrame(columns=['Atom Type', 'Product Atom Idx', 'Target Atom Idx'])
    mcs_atoms, mcs_smarts = mcs_search(unbound_mol, mol, False)
    if mcs_atoms > 0:
        mol_1_match = unbound_mol.GetSubstructMatch(mcs_smarts)
        mol_2_match = mol.GetSubstructMatch(mcs_smarts)
        for i, (idx_1, idx_2) in enumerate(zip(mol_1_match, mol_2_match)):
            atom_entry = pd.DataFrame({
                'Atom Type': unbound_mol.GetAtomWithIdx(idx_1).GetSymbol(),
                'Product Atom Idx': idx_1,
                'Target Atom Idx': idx_2
            },
            index = [i])
            atom_map_df = pd.concat([atom_map_df, atom_entry], ignore_index=True)
    else:
        print("Warning: No common substructure found between the PKS product and target molecule")
        return atom_map_df
    return atom_map_df

def full_map(atom_mapping: pd.DataFrame, module_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Merge PKS product atom and module mapping

    Returns:
        full_map (pd.DataFrame): DataFrame containing all pertinent PKS product
                                 information: atom type, product atom index,
                                 corresponding target atom index, and PKS module index
    """
    full_map_df = pd.merge(
        atom_mapping,
        module_mapping,
        on=['Atom Type', 'Product Atom Idx']).dropna()
    return full_map_df

# Handling lactone alkene stereochemistry functions
def get_lactone_atoms(mol: Chem.Mol) -> tuple:
    """
    Identifies lactone atoms by searching for ester moieties in present ring
    structures.

    Args:
        mol (Chem.Mol): query molecule
    
    Returns:
        lactone_atoms (tuple): lactone ring atom indices
    """
    ester_matches = substructure_search(
        mol, '[C:1](=[O:2])[O:3][C:4]')
    return lactone_size(mol, ester_matches)[1]

def find_target_lactone(mol: Chem.Mol, full_map_df: pd.DataFrame) -> tuple:
    """
    Uses the atom mapping to identify target molecule lactone atoms.

    Args:
        mol (Chem.Mol): query molecule
        full_map_df (pd.DataFrame): atom and module mapping
    
    Returns:
        target_lactone_atoms (tuple): target lactone ring atom indices
    """
    target_lactone_atoms = set()
    lactone_atoms = get_lactone_atoms(mol)
    for atom in lactone_atoms:
        target_idx = full_map_df.loc[full_map_df['Product Atom Idx'] == atom, 'Target Atom Idx']
        target_lactone_atoms.add(int(target_idx))
    return tuple(target_lactone_atoms)

def force_target_lactone_alkene(mol: Chem.Mol, lactone_atoms: tuple) -> Chem.Mol:
    """
    If the target lactone is 6 membered or less, force the alkene to have Z stereochemistry.

    Args:
        mol (Chem.Mol): The target molecule
        lactone_atoms (tuple): target lactone ring atom indices
    
    Returns:
        mol (Chem.Mol): The target molecule with the set lactone Z alkene stereochemistry
    """
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    if len(lactone_atoms) <= 6:
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx in lactone_atoms and end_idx in lactone_atoms:
                    bond.SetStereo(Chem.BondStereo.STEREOZ)
    return mol

def force_pks_lactone_alkene(mol: Chem.Mol, lactone_atoms: tuple, full_map_df: pd.DataFrame, pks_features: dict) -> Chem.Mol:
    """
    Force PKS product lactone alkene stereochemistry based on the responsible DH subtype
    in the original PKS design.

    Args:
        mol (Chem.Mol): PKS product
        lactone_atoms (tuple): PKS product lactone ring atom indices
        full_map_df (pd.DataFrame): atom and module mapping
        pks_features (dict): RetroTide PKS design information
    
    Returns:
        mol (Chem.Mol): The PKS product with lactone alkene stereochemsitry representative of the
                        responsible PKS DH subtype
    """
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    if len(lactone_atoms) <= 6:
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx in lactone_atoms and end_idx in lactone_atoms:
                    begin_module = get_mod_number(
                        full_map_df.loc[full_map_df['Product Atom Idx'] == begin_idx, 'Module Idx'].values[0])
                    end_module = get_mod_number(
                        full_map_df.loc[full_map_df['Product Atom Idx'] == end_idx, 'Module Idx'].values[0])
                    target_module = begin_module if begin_module > end_module else end_module
                    dh_subtype = pks_features['DH Type'][target_module]
                    if dh_subtype == 'Z':
                        bond.SetStereo(Chem.BondStereo.STEREOZ)
                    elif dh_subtype == 'E':
                        bond.SetStereo(Chem.BondStereo.STEREOE)
    return mol

# Assess stereochemistry correspondence functions
ChiralCheckResult = namedtuple('ChiralCheckResult',
                               ['match1', 'match2', 'mmatch1', 'mmatch2', 'cc1', 'cc2'])

def get_rs_stereo_correspondence(unbound_mol: Chem.Mol, mol: Chem.Mol, full_map_df: pd.DataFrame) -> ChiralCheckResult:
    """
    Checks chirality correspondence of mapped PKS product and target molecule stereocenters.

    Args:
        unbound_mol (Chem.Mol): The TE offloaded PKS product
        mol (Chem.Mol): The target molecule
        full_map_df (pd.DataFrame): atom and module mapping

    Returns:
        ChiralCheckResult (namedtuple): 
            - match1, match2 (list): matching atom indices
            - mmatch1, mmatch2 (list): mismatching atom indices
            - cc1, cc2 (dict): chiral center indices and R/S stereochemistry labels
    """
    cc1 = dict(Chem.FindMolChiralCenters(unbound_mol, includeUnassigned=True))
    cc2 = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    match1, match2, mmatch1, mmatch2 = [], [], [], []
    product_atoms = full_map_df['Product Atom Idx'].astype(int).tolist()
    target_atoms = full_map_df['Target Atom Idx'].astype(int).tolist()
    for idx1, idx2 in zip(product_atoms, target_atoms):
        if idx1 in cc1 and idx2 in cc2:
            if cc1[idx1] == cc2[idx2]:
                match1.append(idx1)
                match2.append(idx2)
            else:
                mmatch1.append(idx1)
                mmatch2.append(idx2)
    return ChiralCheckResult(match1, match2, mmatch1, mmatch2, cc1, cc2)

def extract_ez_atoms(mol: Chem.Mol) -> Tuple[list, list]:
    """
    Identifies alkene stereocenters and assigns E/Z stereochemistry labels.

    Args:
        mol (Chem.Mol): query molecule
    
    Returns:
        z_bonds (list): stereocenter atom indices participating in a Z double bond
        e_bonds (list): stereocenter atom indices participating in an E double bond
    """
    z_bonds, e_bonds = [], []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if stereo == Chem.BondStereo.STEREOZ:
                z_bonds.extend([begin_idx, end_idx])
                bond.SetProp('bondNote', '(Z)')
            elif stereo == Chem.BondStereo.STEREOE:
                e_bonds.extend([begin_idx, end_idx])
                bond.SetProp('bondNote', '(E)')
    return z_bonds, e_bonds

AlkeneCheckResult = namedtuple('AlkeneCheckResult',
                                ['match1', 'match2', 'mmatch1', 'mmatch2'])

def get_ez_stereo_correspondence(unbound_mol: Chem.Mol, mol: Chem.Mol, full_map_df: pd.DataFrame) -> AlkeneCheckResult:
    """
    Checks alkene stereochemistry correspondence of mapped PKS product and target
    molecule stereocenters.

    Args:
        unbound_mol (Chem.Mol): The TE offloaded PKS product
        mol (Chem.Mol): The target molecule
        full_map_df (pd.DataFrame): atom and module mapping

    Returns:
        AlkeneCheckResult (namedtuple):
            - match1, match2 (list): matching atom indices
            - mmatch1, mmatch2 (list): mismatching atom indices
    """
    z_1, e_1 = extract_ez_atoms(unbound_mol)
    z_2, e_2 = extract_ez_atoms(mol)
    match1, match2, mmatch1, mmatch2 = [], [], [], []
    product_atoms = full_map_df['Product Atom Idx'].astype(int).tolist()
    target_atoms = full_map_df['Target Atom Idx'].astype(int).tolist()
    for idx1, idx2 in zip(product_atoms, target_atoms):
        if idx1 in z_1 + e_1 and idx2 in z_2 + e_2:
            if (idx1 in z_1 and idx2 in z_2) or (idx1 in e_1 and idx2 in e_2):
                match1.append(idx1)
                match2.append(idx2)
            else:
                mmatch1.append(idx1)
                mmatch2.append(idx2)
    return AlkeneCheckResult(match1, match2, mmatch1, mmatch2)

ProcessingResult = namedtuple('ProcessingResult',
                              ['unbound_mol', 'target_mol', 'mapping',
                               'chiral_result', 'alkene_result'])

def preprocessing(pks_features: dict, unbound_mol: Chem.Mol, target_mol: Chem.Mol) -> ProcessingResult:
    """
    Processes the RetroTide PKS design and product to hand off to the stereochemistry
    correction algorithm.

    Args:
        pks_features (dict): RetroTide PKS design information
        unbound_mol (Chem.Mol): The TE offloaded PKS product
        target_mol (Chem.Mol): The target molecule
    
    Returns:
        ProcessingResult (namedtuple):
            - unbound_mol (Chem.Mol): Processed TE offloaded PKS product
            - target_mol (Chem.Mol): Processed target molecule
            - mapping (pd.DataFrame): Atom and module mapping informaiton
            - chiral_result (ChiralCheckResult): R/S correspondence results
            - alkene_result (AlkeneCheckResult): E/Z correspondence results
    """
    module_mapped_df = module_map(unbound_mol)
    atom_mapped_df = atom_map(unbound_mol, target_mol)
    full_map_df = full_map(atom_mapped_df, module_mapped_df)
    lactone_atoms = get_lactone_atoms(unbound_mol)
    if lactone_atoms:
        target_lactone_atoms = find_target_lactone(unbound_mol, full_map_df)
        unbound_mol = force_pks_lactone_alkene(unbound_mol, lactone_atoms,
                                               full_map_df, pks_features)
        target_mol = force_target_lactone_alkene(target_mol, target_lactone_atoms)
    else:
        Chem.AssignStereochemistry(unbound_mol, force=True, cleanIt=True)
        Chem.AssignStereochemistry(target_mol, force=True, cleanIt=True)
    chiral_result = get_rs_stereo_correspondence(unbound_mol, target_mol, full_map_df)
    alkene_result = get_ez_stereo_correspondence(unbound_mol, target_mol, full_map_df)
    return ProcessingResult(unbound_mol, target_mol, full_map_df, chiral_result, alkene_result)

# Identify stereochemistry mismatch case functions
def extract_carbon_pairs(mol: Chem.Mol, full_map_df: pd.DataFrame) -> list:
    """
    Identifies backbone carbons and their corresponding PKS module indices.

    Args:
        mol (Chem.Mol): The TE offloaded PKS product
        full_map_df (pd.DataFrame): atom and module mapping

    Returns:
        backbone_pairs (list): backbone carbon pairs mapped to sequential PKS modules
    """
    carbons = {}
    for _, row in full_map_df.iterrows():
        if row['Atom Type'] == 'C':
            c_idx = int(row['Product Atom Idx'])
            carbons[c_idx] = row['Module Idx']
    
    backbone_pairs = []
    for c_idx in carbons.keys():
        c_atom = mol.GetAtomWithIdx(c_idx)
        for neighbor in c_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() in carbons.keys():
                neighbor_mod = carbons[neighbor.GetIdx()]
                if neighbor_mod != carbons[c_idx]:
                    pair = tuple(sorted([(c_idx, carbons[c_idx]),
                                         (neighbor.GetIdx(),
                                          neighbor_mod)]))
                    if pair not in backbone_pairs:
                        backbone_pairs.append(pair)
    return backbone_pairs

def address_undefined_stereo(pairs: list, chiral_result: ChiralCheckResult, full_map_df: pd.DataFrame) -> set:
    """
    Identifies target molecule stereocenters with undefined stereochemistry ('?') and
    the corresponding PKS module(s).

    Args:
        pairs (list): backbone carbon pairs mapped to sequential PKS modules
        chiral_result (ChiralCheckResult): R/S correspondence results
        full_map_df (pd.DataFrame): atom and module mapping

    Returns:
        undefined_cc_mods (set): PKS modules mapped to undefined target stereochemistry
    """
    undefined_cc_mods = set([])
    for idx, cip in chiral_result.cc2.items():
        if cip == '?':
            product_idx = full_map_df.loc[full_map_df['Target Atom Idx'] == idx,
                                       'Product Atom Idx'].values[0]
            for pair in pairs:
                if product_idx in (pair[0][0], pair[1][0]):
                    m_i = get_mod_number(pair[0][1])
                    m_i_1 = get_mod_number(pair[1][1])
                    undefined_cc_mods.add(m_i if m_i > m_i_1 else m_i_1)
    return undefined_cc_mods

def identify_pairs_with_mismatches(backbone_pairs: list, mmatch1: list) -> list:
    """
    Cross references backbone pairs with steroechemistry correspondence results.

    Args:
        backbone_pairs (list): backbone carbon pairs mapped to sequential PKS modules
        mmatch1 (list): PKS product atom indices that have mismatched stereochemistry
                        relative to the target molecule
    Returns:
        mismatch_pairs (list): outputs backbone pairs that include at least one
        stereochemistry mismatch
    """
    mismatch_pairs = []
    for pair in backbone_pairs:
        if pair[0][0] in mmatch1 or pair[1][0] in mmatch1:
            mismatch_pairs.append(pair)
    return mismatch_pairs

def check_alpha_carbon_mismatch(mol: Chem.Mol, pair: tuple, mmatch1: list) -> dict:
    """
    Looks for alpha carbons using a substructure search. Updates each pair with
    whether the alpha carbon has mismatched stereochemistry with the target molecule.

    Args:
        mol (Chem.Mol): The TE offloaded PKS product
        pair (tuple): backbone carbon pair
        mmatch1 (list): PKS product atom indices that have mismatched stereochemistry

    Returns:
        substructure_results (dict): updated atom information indicating whether the
                                     alpha carbon of the pair is mismatched
    """
    alpha_carbon_patterns = [
        '[C:1][C:2](=[O:3])',
        '[C:1][C:2][OH:3]',
        '[C:1][C:2][O:3]',
        '[C:1][C:2](=[O:3])[O:4][C:5]',
        '[C:1]/[C:2]=[C:3]/[C:4]',
        '[C:1]/[C:2]=[C:3]\[C:4]',
        '[C:1]([CH2:2])[CH2:3]'
    ]
    substructure_results = {'atom1': False, 'atom2': False}
    for pattern in alpha_carbon_patterns:
        matches = substructure_search(mol, pattern)
        for match in matches:
            if pair[0][0] == match[0] and pair[0][0] in mmatch1:
                substructure_results.update({
                    'atom1': True
                })
            if pair[1][0] == match[0] and pair[1][0] in mmatch1:
                substructure_results.update({
                    'atom2': True
                })
    return substructure_results

def check_beta_carbon_mismatch(mol: Chem.Mol, pair: tuple, mmatch1: list) -> dict:
    """
    Looks for beta carbons using a substructure search. Updates each pair with
    whether the beta carbon has mismatched stereochemistry with the target molecule.

    Args:
        mol (Chem.Mol): The TE offloaded PKS product
        pair (tuple): backbone carbon pair
        mmatch1 (list): PKS product atom indices that have mismatched stereochemistry
    
    Returns:
        substructure_results (dict): updated atom information indicating whether the beta
                                     carbon of the pair is mismatched
    """
    beta_carbon_patterns = [
        '[C:1][OH:2]',
        '[C:1][O:2][C:3](=[O:4])'
    ]
    substructure_results = {'atom1': False, 'atom2': False}
    for pattern in beta_carbon_patterns:
        matches = substructure_search(mol, pattern)
        for match in matches:
            if pair[0][0] == match[0] and pair[0][0] in mmatch1:
                substructure_results.update({
                    'atom1': True
                })
            if pair[1][0] == match[0] and pair[1][0] in mmatch1:
                substructure_results.update({
                    'atom2': True
                })
    return substructure_results

def check_alkene_mismatch(mol: Chem.Mol, pair: tuple, alkene_mmatch1: list) -> dict:
    """
    Looks for carbons present in a double bond using a substructure search. Updates each
    pair with whether the alkene carbons have mismatched stereochemistry with the target molecule.

    Args:
        mol (Chem.Mol): The TE offloaded PKS product
        pair (tuple): backbone carbon pair
        alkene_mmatch1 (list): PKS product atom indices that have mismatched alkene stereochemistry
    
    Returns:
        substructure_results (dict): updated atom information indicating whether the pair is an
                                     alkene mismatch
    """
    alkene_pattern = '[C:1]=[C:2]'
    substructure_results = {'atom1': False, 'atom2': False}
    matches = substructure_search(mol, alkene_pattern)
    for match in matches:
        if pair[0][0] == match[0] and pair[0][0] in alkene_mmatch1:
            substructure_results.update({
                'atom1': True
            })
        if pair[1][0] == match[0] and pair[1][0] in alkene_mmatch1:
            substructure_results.update({
                'atom2': True
            })
    return substructure_results

def check_cc_mismatch_cases(mol: Chem.Mol, carbon_pairs: list, cc_mmatch1: list) -> list:
    """
    Assesses each backbone pair that has at least one stereochemistry mismatch and updates
    whether the alpha and/or beta carbons are mismatched.

    Args:
        mol (Chem.Mol): The TE offloaded PKS product
        carbon_pairs (list): backbone carbon pairs
        cc_mmatch1 (list): PKS product atom indices that have mismatched stereochemistry
    
    Returns:
        cc_mismatch_results (list): chiral mismatch information for each pair
    """
    cc_mismatch_pairs = identify_pairs_with_mismatches(carbon_pairs, cc_mmatch1)
    cc_mismatch_results = []
    for pair in cc_mismatch_pairs:
        pair_info = {
            'c_idx_i': pair[0][0],
            'module_i': pair[0][1],
            'c_idx_i+1': pair[1][0],
            'module_i+1': pair[1][1],
            'alpha_mismatch' : False,
            'beta_mismatch' : False,
            }
        alpha_results = check_alpha_carbon_mismatch(mol, pair, cc_mmatch1)
        beta_results = check_beta_carbon_mismatch(mol, pair, cc_mmatch1)
        if (alpha_results['atom1'] is True) or (alpha_results['atom2'] is True):
            pair_info.update({
                'alpha_mismatch': True
            })
        if (beta_results['atom1'] is True) or (beta_results['atom2'] is True):
            pair_info.update({
                'beta_mismatch': True
            })
        cc_mismatch_results.append(pair_info)
    return cc_mismatch_results

def check_alkene_mismatch_cases(mol: Chem.Mol, carbon_pairs: list, alkene_mmatch1: list) -> list:
    """
    Assess each backbone pair for an alkene stereochemistry mismatch and updates the
    pair information accordingly.

    Args:
        mol (Chem.Mol): The TE offloaded PKS product
        carbon_pairs (list): backbone carbon pairs
        alkene_mmatch1 (list): PKS product atom indices that have mismatched alkene stereochemistry

    Returns:
        alkene_mismatch_results (list): alkene mismatch information for each pair
    """
    alkene_mismatch_pairs = identify_pairs_with_mismatches(carbon_pairs, alkene_mmatch1)
    alkene_mismatch_results = []
    for pair in alkene_mismatch_pairs:
        pair_info = {
            'c_idx_i': pair[0][0],
            'module_i': pair[0][1],
            'c_idx_i+1': pair[1][0],
            'module_i+1': pair[1][1],
            'alkene_mismatch': False
        }
        alkene_results = check_alkene_mismatch(mol, pair, alkene_mmatch1)
        if (alkene_results['atom1'] is True) or (alkene_results['atom2'] is True):
            pair_info.update({
                'alkene_mismatch': True
            })
        alkene_mismatch_results.append(pair_info)
    return alkene_mismatch_results

def get_cc_mismatch_results(mol: Chem.Mol, full_map_df: pd.DataFrame, cc_result: ChiralCheckResult) -> tuple[list, list]:
    """
    Extracts R/S stereochemistry correspondence results and identifies whether the
    alpha and/or beta carbons are mismatched for each backbone carbon pair. Also
    identifies any undefined stereochemistry in the target molecule and identifies
    the corresponding PKS module(s).

    Args:
        mol (Chem.Mol): The TE offloaded PKS product
        full_map_df (pd.DataFrame): atom and module mapping
        cc_result (ChiralCheckResult): R/S correspondence results

    Returns:
        mismatch_results (list): alpha and beta carbon mismatch infromation for each pair
        type_u_mods (list): modules mapped to target undefined stereocenters
    """
    pairs = extract_carbon_pairs(mol, full_map_df)
    type_u_mods = address_undefined_stereo(pairs, cc_result, full_map_df)
    mismatch_results = check_cc_mismatch_cases(mol, pairs, cc_result.mmatch1)
    return mismatch_results, type_u_mods

def get_alkene_mismatch_results(mol: Chem.Mol, full_map_df: pd.DataFrame, alkene_result) -> list:
    """
    Extracts E/Z stereochemistry correspondence results and identifies whether the alkene carbons
    are mismatched for each backbone carbon pair.

    Args:
        mol (Chem.Mol): The TE offloaded PKS product
        full_map_df (pd.DataFrame): atom and module mapping
        alkene_result (AlkeneCheckResult): E/Z correspondence results
    
    Returns:
        mismatch_results (list): alkene carbon mismatch information for each backbone carbon pair
    """
    pairs = extract_carbon_pairs(mol, full_map_df)
    mismatch_results = check_alkene_mismatch_cases(mol, pairs, alkene_result.mmatch1)
    return mismatch_results

# Employ reductive loop swaps functions
def kr_type_logic(pks_features: dict, target_mod: int, alpha: bool, beta: bool) -> str:
    """
    Assesses each mismatched pair and fits to one of the KR swap cases, to then logically
    make the appropriate KR swap based on the original KR subtype.

    Args:
        pks_features (dict): RetroTide PKS design information
        target_mod (int): PKS module index to make the KR swap in
        alpha (bool): whether the alpha carbon is mismatched
        beta (bool): whether the beta carbon is mismatched
    
    Returns:
        new_kr_type (str): The new KR subtype determined by the algorithm KR Swaps for the
                           target module
    """
    old_kr_type = pks_features['KR Type'][target_mod]
    substrate = pks_features['Substrate'][target_mod]
    new_kr_type = None
    # Single cases
    if (substrate == 'Malonyl-CoA' and pks_features['ER Type'][target_mod] == 'None'):
        if old_kr_type == 'A':
            new_kr_type = 'B'
        else:
            new_kr_type = 'A'
        print('    Case 1: Malonyl-CoA has no alpha substituent. Only beta is wrong -> swapping letter only')
    elif (substrate == 'Malonyl-CoA' and pks_features['ER Type'][target_mod] != 'None'):
        new_kr_type = 'B'
    elif (substrate == 'Methylmalonyl-CoA' and pks_features['DH Type'][target_mod] != 'None'):
        new_kr_type = 'B1'
    # Complex cases
    elif old_kr_type == 'None':
        if alpha:
            new_kr_type = 'C2'
            print('    Case 5: Adding KR type C2 to perform epimerization of the alpha chiral center')
        else:
            print('    No KR domain and not an alpha carbon mismatch -> cannot fix by KR swap')
    else:
        letter = old_kr_type[0]
        number = old_kr_type[1:] if len(old_kr_type) >= 2 else '1'
        if alpha and beta:
            new_letter = 'B' if letter == 'A' else 'A'
            new_number = '2' if number == '1' else '1'
            new_kr_type = new_letter + new_number
            print('    Case 4: Both are wrong -> swapping both letter and number')
        elif alpha and not beta:
            new_number = '2' if number == '1' else '1'
            new_kr_type = letter + new_number
            print('    Case 3: Only alpha is wrong -> swapping number only')
        elif not alpha and beta:
            new_letter = 'B' if letter == 'A' else 'A'
            new_kr_type = new_letter + number
            print('    Case 2: Only beta is wrong -> swapping letter only')
        else:
            new_kr_type = old_kr_type
            print('    No case found -> returning the original KR type')
    return new_kr_type

def er_type_logic(pks_features: dict, target_mod: int) -> str:
    """
    Assesses remaining alpha carbon R/S stereochemistry mismatches post KR swaps, and swaps
    the ER subtype if possible.

    Args:
        pks_features (dict): RetroTide PKS design information
        target_mod (int): PKS module index to make the ER swap in
    
    Retruns:
        new_er_type (str): The new ER subtype determined by the KR Swaps algorithm for the
                           target module
    """
    if pks_features['ER Type'][target_mod] != 'None':
        old_er_type = pks_features['ER Type'][target_mod]
        new_er_type = None
        substrate = pks_features['Substrate'][target_mod]
        if substrate != 'Malonyl-CoA':
            new_er_type = 'D' if old_er_type == 'L' else 'L'
            print('   Case 7: Swapping ER subtype')
        else:
            return old_er_type # Ignore the ER subtype as malonyl-CoA has no alpha chiral center
        return new_er_type
    print(f'   No ER domain present in {target_mod} -> cannot fix mismatch by ER swap')
    return 'None'

def kr_dh_type_logic(pks_features: dict, target_mod: int) -> tuple[str, str]:
    """
    Assesses E/Z alkene stereochemistry mismatches, and swaps the DH-KR reductive loop
    subtypes accordingly.

    Args:
        pks_features (dict): RetroTide PKS design information
        target_mod (int): PKS module index to make the DH-KR swap in
    
    Returns:
        new_kr_type (str): the new KR subtype determined by the KR Swaps algorithm for the
                           target module while adhering to DH-KR subtype compatibility
        new_dh_type (str): the new DH subtype determined by the KR Swaps algorithm for the
                          target module while adhering to DH-KR subtype compatibility
    """
    old_dh_type = pks_features['DH Type'][target_mod]
    new_kr_type, new_dh_type = None, None
    substrate = pks_features['Substrate'][target_mod]
    if substrate != 'Malonyl-CoA':
        new_kr_type, new_dh_type = 'B1', 'E'
        print('    PKS biosynthesis incompatability. Cannot synthesize a Z alkene when the AT substrate has an alpha substituent')
    else:
        print('    Case 6: Swapping DH-KR subtypes')
        if old_dh_type == 'Z':
            new_kr_type, new_dh_type = 'B', 'E'
        else:
            new_kr_type, new_dh_type = 'A', 'Z'
    return new_kr_type, new_dh_type

def identify_kr_swap_case(pks_features: dict, result: dict) -> dict:
    """
    Uses the KR type logic to make the correct KR subtype swap in the higher ordered
    module of a given mismatched pair.

    Args:
        pks_features (dict): RetroTide PKS design information
        result (dict): mismatch information for a given backbone carbon pair
    
    Returns:
        pks_features (dict): PKS design information with the updated KR subtypes proposed by
                             the KR Swaps algorithm
    """
    target_mod = get_mod_number(result['module_i+1'])
    old_kr_type = pks_features['KR Type'][target_mod]
    print(f'----Analyzing mismatch(s) in pair ({result["c_idx_i"]}, {result["c_idx_i+1"]})----')
    print(f'    alpha_mismatch: {result["alpha_mismatch"]}, beta_mismatch: {result["beta_mismatch"]}')
    new_kr_type = kr_type_logic(pks_features, target_mod,
                                alpha=result['alpha_mismatch'], beta=result['beta_mismatch'])
    if new_kr_type != old_kr_type:
        pks_features['KR Type'][target_mod] = new_kr_type
        print(f'    Making KR swap in module {target_mod}: Type {old_kr_type} -> Type {new_kr_type}')
    return pks_features

def identify_er_swap_case(pks_features: dict, result: dict) -> dict:
    """
    Uses the ER type logic to make the correct ER subtype swap in the higher ordered
    module of a given mismatched pair.

    Args:
        pks_features (dict): RetroTide PKS design information
        result (dict): mismatch information for a given backbone carbon pair
    
    Returns:
        pks_features (dict): PKS design information with the updated ER subtypes proposed by
                             the KR Swaps algorithm
    """
    target_mod = get_mod_number(result['module_i+1'])
    old_er_type = pks_features['ER Type'][target_mod]

    print(f'----Analyzing mismatch(s) in pair ({result["c_idx_i"]}, {result["c_idx_i+1"]})----')
    print(f'    alpha_mismatch: {result["alpha_mismatch"]}, beta_mismatch: {result["beta_mismatch"]}')
    if result['alpha_mismatch'] is True:
        new_er_type = er_type_logic(pks_features, target_mod)
        pks_features['ER Type'][target_mod] = new_er_type
        print(f'    Making ER swap in module {target_mod}: Type {old_er_type} -> Type {new_er_type}')
    return pks_features

def identify_dh_swap_case(pks_features: dict, result: dict) -> dict:
    """
    Uses the DH-KR type logic to make the correct DH-KR subtype swap in the higher ordered
    module of a given mismatched pair.

    Args:
        pks_features (dict): RetroTide PKS design information
        result (dict): mismatch information for a given backbone carbon pair
    
    Returns:
        pks_features (dict): PKS design information with the updated DH-KR subtypes proposed by
                             the KR Swaps algorithm
    """
    target_mod = get_mod_number(result['module_i+1'])
    old_kr_type = pks_features['KR Type'][target_mod]
    old_dh_type = pks_features['DH Type'][target_mod]
    
    print(f'----Analyzing alkene mismatch(s) in pair ({result["c_idx_i"]}, {result["c_idx_i+1"]})----')
    print(f'    alkene mismatch: {result["alkene_mismatch"]}')
    new_kr_type, new_dh_type = kr_dh_type_logic(pks_features, target_mod)
    pks_features['KR Type'][target_mod] = new_kr_type
    pks_features['DH Type'][target_mod] = new_dh_type
    print(f'    Making DH-KR swap in module {target_mod}: {old_dh_type}-{old_kr_type} -> {new_dh_type}-{new_kr_type}')
    return pks_features

def kr_swaps(pks_features: dict, mismatch_results: list) -> dict:
    """
    Assesses each mismatched pair and employs KR domain swaps to perform an R/S stereochemistry
    correction.

    Args:
        pks_features (dict): RetroTide PKS design information
        mismatch_results (list): alpha and beta carbon mismatch information for each backbone
                                 carbon pair
    
    Returns:
        pks_features (dict): The updated PKS design post KR domain swaps
    """
    for result in mismatch_results:
        identify_kr_swap_case(pks_features, result)
    return pks_features

def er_swaps(pks_features: dict, mismatch_results: list) -> dict:
    """
    Assess each remaining mismatched pair after making KR domain swaps, and employs
    ER domain swaps to perform epimerization of the alpha carbon.

    Args:
        pks_features (dict): RetroTide PKS design information
        mismatch_results (list): alpha and beta carbon mismatch information for each
                                 backbone carbon pair
    
    Returns:
        pks_features (dict): The updated PKS design post ER domain swaps
    """
    for result in mismatch_results:
        identify_er_swap_case(pks_features, result)
    return pks_features

def dh_swaps(pks_features: dict, mismatch_results: list) -> dict:
    """
    Assesses each alkene mismatched pair and employs DH-KR domain swaps to perform
    an E/Z stereochemistry correction.

    Args:
        pks_features (dict): RetroTide PKS design information
        mismatch_results (list): alkene carbon mismatch information for each backbone
                                 carbon pair
    
    Returns:
        pks_features (dict): The updated PKS design post DH-KR domain swaps
    """
    for result in mismatch_results:
        identify_dh_swap_case(pks_features, result)
    return pks_features

def new_pks_design(pks_features: dict) -> tuple[Chem.Mol, list]:
    """
    Reconstructs a PKS design based on the updated KR, KR-DH, and ER subtypes proposed
    by the KR Swaps stereochemistry correction algorithm.

    Args:
        pks_features (dict): RetroTide PKS design information

    Returns:
        pks_product (Chem.Mol): The final PKS product after stereocorrections
        pks_design (list): The final PKS design after stereocorrections
    """
    pks_design = []
    for idx, substrate in enumerate(pks_features['Substrate']):
        substrate = pks_features['Substrate'][idx]
        if idx == 0:
            domains_dict = OrderedDict({bcs.AT: bcs.AT(active=True, substrate=substrate)})
            module = bcs.Module(domains=domains_dict, loading=True)
            pks_design.append(module)
        else:
            at = bcs.AT(active=True, substrate=substrate)
            domains_dict = OrderedDict({bcs.AT: at})
            if pks_features['KR Type'][idx] != 'None':
                domains_dict.update({bcs.KR: bcs.KR(active=True,
                                                    type=pks_features['KR Type'][idx])})
            if pks_features['DH Type'][idx] != 'None':
                domains_dict.update({bcs.DH: bcs.DH(active=True,
                                                    type=pks_features['DH Type'][idx])})
            if pks_features['ER Type'][idx] != 'None':
                domains_dict.update({bcs.ER: bcs.ER(active=True,
                                                    type=pks_features['ER Type'][idx])})
            module = bcs.Module(domains=domains_dict, loading=False)
            pks_design.append(module)
    cluster_f = bcs.Cluster(modules=pks_design)
    pks_product = cluster_f.computeProduct(structureDB)
    return pks_product, pks_design

def insert_unknown_types(pks_features: dict, type_u_modules: set) -> dict:
    """
    Replaces the KR subtype of modules mapped to target stereocenters with undefined
    R/S stereochemistry with 'U' to indicate unknown.

    Args:
        pks_features (dict): RetroTide PKS design information
        type_u_modules (set): PKS modules mapped to target stereocenters with undefined
                              stereochemistry
    
    Returns:
        pks_features (dict): The PKS design information reflecting these changes
    """
    for mod_idx in type_u_modules:
        pks_features['KR Type'][mod_idx] = 'U'
    return pks_features

def parse_final_design(pks_design_features: dict) -> list:
    """
    Processes the stereochemsitry corrected PKS design into an easily
    readable format.

    Args:
        pks_design_features (dict): RetroTide PKS design information
    
    Returns:
        parsed_design (list): The final PKS design. For each module, includes:
                                - Module number
                                - AT substrate
                                - Reductive loop composition
    """
    parsed_design = []
    for idx in pks_design_features['Module']:
        reductive_loop = str('')
        if pks_design_features['ER Type'][idx] != 'None':
            reductive_loop += (f'DH({pks_design_features["DH Type"][idx]})')
            reductive_loop += (f'-ER({pks_design_features["ER Type"][idx]})')
            reductive_loop += (f'-KR({pks_design_features["KR Type"][idx]})')
        elif pks_design_features['DH Type'][idx] != 'None':
            reductive_loop += (f'DH({pks_design_features["DH Type"][idx]})')
            reductive_loop += (f'-KR({pks_design_features["KR Type"][idx]})')
        elif pks_design_features['KR Type'][idx] != 'None':
            reductive_loop += (f'KR({pks_design_features["KR Type"][idx]})')
        else:
            reductive_loop = None
        entry = [f'M{idx}', f'AT Substrate: {pks_design_features["Substrate"][idx]}',
                 f'Reductive Loop: {reductive_loop}']
        parsed_design.append(entry)
    return parsed_design

CorrectionResult = namedtuple('CorrectionResult',
                                ['pks_product', 'pks_design', 'target', 'mapping',
                                 'chiral_result', 'alkene_result'])

def postprocessing(pks_features_updated: dict, target_mol: Chem.Mol, offload_mech: str, type_u_mods: set) -> CorrectionResult:
    """
    Processess the PKS design post KR Swaps stereochemistry correction. Outputs the final design
    and corresponding PKS product. Determines the R/S and E/Z stereochemistry correspondence
    between the PKS product and target molecule post correction.

    Args:
        pks_features_updated (dict): RetroTide PKS design information post KR Swaps corrections
        target_mol (Chem.Mol): The target molecule
        offload_mech (str): TE offloading mechanism
        type_u_mods (set): PKS modules mapped to target stereocenters with undefined
                           stereochemistry
    
    Returns:
        CorrectionResult (namedtuple):
        - pks_product (Chem.Mol): The final PKS product after stereocorrections
        - pks_design (list): The final PKS design after stereocorrections
        - target (Chem.Mol): The target molecule
        - mapping (pd.DataFrame): atom and module mapping
        - chiral_result (ChiralCheckResult): R/S correspondence results post correction
        - alkene_result (AlkeneCheckResult): E/Z correspondence results post correction
    """
    new_design = new_pks_design(pks_features_updated)[1]
    new_design_features = get_design_features(new_design)
    if type_u_mods:
        new_design_features = insert_unknown_types(new_design_features, type_u_mods)
    design_result = parse_final_design(new_design_features)
    mapped_product = module_map_product(new_design)
    new_pks_product = te_offload(mapped_product, target_mol, offload_mech)[0]
    pp_result = preprocessing(pks_features_updated, new_pks_product, target_mol)
    return CorrectionResult(pp_result.unbound_mol, design_result, pp_result.target_mol,
                            pp_result.mapping, pp_result.chiral_result, pp_result.alkene_result)

def kr_swaps_algorithm(unbound_mol: Chem.Mol, target_mol: Chem.Mol, full_map_df: pd.DataFrame, pks_features: dict, offload_mech: str, chiral_result, alkene_result):
    """
    Main function to execute KR, DH-KR, and ER domain swaps to correct both the R/S and E/Z
    stereochemistry to best match that of the target molecule.

    Args:
        unbound_mol (Chem.Mol): The TE offloaded PKS product
        target_mol (Chem.Mol): The target molecule
        full_map (pd.DataFrame): atom and module mapping
        pks_features (dict): RetroTide PKS design information
        offload_mech (str): TE offloading mechanism
        chiral_result (ChiralCheckResult): R/S correspondence results
        alkene_result (AlkeneCheckResult): E/Z correspondence results
    
    Returns:
        CorrectionResult (namedtuple): PKS product and PKS design information post
                                       stereochemistry correction
    """
    print('Correcting R/S stereochemistry')
    cc_mismatch_results, type_u_mods = get_cc_mismatch_results(unbound_mol,
                                                               full_map_df,
                                                               chiral_result)
    pks_features_updated = kr_swaps(pks_features, cc_mismatch_results)

    print('Correcting E/Z stereochemistry')
    alkene_mismatch_results = get_alkene_mismatch_results(unbound_mol,
                                                          full_map_df,
                                                          alkene_result)
    pks_features_updated = dh_swaps(pks_features_updated, alkene_mismatch_results)

    krs_result = postprocessing(pks_features_updated, target_mol, offload_mech, type_u_mods)
    if krs_result.chiral_result.mmatch1:
        print('Correcting remianing R/S mismatches')
        cc_mismatch_results_f, type_u_mods = get_cc_mismatch_results(
            krs_result.pks_product, krs_result.mapping, krs_result.chiral_result)
        pks_features_updated = er_swaps(pks_features_updated, cc_mismatch_results_f)
        return postprocessing(pks_features_updated, target_mol, offload_mech, type_u_mods)
    return krs_result

# Post processing functions
def add_atom_labels(mol: Chem.Mol, chiral_centers: dict) -> Chem.Mol:
    """
    Adds atom notes to specify the atom type, index, and R/S stereochemistry
    label if appropriate.

    Args:
        mol (Chem.Mol): mol to add labels to
        chiral_centers (dict): stereocenter indices and their labels
    
    Returns:
        mol (Chem.Mol): labelled mol
    """
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_label = f'{atom.GetSymbol()}:{atom_idx}'
        if atom_idx in chiral_centers:
            cip_label = f'{atom_label} ({chiral_centers[atom_idx]})'
        else:
            cip_label = atom_label
        atom.SetProp('atomNote', cip_label)
    return mol

def visualize_stereo_correspondence(mol1: Chem.Mol, mol2: Chem.Mol, chiral_result, alkene_result):
    """
    Visualize the stereochemistry correspondence between two molecules.

    Args:
        mol1, mol2 (Chem.Mol): query mols
        chiral_result (ChiralCheckResult): R/S correspondence results
        alkene_result (AlkeneCheckResult): E/Z correspondence results
    
    Returns:
        img (SVG): an SVG depicting the stereochemistry correspondence between the two molecules
    """
    add_atom_labels(mol1, chiral_result.cc1)
    add_atom_labels(mol2, chiral_result.cc2)

    highlight_1, highlight_2 = {}, {}

    for atom_idx in chiral_result.match1:
        highlight_1[atom_idx] = (0, 1, 1)
    for atom_idx in chiral_result.match2:
        highlight_2[atom_idx] = (0, 1, 1)
    for atom_idx in chiral_result.mmatch1:
        highlight_1[atom_idx] = (1, 0, 0)
    for atom_idx in chiral_result.mmatch2:
        highlight_2[atom_idx] = (1, 0, 0)
    for atom_idx in alkene_result.match1:
        highlight_1[atom_idx] = (0, 1, 1)
    for atom_idx in alkene_result.match2:
        highlight_2[atom_idx] = (0, 1, 1)
    for atom_idx in alkene_result.mmatch1:
        highlight_1[atom_idx] = (1, 0, 0)
    for atom_idx in alkene_result.mmatch2:
        highlight_2[atom_idx] = (1, 0, 0)
 
    all_1 = chiral_result.match1 + chiral_result.mmatch1 + alkene_result.match1 + alkene_result.mmatch1
    all_2 = chiral_result.match2 + chiral_result.mmatch2 + alkene_result.match2 + alkene_result.mmatch2
    all_bonds_1 = set(alkene_result.match1 + alkene_result.mmatch1)
    all_bonds_2 = set(alkene_result.match2 + alkene_result.mmatch2)

    bond_indices_1 = [bond.GetIdx() for bond in mol1.GetBonds()
                      if bond.GetBeginAtomIdx() in all_bonds_1
                      and bond.GetEndAtomIdx() in all_bonds_1]
    bond_indices_2 = [bond.GetIdx() for bond in mol2.GetBonds()
                      if bond.GetBeginAtomIdx() in all_bonds_2
                      and bond.GetEndAtomIdx() in all_bonds_2]
    
    AllChem.Compute2DCoords(mol1)
    AllChem.GenerateDepictionMatching2DStructure(mol2, mol1, acceptFailure=True)

    svg = Draw.MolsToGridImage([mol1, mol2], legends=['PKS Product', 'Target'],
                                molsPerRow=2, highlightAtomLists=[all_1, all_2],
                                highlightAtomColors=[highlight_1, highlight_2],
                                highlightBondLists=[bond_indices_1, bond_indices_2],
                                useSVG=True,
                                subImgSize=(500, 400))
    
    svg = svg.replace(
        "width='500.0' height='400.0' x='0.0' y='0.0'",
        "width='1000.0' height='400.0' x='0.0' y='0.0'"
    )
    return svg

def compute_jaccard_sim(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute the Jaccard similarity between two molecules.
    Uses MAP4C fingerprints: Min-hashed Atom Pair Chiral Fingerprints
                             (https://github.com/reymond-group/mapchiral)

    Args:
        mol1, mol2 (Chem.Mol): query molecules
    
    Returns:
        jaccard_sim (float): Jaccard similarity between the two molecules
    """
    mol1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol1))
    mol2 = Chem.MolFromSmiles(Chem.MolToSmiles(mol2))
    fp1 = encode(mol1, max_radius=2, n_permutations=2048, mapping=False)
    fp2 = encode(mol2, max_radius=2, n_permutations=2048, mapping=False)
    return jaccard_similarity(fp1, fp2)

# Run stereochemistry correction
def krswaps_stereo_correction(target_smi: str, stereo: str, offload_mech: str):
    """
    Main function to run RetroTide with a post processing stereochemistry correction using the
    KR Swaps algorithm.

    Args:
        target_smi (str): The user-defined SMILES string of the target molecule
        stereo (str): The user-defined level of molecule stereochemistry to consider
        offload_mech (str): The user-defined TE offloading mechanism
    
    Returns:
        Images depicting the stereochmistry correspondence pre and post correction.
        The PKS precursor to the target molecule (if appropriate).
        The final PKS design and PKS product after stereochemistry correction.
        MCS and Jaccard similarity metrics comparing the predicted PKS product to the entire
        target molecule.
    """
    target_mol = Chem.MolFromSmiles(canonicalize_smiles(target_smi, stereo))
    unbound_pks_product, pks_design_features = get_retro_tide_results(target_smi,
                                                                      stereo,
                                                                      offload_mech)
    pks_target_mol, mcs_score = get_pks_target(unbound_pks_product, target_mol)
    pp_result = preprocessing(pks_design_features, unbound_pks_product, pks_target_mol)
    krs_result = kr_swaps_algorithm(
        pp_result.unbound_mol,
        pp_result.target_mol,
        pp_result.mapping,
        pks_design_features,
        offload_mech,
        pp_result.chiral_result,
        pp_result.alkene_result)
    return {
        'stereo_before': visualize_stereo_correspondence(pp_result.unbound_mol,
                                                         pp_result.target_mol,
                                                         pp_result.chiral_result,
                                                         pp_result.alkene_result),
        'stereo_after': visualize_stereo_correspondence(krs_result.pks_product,
                                                        krs_result.target,
                                                        krs_result.chiral_result,
                                                        krs_result.alkene_result),
        'final_pks_design': krs_result.pks_design,
        'target_molecule': Chem.MolToSmiles(target_mol),
        'target_pks_precursor': Chem.MolToSmiles(pks_target_mol),
        'final_pks_product': Chem.MolToSmiles(krs_result.pks_product),
        'mcs_similarity': mcs_score,
        'jaccard_i': compute_jaccard_sim(pp_result.unbound_mol, target_mol),
        'jaccard_f': compute_jaccard_sim(krs_result.pks_product, target_mol)
    }

def output_results(results, job_name, output_path):
    """
    Output RetroTide + KR Swaps stereochemistry correction results to a JSON file and SVG images.
    """
    with open(f'{output_path}/{job_name}_stereocorrection_results.json',
              'w', encoding='utf-8') as json_file:
        json.dump({
            'Target Molecule': results['target_molecule'],
            'Target PKS Precursor': results['target_pks_precursor'],
            'Final PKS Design': results['final_pks_design'],
            'Final PKS Product': results['final_pks_product'],
            'MCS Similarity': results['mcs_similarity'],
            'Initial Jaccard Similarity': results['jaccard_i'],
            'Final Jaccard Similarity': results['jaccard_f']
        }, json_file, indent=2)
    with open(f'{output_path}/{job_name}_initial_stereo_correspondence.svg',
              'w', encoding='utf-8') as pre_img:
        pre_img.write(results['stereo_before'])
    with open(f'{output_path}/{job_name}_final_stereo_correspondence.svg',
              'w', encoding='utf-8') as post_img:
        post_img.write(results['stereo_after'])

# -*- coding: utf-8 -*-
"""
This file defines a framework for representing and manipulating Polyketide Synthase (PKS) designs in computational metabolic engineering, 
allowing for the exploration of the vast design space of polyketide products.

This framework focuses on constructing PKS enzyme complexes through a modular approach, 
with an emphasis on the flexibility of PKS systems via domain-level manipulations.
This approach aims to support the engineering of novel polyketides with desired structural and biological properties.
Functions within these classes allow for the representation of domain design spaces, execution of domain-specific biochemical reactions on polyketide chains, 
and the exploration of possible reactants and products involved in these reactions. 

Classes:
    Domain: An abstract base class for PKS catalytic domains, providing a template for defining specific domain functionalities.
        AT (Acyltransferase), KR (Ketoreductase), DH (Dehydratase), ER (Enoylreductase),
        TE (Thioesterase): Classes representing various (but not exhaustive) types of Domains.
        Note: the way these domains work is not a 1:1 mapping to the real world, but rather a simplified version.
    Module: Represents a single PKS module, comprising one or multiple domains, that contributes to the stepwise construction of polyketide chains.
    Cluster: Aggregates multiple Module instances to represent a complete PKS design, facilitating the computation of resultant polyketide products.

Typical usage example:
    loading_module_domains = {
        AT: AT(active=True, substrate='Acetyl-CoA'),
    }
    loading_module = Module(domains=loading_module_domains, loading=True)

    module1_domains = {
        AT: AT(active=True, substrate='Malonyl-CoA'),
        KR: KR(active=True, type='B1'),
        DH: DH(active=True),
        TE: TE(active=True)
        }
    module1 = Module(domains=module1_domains, loading=false)
    moduleStructure = module1.computeProduct()
    
    cluster = Cluster(modules=[loading_module, module1])
    chemStructure = cluster.computeProduct()

Created on Tue Jan 11 21:25:37 2022
@author: Tyler Backman, Vincent Blay
"""

from __future__ import annotations
import cobra
from cobra.core.metabolite import Metabolite
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, Draw, rdmolops
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from collections import OrderedDict
from copy import copy
from json import dumps
from typing import List, Dict, Type, Optional, Tuple
from typing_extensions import override
import pkg_resources

#%% PREPARE STARTERS AND EXTENDERS

def set_starters_extenders(path_starters: str = '', path_extenders: str = ''):
    """
    Loads the lists of starters and extenders used in RetroTide.

    This function initializes the global variables `starters` and `extenders` by loading them from specified files.
    If no paths are provided, default lists included with the RetroTide package are used.

    Args:
        path_starters (str, optional): The file path to a text file containing the list of starter molecules in SMILES format.
            Defaults to an empty string, which loads the default starters list.
        path_extenders (str, optional): The file path to a text file containing the list of extender molecules in SMILES format.
            Defaults to an empty string, which loads the default extenders list.

    Returns:
        Nothing

    Note:
        - `starters` and `extenders` are critical global variables within RetroTide and should not be manually altered outside of this function.
        - The specified files should contain SMILES strings with a title line and be tab-delimited.
    """
    # DATABASE OF PKS STARTERS
    if path_starters == '':
        path_starters = 'data/starters.smi'  # always use slash
    filepath = pkg_resources.resource_filename(__name__, path_starters)
    suppl = Chem.rdmolfiles.SmilesMolSupplier(
        filepath,
        delimiter='\t',
        titleLine=True,
        sanitize=True,
    )
    
    # now let's process these into a dict, and 'virtually attach to the ACP' by removal of the CoA
    global starters
    starters = {}
    
    for m in suppl:
        Chem.SanitizeMol(m)
        starters[m.GetProp('_Name')] = m
        
    # DATABASE OF PKS EXTENDER SUBSTRATES
    if path_extenders == '':
        path_extenders = 'data/extenders.smi'  # always use slash
    filepath = pkg_resources.resource_filename(__name__, path_extenders)
    extenderSuppl = Chem.rdmolfiles.SmilesMolSupplier(
        filepath,
        delimiter='\t',
        titleLine=True,
        sanitize=True,
    )
    
    # now let's process these into a dict, and 'virtually attach to the ACP' by removal of the CoA
    global extenders
    extenders = {}
    
    for m in extenderSuppl:
        rxn = AllChem.ReactionFromSmarts(
            # This reaction removes the CoA
            '[O:1]=[C:2]([O:3])[C:4][C:5](=[O:6])[S:7]'
            '>>'
            '[*:10]-[C:2](=[O:1])[C:4][C:5](=[O:6])[S:7].[O:3]')
        prod = rxn.RunReactants([m])[0][0]
        Chem.SanitizeMol(prod)
        extenders[m.GetProp('_Name')] = prod

    return

# Loads the default starters and extenders lists. 
set_starters_extenders()
# Users can overwrite the default list of starters and extenders
# by calling this function with adequate inputs.

#%% DEFINE OBJECTS FOR REPRESENTING PKS

class Cluster:
    """
    A class representing a PKS (Polyketide Synthase) design, which is essentially a list of modules.

    Attributes:
        modules (List[Module]): A list of modules that make up the PKS design.
    """

    def __init__(self, modules: List[Module] = None):
        """
        Initializes the Cluster instance with a list of modules.

        Args:
            modules (list, optional): A list of modules to be included in the PKS design. Defaults to an empty list if None is provided.
        """        
        if modules:
            self.modules = modules
        else:
            self.modules = []
            
    def computeProduct(self, structureDB: Dict[Module, Mol], chain: Mol = None) -> Mol:
        """
        Computes the chemical product of this PKS design based on the sequence of modules. 
        
        If a molecule object is passed as 'chain', only the final module operation is performed on this chain and returned. 
        This behavior is intended to accelerate retrobiosynthesis by focusing on the final modification step.

        Args:
            structureDB (Dict[Module, rdchem.Mol]): A database containing the structures and operations associated with each module.
            chain (Optional[rdchem.Mol]): A molecule object representing the current chain to be modified by the final module. If None, the entire PKS design is processed. Defaults to None.

        Returns:
            rdchem.Mol: The chemical product as a molecule object after applying the PKS design transformations.
        """
        if len(self.modules) == 0:
            return chain

        prod: Optional[Mol] = None
        modulesToExecute: List[Module] = self.modules
        if chain:
            prod = chain
            modulesToExecute = [self.modules[-1]] # last module only
            
        for idx, module in enumerate(modulesToExecute):
            if TE in module.domains:
                return module.domains[TE].operation(prod)
            
            if prod:
                moduleStructure = structureDB[module]
                # Select the previous module relative to current position in modulesToExecute
                # (instead of always using the second-to-last module of the full cluster)
                if idx > 0:
                    prev_mod = modulesToExecute[idx-1]
                else:
                    # No previous module within modulesToExecute; skip context-dependent logic
                    prev_mod = None
                if DH in module.domains and ER not in module.domains:
                    if prev_mod and ER in prev_mod.domains: #prev: alkane, current: alkene
                        if getattr(module.domains[DH], "type", None) == "Z":
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                 '[*:4][C:5]=[C:6][C:7]>>'
                                 '[#6:10]/[C:5]=[C:6]\[C:7]'
                                 '.[*:4].[C:1](=[O:2])[S:3]')) # Keep Z stereo
                        else:
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]=[C:6][C:7]>>'
                                    '[#6:10]/[C:5]=[C:6]/[C:7]'
                                    '.[*:4].[C:1](=[O:2])[S:3]')) # Keep E stereo
                    elif prev_mod: #prev: alkene, current: alkene
                        if (DH in prev_mod.domains and getattr(prev_mod.domains[DH], "type", None) == "Z") and \
                           (DH in module.domains and getattr(module.domains[DH], "type", None) == "Z"):
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[C:12]/[C:11]=[#6:10]\[C:1](=[O:2])[S:3].'
                                 '[*:4][C:5]=[C:6][C:7]>>'
                                 '[C:12]/[C:11]=[#6:10]\[C:5]=[C:6]/[C:7]'
                                 '.[*:4].[C:1](=[O:2])[S:3]'))
                        elif (DH in prev_mod.domains and getattr(prev_mod.domains[DH], "type", None) == "E") and \
                             (DH in module.domains and getattr(module.domains[DH], "type", None) == "E"):
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[C:12]/[C:11]=[#6:10]/[C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]=[C:6][C:7]>>'
                                    '[C:12]/[C:11]=[#6:10]/[C:5]=[C:6]/[C:7]'
                                    '.[*:4].[C:1](=[O:2])[S:3]'))
                        elif (DH in prev_mod.domains and getattr(prev_mod.domains[DH], "type", None) == "Z") and \
                             (DH in module.domains and getattr(module.domains[DH], "type", None) == "E"):
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[C:12]/[C:11]=[#6:10]\[C:1](=[O:2])[S:3].'
                                 '[*:4][C:5]=[C:6][C:7]>>'
                                 '[C:12]/[C:11]=[#6:10]\[C:5]=[C:6]\[C:7]'
                                 '.[*:4].[C:1](=[O:2])[S:3]'))
                        elif (DH in prev_mod.domains and getattr(prev_mod.domains[DH], "type", None) == "E") and \
                             (DH in module.domains and getattr(module.domains[DH], "type", None) == "Z"):
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[C:12]/[C:11]=[#6:10]/[C:1](=[O:2])[S:3].'
                                 '[*:4][C:5]=[C:6][C:7]>>'
                                 '[C:12]/[C:11]=[#6:10]/[C:5]=[C:6]\[C:7]'
                                 '.[*:4].[C:1](=[O:2])[S:3]'))
                        else: # i.e. alc to alkene (no DH in prev_mod)
                            if getattr(module.domains[DH], "type", None) == "Z":
                                rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                 '[*:4][C:5]=[C:6][C:7]>>'
                                 '[#6:10]/[C:5]=[C:6]\[C:7]'
                                 '.[*:4].[C:1](=[O:2])[S:3]')) # Keep Z stereo
                            else:
                                rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]=[C:6][C:7]>>'
                                    '[#6:10]/[C:5]=[C:6]/[C:7]'
                                    '.[*:4].[C:1](=[O:2])[S:3]')) # Keep E stereo
                    else: # no previous module ( - to alkene)
                        if getattr(module.domains[DH], "type", None) == "Z":
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                 '[*:4][C:5]=[C:6][C:7]>>'
                                 '[#6:10]/[C:5]=[C:6]\[C:7]'
                                 '.[*:4].[C:1](=[O:2])[S:3]')) # Keep Z stereo
                        else:
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]=[C:6][C:7]>>'
                                    '[#6:10]/[C:5]=[C:6]/[C:7]'
                                    '.[*:4].[C:1](=[O:2])[S:3]')) # Keep E stereo
                elif DH in module.domains and ER in module.domains: #If ER in current mod, can only have type E DH domains
                    if prev_mod and ER in prev_mod.domains: #prev: alkane, current: alkane
                        rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]~[C:6]>>'
                                    '[#6:10][C:5]~[C:6]'
                                    '.[*:4].[C:1](=[O:2])[S:3]'))
                    elif prev_mod: #prev: alkene, current: alkane
                        if DH in prev_mod.domains and getattr(prev_mod.domains[DH], "type", None) == "Z":
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[C:12]/[C:11]=[#6:10]\[C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]~[C:6]>>'
                                    '[C:12]/[C:11]=[#6:10]\[C:5]~[C:6]'
                                    '.[*:4].[C:1](=[O:2])[S:3]'))
                        elif DH in prev_mod.domains and getattr(prev_mod.domains[DH], "type", None) == "E":
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[C:12]/[C:11]=[#6:10]/[C:1](=[O:2])[S:3].'
                                 '[*:4][C:5]~[C:6]>>'
                                 '[C:12]/[C:11]=[#6:10]/[C:5]~[C:6]'
                                 '.[*:4].[C:1](=[O:2])[S:3]'))
                        else: #No DH in prev mod either
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]~[C:6]>>'
                                    '[#6:10][C:5]~[C:6]'
                                    '.[*:4].[C:1](=[O:2])[S:3]'))
                    else: # no previous module
                        rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]~[C:6]>>'
                                    '[#6:10][C:5]~[C:6]'
                                    '.[*:4].[C:1](=[O:2])[S:3]'))
                elif prev_mod and DH not in module.domains: # (i.e. only KR in current domain)
                    if ER not in prev_mod.domains: 
                        if DH in prev_mod.domains and getattr(prev_mod.domains[DH], "type", None) == "Z":
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[C:12]/[C:11]=[#6:10]\[C:1](=[O:2])[S:3].'
                                        '[*:4][C:5]~[C:6]>>'
                                        '[C:12]/[C:11]=[#6:10]\[C:5]~[C:6]'
                                        '.[*:4].[C:1](=[O:2])[S:3]'))
                        elif DH in prev_mod.domains and getattr(prev_mod.domains[DH], "type", None) == "E":
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[C:12]/[C:11]=[#6:10]/[C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]~[C:6]>>'
                                    '[C:12]/[C:11]=[#6:10]/[C:5]~[C:6]'
                                    '.[*:4].[C:1](=[O:2])[S:3]'))
                        else: #No DH in prev mod either
                            rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                        '[*:4][C:5]~[C:6]>>'
                                        '[#6:10][C:5]~[C:6]'
                                        '.[*:4].[C:1](=[O:2])[S:3]'))
                    else:
                        # prev_mod has ER, so prev_mod product is alkane
                        rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]~[C:6]>>'
                                    '[#6:10][C:5]~[C:6]'
                                    '.[*:4].[C:1](=[O:2])[S:3]'))
                else:
                    # perform condensation if this isn't in the starter
                    rxn: ChemicalReaction = AllChem.ReactionFromSmarts(('[#6:10][C:1](=[O:2])[S:3].'
                                    '[*:4][C:5]~[C:6]>>'
                                    '[#6:10][C:5]~[C:6]'
                                    '.[*:4].[C:1](=[O:2])[S:3]'))
                
                prod: Mol = rxn.RunReactants((prod, moduleStructure))[0][0]
                Chem.SanitizeMol(prod)
                
            else:
                # starter module
                prod = starters[module.domains[AT].substrate]
                
        return prod

    def create_atom_maps(self, structureDB: Dict[Module, Mol]) -> List[Mol]:
        """
        Generates an atom mapping of the PKS product arising from each module of the instantiated PKS cluster object

        This helps with tracking which atoms have been added and/ or transformed by which module.

        Args:
            structureDB (Dict[Module, rdchem.Mol]): A database containing the structures and operations associated with each module.

        Returns:
            List[Mol]: An ordered list of all atoms mappings. The first element in this list corresponds to the
                       atom-mapped PKS product generated by the loading module while the last element in this list
                       corresponds to the atom-mapped PKS product generated by the final extension module
        """

        def find_farthest_carbon_from_sulfur(PKS_product: Mol) -> Tuple[int, int]:
            """
            Takes a PKS product and finds the carbon atom farthest from the sulfur radical, which is a placeholder for bond to ACP protein domain.

            Args:
                PKS_product(rdChem.Mol): a mol object representing the PKS product.
            Returns:
                Tuple[int, int]: the index of the carbon atom farthest from the PKS product as well as the corresponding bond distance
            """
            # first, we identify the sulfur atom in the PKS product
            sulfur_idx = None
            for atom in PKS_product.GetAtoms():
                if atom.GetSymbol() == "S":
                    sulfur_idx = atom.GetIdx()
                    break

            if sulfur_idx is None:
                raise ValueError("No sulfur atom found in the molecule.")

            # then, grab the atom indices of all carbon atoms in this PKS substrate
            carbon_indices = [atom.GetIdx() for atom in PKS_product.GetAtoms() if atom.GetSymbol() == "C"]

            if not carbon_indices:
                raise ValueError("No carbon atoms found in the molecule.")

            # compute the shortest path distances from sulfur
            distances = rdmolops.GetDistanceMatrix(PKS_product)

            # find the farthest carbon atom
            farthest_carbon_idx = max(carbon_indices, key=lambda idx: distances[sulfur_idx, idx])
            farthest_distance = distances[sulfur_idx, farthest_carbon_idx]

            return farthest_carbon_idx, farthest_distance

        def add_fluorine_to_farthest_carbon(PKS_product: Mol, farthest_carbon_idx: int) -> Mol:
            """
            Adds a fluorine atom to the carbon atom previously identified as farthest from the sulfur radical (which represents bond to ACP domain).

            Args:
                PKS_product(rdChem.Mol): a mol object representing the PKS product.
                farthest_carbon_idx(int): the internal RDKit atom index of the carbon that is the farthest away frm the sulfur radical.

            Returns:
                rdChem.Mol: the updated molecule with the fluorine atom added to the specified carbon atom.
            """

            editable_mol = Chem.EditableMol(PKS_product)
            fluorine_idx = editable_mol.AddAtom(Chem.Atom("F"))

            # add a bond between the specific carbon atom (e.g., index 1) and fluorine
            editable_mol.AddBond(farthest_carbon_idx, fluorine_idx, Chem.rdchem.BondType.SINGLE)

            PKS_product_w_F = editable_mol.GetMol()

            return PKS_product_w_F

        def replace_fluorine_with_CoA(PKS_product_w_F: Mol) -> Mol:
            """
            Replaces the fluorine atom with a -CoA group to help in aligning molecules for downstream maximum common substructure comparisons.

            Args:
                 PKS_product_w_F (rdChem.Mol): a mol object representing the PKS product with fluorine attached.

            Returns:
                rdChem.Mol: a mol object representing the PKS product with -CoA attached instead of fluorine.
            """

            attach_CoA_rxn_pattern = '[C,c:1]-[F:2].[S:3]>>[C,c:1]-[S:3].[F:2]'
            attach_CoA_rxn = AllChem.ReactionFromSmarts(attach_CoA_rxn_pattern)

            CoA_group = Chem.MolFromSmiles(
                'CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)NCCS)O')

            for atom in CoA_group.GetAtoms():
                atom.SetProp("atomLabel", "X")

            PKS_product_w_CoA = attach_CoA_rxn.RunReactants((PKS_product_w_F, CoA_group))[0][0]
            return PKS_product_w_CoA

        def detach_CoA_frm_labelled_PKS_product(PKS_product_w_CoA: Mol):
            """
            Detaches the -CoA group from the carbon atom that is farthest from the sulfur radical to regenerate the original PKS product.

            Args:
                PKS_product_w_CoA(rdChem.Mol): a mol object representing the PKS product with a -CoA group.

            Returns:
                rdchem.Mol: a mol object representing the original PKS product without the -CoA group attached.
            """
            detach_CoA_rxn_pattern = '[C,c:1]-[S:3]-[C:4]-[C:5]-[N:6]>>[C,c:1].[S:3]-[C:4]-[C:5]-[N:6]'
            detach_CoA_rxn = AllChem.ReactionFromSmarts(detach_CoA_rxn_pattern)
            labelled_PKS_product = detach_CoA_rxn.RunReactants((PKS_product_w_CoA,))[0][0]
            return labelled_PKS_product

        modules_list = self.modules
        atom_labels_list = [f"LM" if i == 0 else f"M{i}" for i in range(len(modules_list))]
        labelled_mols_list = []

        for module_num in range(0, len(modules_list)):

            # for the loading module specifically
            if module_num == 0:

                # generate the loading module product
                cluster = Cluster(modules=[modules_list[0]])  # input to bcs.Cluster is list of modules
                LM_product = cluster.computeProduct(structureDB)

                # attach a florine atom to the farthest carbon from the sulfur radical (ACP placeholder)
                farthest_carbon_idx, _ = find_farthest_carbon_from_sulfur(LM_product)
                LM_product_w_F = add_fluorine_to_farthest_carbon(LM_product, farthest_carbon_idx)

                # then swap the florine for a -CoA group so all PKS products can be easily aligned
                LM_product_w_CoA = replace_fluorine_with_CoA(LM_product_w_F)

                for atom in LM_product_w_CoA.GetAtoms():
                    if atom.HasProp("atomLabel") == 0:
                        atom.SetProp("atomLabel", "LM")

                labelled_mols_list.append(LM_product_w_CoA)

            else:

                current_modules_sublist = modules_list[0:(module_num + 1)]
                atom_labels_sublist = atom_labels_list[0:(module_num + 1)]
                current_cluster = Cluster(modules=current_modules_sublist)
                current_product = current_cluster.computeProduct(structureDB)

                # similar to the loading module, attach a fluorine to the farthest carbon atom from ACP
                farthest_carbon_idx, _ = find_farthest_carbon_from_sulfur(current_product)
                current_product_w_F = add_fluorine_to_farthest_carbon(current_product, farthest_carbon_idx)
                current_product_w_CoA = replace_fluorine_with_CoA(current_product_w_F)

                # for each previous_product starting from up until the previous module
                for i, previous_product_w_CoA in enumerate(labelled_mols_list[::-1]):

                    MCS_result = rdFMCS.FindMCS([current_product_w_CoA, previous_product_w_CoA],
                                                timeout=1,
                                                matchChiralTag=False,
                                                bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact)

                    match = current_product_w_CoA.GetSubstructMatch(MCS_result.queryMol)

                    label = atom_labels_sublist[::-1][i]

                    for atom in current_product_w_CoA.GetAtoms():
                        if atom.GetIdx() not in match:
                            if atom.HasProp("atomLabel") == 0:
                                atom.SetProp("atomLabel", label)

                labelled_mols_list.append(current_product_w_CoA)

        # for all the labelled PKS substrates, detach the -CoA group
        final_mols_list = [detach_CoA_frm_labelled_PKS_product(mol) for mol in labelled_mols_list]

        # finally, label atoms that were previously attached to the -CoA group with "LM"
        for mol in final_mols_list:
            for atom in mol.GetAtoms():
                if atom.HasProp("atomLabel") == 0:
                    atom.SetProp("atomLabel", "LM")

        return final_mols_list

    @staticmethod
    def visualize_atom_maps(atom_mapped_mols: List[Mol]) -> None:
        """
        Visualize the atom-mapped PKS products in a jupyter notebook as a RDKit mols grid.

        Args:
            atom_mapped_mols (List[Mol]): list of atom mapped mol objects
        """

        return Draw.MolsToGridImage(mols = atom_mapped_mols,
                                    molsPerRow = 3,
                                    subImgSize = (500,200),
                                    legends = ["Loading Module" if i==0 else f"Module {i}" for i in range(len(atom_mapped_mols))],)

class Module:
    """
    A class representing a PKS (Polyketide Synthase) module.

    Attributes:
        product (str): The product of the module.
        iterations (int): The number of iterations the module undergoes.
        domains (OrderedDict): An ordered dictionary where keys are domain classes and values are domain objects.
        loading (bool): Indicates whether the module is a loading module.
    """

    def __init__(self, product: str = '', iterations: int = 1, domains: OrderedDict[Type[Domain], Domain] = None, loading: bool = False):
        """
        Initializes the Module instance with product, iterations, domains, and loading status.

        Args:
            product (str): The product of the module. Defaults to an empty string.
            iterations (int): The number of iterations the module undergoes. Defaults to 1.
            domains (OrderedDict): An ordered dictionary of domain classes and objects. Defaults to None, which initializes an empty OrderedDict.
            loading (bool): Indicates whether the module is a loading module. Defaults to False.
        """
        self.product = product
        self.iterations = iterations
        self.loading = loading
        if domains:
            self.domains = domains
        else:
            self.domains = OrderedDict() 
    
    @staticmethod
    def domainTypes() -> List[Type[Domain]]:
        """
        Returns all domain types that can occur in a PKS module in the catalytic order in which they operate.

        Returns:
            List[Domain]: A list of Domain classes representing the types of domains in a PKS module.
        """
        return [AT, KR, DH, ER, TE]
    
    def computeProduct(self) -> Mol:
        """
        Computes the chemical product of the module based on the operations of its domains.

        Returns:
            Chem.Mol: The chemical product as a molecule object. Returns None if there are no domains in the module.
        """        
        chain = None
        for domain in self.domains.values():
            chain = domain.operation(chain)
            
        return chain
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the module, including its domains and loading status.

        Returns:
            str: A string representation of the module.
        """
        return repr([cls.__name__ + repr(domain) for cls, domain in self.domains.items()] + ['loading: ' + repr(self.loading)])
    
    def __hash__(self) -> int:
        """
        Produces a unique hash key for the module based on its domain configuration and loading status.

        Returns:
            int: The hash value of the module.
        """
        return hash(tuple(self.domains.values()) + (self.loading,))

    def __eq__(self, other: Module) -> bool:
        """
        Checks if another object is equal to this Module instance based on their hash values.

        Args:
            other (Module): The object to compare with this Module instance.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if isinstance(other, type(self)):
            return hash(self) == hash(other)
        else:
            return False

    def __ne__(self, other: Module)  -> bool:
        """
        Checks if another object is not equal to this Module instance.

        Args:
            other (Module): The object to compare with this Module instance.

        Returns:
            bool: True if the objects are not equal, False otherwise.
        """        
        return (not self.__eq__(other))  
        
class Domain:
    """
    Abstract base class used to build PKS catalytic domains.

    Attributes:
        active (bool): Indicates whether the domain is active.
    """

    def __init__(self, active):
        """
        Initializes a new domain with a design as reported by designSpace.

        Args:
            active (bool): Indicates whether the domain is active.
        """
        self.active = active
        
    def design(self) -> Dict[str, any]:
        """
        Reports the design of this object.

        Returns:
            dict: The design of the domain as a dictionary of its attributes.
        """
        return vars(self)
    
    @classmethod
    def designSpace(cls, module: Module = None) -> List[Domain]:
        """
        Returns a set of Domain objects representing the full design space of this domain.
        Can optionally take a PKS module to report only the compatible configurations
        of this domain with that design. Domains of this type in the design are ignored.
        If incompatible domains are included in the design, it just returns an empty list.

        Args:
            module (Module, default=None): A PKS module to limit the design space
            to only those compatible with the module.

        Returns:
            List[Domain]: A set of domain objects representing the full design space.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        # module--allows you limit the design space to only those compatible with the module
        # this one allows you to combinatorially generate the design space
        raise NotImplementedError
        
    def operation(self, chain: Mol) -> Mol:
        """
        Executes this domain's operation on top of an existing PKS chain 
        and returns the chemical product.

        Args:
            chain (Mol): The current PKS chain as an RDKit mol object.

        Returns:
            Mol: the chemical product as an RDKit mol object.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
        
    def reactants(self) -> List[Metabolite]:
        """
        Returns all reactants of this domain, excluding the substrate (polyketide chain).

        Returns:
            List[Metabolite]: A list of Cobrapy metabolites representing the reactants.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """        
        raise NotImplementedError        
        
    def products(self) -> List[Metabolite]:
        """
        Returns all products of this domain, excluding the polyketide chain.

        Returns:
            List[Metabolite]: A list of Cobrapy metabolites representing the products.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Returns a string representing this domain type. 
        
        This string is used for text-based storage of PKS designs, or reporting to the user.
        Only prints activity if active=False to keep things concise.

        Returns:
            str: A string representation of the domain.
        """
        if self.active:
            designCopy = copy(self.design())
            del designCopy['active']
            return(repr(designCopy))
        else:
            return(repr(self.design()))
        
    def __hash__(self):
        """
        Produces a unique hash key for each domain configuration based on its type and attributes.

        Returns:
            int: The hash value of the domain.
        """
        # Serialize the domain design sorted by keys for consistency
        serialized_design = dumps(self.design(), sort_keys=True)
        
        # Combine the type of the class with the serialized design
        # This ensures that different classes with identical data do not produce the same hash
        return hash((type(self), serialized_design))

    def __eq__(self, other) -> bool:
        """
        Checks if another object is equal to this Domain instance based on their hash values.

        Args:
            other (Any): The object to compare with this Domain instance.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if isinstance(other, type(self)):
            return hash(self) == hash(other)
        else:
            return False

    def __ne__(self, other) -> bool:
        """
        Checks if another object is not equal to this Domain instance.

        Args:
            other (Any): The object to compare with this Domain instance.

        Returns:
            bool: True if the objects are not equal, False otherwise.
        """
        return (not self.__eq__(other))
        
class AT(Domain):
    """
    Represents an Acyltransferase (AT) domain.
    
    This domain class performs PKS chain elongation, simulating the combined activities of the KS, AT, and ACP domains.
    For more information on these domains, see:
    Keatinge-Clay, Adrian T. "The structures of type I polyketide synthases." Natural product reports 29.10 (2012): 1050-1073.
    doi: 10.1039/c2np20019h

    Attributes:
        active (bool): See base class.
        substrate (str): The substrate used to extend the chain. The substrates
                         are different for loading modules or extension modules.
    """

    def __init__(self, active, substrate):
        """
        Initializes a new AT domain with specified activity status and substrate.

        Args:
            active (bool): Indicates whether the domain is active.
            substrate (str): The substrate used by the AT domain. 
                The substrates are different for loading modules or extension modules.
        """
        super().__init__(active)
        self.substrate = substrate
    
    @classmethod
    @override
    def designSpace(cls, module=False) -> List[AT]:
        if module:
            if module.loading:
                return [cls(active=True, substrate=s) for s in starters.keys()]
        
        # return only extension ATs unless passed a loading module for context
        return [cls(active=True, substrate=s) for s in extenders.keys()]
        
    def operation(self, chain: Mol = None, loading: bool = False) -> Mol:
        """
        Performs the AT domain's operation on the given chain.

        Args:
            chain (Optional[Mol]): The current polyketide chain. If None, a new chain
                                   is started based on the substrate. Defaults to None.
            loading (bool): Indicates if the operation is for a loading module. Defaults to False.

        Returns:
            Mol: The result of the AT domain operation on the polyketide chain.

        Raises:
            NotImplementedError: If the operation logic is not implemented.
        """        
        if not chain:
            try:
                if loading:
                    # Attempt to access the substrate in the starters collection
                    return starters[self.substrate]
                else:
                    # Attempt to access the substrate in the extenders collection
                    return extenders[self.substrate]
            except KeyError:
                # Raise a more informative exception if the substrate is not found
                raise ValueError(f"Substrate '{self.substrate}' not found in {'starters' if loading else 'extenders'} collection.")
        else:
            # ATs here don't perform condensation, so need to operate first
            # the condensation is performed afterwards
            raise NotImplementedError

    def reactants(self) -> List[Metabolite]:
        """
        Returns the reactants involved in the AT domain's catalytic activity, excluding the polyketide chain substrate.

        This method identifies the specific CoA-bound substrate (e.g., Malonyl-CoA, Methylmalonyl-CoA) that the domain uses to extend the polyketide chain. 
        It returns a list of Cobrapy metabolite objects representing the reactants needed for the AT domain's activity, 
        including the specific substrate and a hydrogen ion.

        Returns:
            List[Metabolite]: A list of Cobrapy metabolites representing the reactants required for the AT domain's activity.
            This typically includes a hydrogen ion and the specific CoA-bound substrate used by the AT domain.

        Note:
            The specific CoA-bound substrate is determined by the `substrate` attribute of the AT domain instance.
        """
        # Define the hydrogen ion as a common reactant.
        hydrogen_ion = cobra.Metabolite('h_c', compartment='c')
        
        # Define the substrate-specific Cobrapy metabolite based on the AT domain's substrate.
        if self.substrate == 'Malonyl-CoA':
            substrate_cobrapy = cobra.Metabolite('malcoa_c', compartment='c')
        elif self.substrate == 'Methylmalonyl-CoA':
            substrate_cobrapy = cobra.Metabolite('mmcoa__S_c', compartment='c')
        else:
            # For other substrates, create a metabolite object with a generic name.
            substrate_cobrapy = cobra.Metabolite(self.substrate.lower().replace(" ", "_") + '_c', compartment='c')
        
        # Return a list of the reactants.
        return [hydrogen_ion, substrate_cobrapy]
        
    @override
    def products(self):
        return [
            cobra.Metabolite('coa_c', compartment='c'),
            cobra.Metabolite('co2_c', compartment='c')
        ]
            
class KR(Domain):
    """
    Represents a Ketoreductase (KR) domain within a Polyketide Synthase (PKS) module.

    For more information on this domain, see:
    Keatinge-Clay, Adrian T. "The structures of type I polyketide synthases." Natural product reports 29.10 (2012): 1050-1073.
    doi: 10.1039/c2np20019h

    Attributes:
        TYPE_CHOICES (ClassVar[Set[str]]): A class variable that defines the valid types of 
            KR domains based on their biochemical activity and specificity. 
            This can include various stereochemical outcomes of the reduction process.
        active (bool): See base class.
        type (str): Specifies the type of the KR domain, which determines the stereochemistry of the reduction reaction. 
            Must be one of the specified `TYPE_CHOICES`.
    """    
    # TYPE_CHOICES = {'B1', 'B', 'C1'} # 2D change
    TYPE_CHOICES = {'A1', 'A2', 'A', 'B1', 'B2', 'B', 'C1', 'C2'}
    # TYPE_CHOICES = {'A1', 'A2', 'A', 'B1', 'B2', 'B', 'C1', 'C2', 'U'}

    def __init__(self, active: bool, type: str):
        """
        Initializes a new KR domain with specified activity and type.

        Args:
            active (bool): Indicates whether the domain is active.
            type (str): The type of the KR domain, must be one of the specified TYPE_CHOICES.

        Raises:
            AssertionError: If the specified type is not in TYPE_CHOICES.
        """
        assert type in self.TYPE_CHOICES, f"Type {type} is not a valid KR domain type."
        super().__init__(active)
        self.type = type
        
    @classmethod
    @override
    def designSpace(cls, module: Module = None) -> List[KR]:
        updatedTypeChoices = copy(cls.TYPE_CHOICES)
        
        if module and module.domains[AT].substrate != 'Malonyl-CoA':
            print()
            # if the domain occurs in a module WITHOUT a Malonyl-CoA AT, remove the A/B type
            updatedTypeChoices.difference_update({'A', 'B'})
        elif module:
            # if the domain occurs in a module WITH a Malonyl-CoA AT, keep only the A/B type
            updatedTypeChoices.difference_update({'A1', 'A2', 'B1', 'B2', 'C1', 'C2'})
        
        return [cls(active=True, type=type) for type in updatedTypeChoices] + [cls(active=False, type='B1')]
    
    def operation(self, chain: Mol) -> Mol:
        """
        Applies the specific ketoreduction reaction associated with this KR domain type to a polyketide chain.

        This method executes a chemical transformation based on the KR domain's type, reducing a ketone group to a hydroxyl group on the polyketide chain. 
        The specific type of KR domain determines the stereochemistry of the resulting hydroxyl group, 
        with each type ('A1', 'A2', 'A', 'B1', 'B2', 'B', 'C1', 'C2') corresponding to different stereochemical outcomes.

        Args:
            chain (Mol): An RDKit Mol object representing the current polyketide chain, which must contain a ketone group susceptible to reduction by this KR domain.

        Returns:
            Mol: An RDKit Mol object representing the modified polyketide chain after the reduction reaction has been applied. 
            The product's stereochemistry is determined by the KR domain type.

        Raises:
            AssertionError: If the input polyketide chain does not contain a ketone group matching the expected pattern for reduction,
            indicating that the chain is not a suitable substrate for this KR domain.
        """
        assert len(chain.GetSubstructMatches(Chem.MolFromSmiles('C(=O)CC(=O)S'),
                   useChirality=True)) == 1, Chem.MolToSmiles(chain)

        if self.type == 'A1':
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C@:2]([O:3])[C@:4]'
                                                   '[C:5](=[O:6])[S:7]'))
        elif self.type == 'A2':
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C@:2]([O:3])[C@@:4]'
                                                   '[C:5](=[O:6])[S:7]'))
        elif self.type == 'A':
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C@:2]([O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]'))
        elif self.type == 'B1':
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C@@:2]([O:3])[C@:4]'
                                                   '[C:5](=[O:6])[S:7]'))
        elif self.type == 'B2':
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C@@:2]([O:3])[C@@:4]'
                                                   '[C:5](=[O:6])[S:7]'))
        elif self.type == 'B':
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C@@:2]([O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]'))
        elif self.type == 'C2': # performs epimerization
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C:2](=[O:3])[C@@:4]'
                                                   '[C:5](=[O:6])[S:7]'))
        elif self.type == 'C1': # does not change stereochemistry
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C:2](=[O:3])[C@:4]'
                                                   '[C:5](=[O:6])[S:7]'))
        else:
            # By first specifying some stereochemistry in the reactants
            # and then explicitly "losing" the stereochemistry in the products
            # we can forget the stereochemistry in our molecule
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2](=[O:3])[C@:4]'
                                                   '[C:5](=[O:6])[S:7]>>'
                                                   '[#0:1][C:2]([O:3])[C:4]'
                                                   '[C:5](=[O:6])[S:7]'))
            
        prod = rxn.RunReactants((chain,))[0][0]
        Chem.SanitizeMol(prod)
        return prod
        
    def reactants(self) -> List[Metabolite]:
        """
        Returns the reactants required for the ketoreduction reaction catalyzed by the KR domain (NADPH and a proton (H+)),
            excluding the polyketide chain substrate itself.

        The stoichiometry of the reaction is represented by the equation:
        ketone_pks_product + NADPH + H+ -> hydroxyl_pks_product + NADP+.

        Returns:
            List[Metabolite]: A list of Cobrapy Metabolite objects representing the reactants for the 
            ketoreduction process.
        """
        
        return [
            cobra.Metabolite('nadph_c', compartment='c'),
            cobra.Metabolite('h_c', compartment='c'),
        ]     
        
    @override
    def products(self) -> List[Metabolite]:
        """
        Stoich: ketone_pks_product + NADPH + H+ -> hydroxyl_pks_product + NADP+
        """
        return [
            cobra.Metabolite('nadp_c', compartment='c'),
        ]     
    
class DH(Domain):
    """
    Represents the Dehydratase (DH) domain in a Polyketide Synthase (PKS) module.

    For more information on this domain, see:
    Keatinge-Clay, Adrian T. "The structures of type I polyketide synthases." Natural product reports 29.10 (2012): 1050-1073.
    doi: 10.1039/c2np20019h
    """
    TYPE_CHOICES = {'Z', 'E'}

    def __init__(self, active: bool, type: str):
        """
        Initializes a new DH domain with specified activity and type.

        Args:
            active (bool): Indicates whether the domain is active.
            type (str): The type of the DH domain, must be one of the specified TYPE_CHOICES.
        
        Raises:
            AssertionError: If the type is not one of the specified TYPE_CHOICES.
        """
        assert type in self.TYPE_CHOICES, f"Type {type} is not a valid DH domain type."
        super().__init__(active)
        self.type = type

    @classmethod
    @override
    def designSpace(cls, module: Module = None) -> List[DH]:
        # adding False as a design type specifies that this domain is optional,
        # e.g. a PKS can exist without it
        updatedTypeChoices = copy(cls.TYPE_CHOICES)

        if not module:
            return [cls(active=True, type=type) for type in updatedTypeChoices] + [cls(active=False, type='E')]

        if not KR in module.domains:
            return [cls(active=False, type='E')]

        # require that we have an active A/B/B1 KR type
        if module.domains[KR].active and (module.domains[KR].type in {'A', 'B', 'B1', 'U'}):
            if module and module.domains[KR].type != 'A':
                updatedTypeChoices.difference_update({'Z'})
            else:
                updatedTypeChoices.difference_update({'E'})
            return [cls(active=True, type=type) for type in updatedTypeChoices] + [cls(active=False, type='E')]
        else:
            return [cls(active=False, type='E')]

    def operation(self, chain: Mol) -> Mol:
        """
        Executes the dehydration reaction catalyzed by the Dehydratase domain on a polyketide chain.

        This method performs a chemical transformation that removes a water molecule from the polyketide chain, 
        specifically targeting a hydroxyl group adjacent to a ketone.
        The method first attempts to dehydrate the substrate while preserving the stereochemistry of any methyl 
        groups attached to the α-carbon. If the initial reaction fails due to specific stereochemical configurations 
        (e.g., if sanitization fails because the resulting structure is not chemically valid), an alternative 
        reaction pathway is attempted, which adjusts the stereochemistry at the α-carbon.

        Args:
            chain (Mol): An RDKit Mol object representing the current polyketide chain substrate for the reaction. 
                        The substrate must contain a hydroxyl group adjacent to a ketone for the reaction to proceed.

        Returns:
            Mol: An RDKit Mol object representing the polyketide chain after the Dehydratase-catalyzed reaction.

        Raises:
            AssertionError: If the input polyketide chain does not contain a hydroxyl group adjacent to a ketone, 
                            indicating that the chain is not a suitable substrate for dehydration by this domain.
        """
        # try setting CH unchanged
        assert len(chain.GetSubstructMatches(Chem.MolFromSmiles('C(O)CC(=O)S'),
           useChirality=True)) == 1, Chem.MolToSmiles(chain, isomericSmiles=True)

        if self.type == 'Z':
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>'
                                              '[#0:1]/[CH1:2]=[CH1:4]\[C:6](=[O:7])[S:8].[O:3]'))
        elif self.type == 'E':
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>'
                                              '[#0:1]/[CH1:2]=[CH1:4]/[C:6](=[O:7])[S:8].[O:3]'))
        else:
            rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>'
                                              '[#0:1][CH1:2]=[CH1:4][C:6](=[O:7])[S:8].[O:3]'))
        prod = rxn.RunReactants((chain,))[0][0]
        try:
            Chem.SanitizeMol(prod)
        except ValueError: 
            # if this has a methyl attached on the alpha carbon, we'll set CH0
            if self.type == 'Z':
                rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>'
                                                '[#0:1]/[CH1:2]=[CH0:4]\[C:6](=[O:7])[S:8].[O:3]'))
            elif self.type == 'E':
                rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>'
                                                '[#0:1]/[CH1:2]=[CH0:4]/[C:6](=[O:7])[S:8].[O:3]'))
            else:
                rxn = AllChem.ReactionFromSmarts(('[#0:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>'
                                                '[#0:1][CH1:2]=[CH0:4][C:6](=[O:7])[S:8].[O:3]'))
            prod = rxn.RunReactants((chain,))[0][0]
            Chem.SanitizeMol(prod)
        return prod
        
    def reactants(self) -> List[Metabolite]:
        """
        Identifies the reactants involved in the dehydration reaction catalyzed by the Dehydratase domain.

        This method returns a list of external reactants required for the Dehydratase domain's activity. 
        Since the reaction primarily involves the loss of a water molecule directly from the substrate, 
        and does not require additional external reactants, the returned list is empty.

        The stoichiometry of the dehydration reaction is represented by the following equation:
            hydroxyl_pks_product -> alkene_pks_product + H2O
        
        Returns:
            List[Metabolite]: An empty list, indicating no external reactants are required for the 
            dehydration reaction catalyzed by this domain.
        """
        return []
    
    @override
    def products(self) -> List[Metabolite]:
        return [
            cobra.Metabolite('h2o_c', compartment='c'),
        ]    
    
class ER(Domain):
    """
    Represents an EnoylReductase (ER) domain within a PKS module. 
    
    For more information on this domain, see:
    Keatinge-Clay, Adrian T. "The structures of type I polyketide synthases." Natural product reports 29.10 (2012): 1050-1073.
    doi: 10.1039/c2np20019h
    """
    TYPE_CHOICES = {'L', 'D'}

    def __init__(self, active: bool, type:str):
        """
        Initializes a new ER domain with specified activity and type.

        Args:
            active (bool): Indicates whether the domain is active.
            type (str): The type of the ER domain, must be one of the specified TYPE_CHOICES.
        
        Raises:
            AssertionError: If the type is not one of the specified TYPE_CHOICES.
        """
        assert type in self.TYPE_CHOICES, f"Type {type} is not a valid ER domain type."
        super().__init__(active)
        self.type = type
        
    @classmethod
    @override
    def designSpace(cls, module: Module = None) -> List[ER]:
        """
        Determines the design space for the ER domain within a PKS module. 
        This involves deciding whether the ER domain should be active based on the module's composition, specifically the presence and activity of a DH domain.

        Args:
            module (Module, optional): The PKS module to consider when determining the ER domain's design space. Defaults to None, implying no module context is provided.

        Returns:
            List[ER]: A list of ER domain instances representing the design space, which may include both active and inactive states depending on the module's configuration.
        """
        # adding False as a design type specifies that this domain is optional,
        # e.g. a PKS can exist without it
        updatedTypeChoices = copy(cls.TYPE_CHOICES)
        
        if not module:
            return [cls(active=True, type=type) for type in updatedTypeChoices] + [cls(active=False, type = 'L')]

        if not DH in module.domains:
            return [cls(active=False, type = 'L')]

        # require that we have an active DH type (NEED TO REQUIRE AN ACTIVE DH TYPE E ONLY)
        if module.domains[DH].active and (module.domains[DH].type in {'E'}):
            return [cls(active=True, type=type) for type in updatedTypeChoices] + [cls(active=False, type = 'L')]
        else:
            return [cls(active=False, type = 'L')]

    def operation(self, chain: Mol) -> Mol:
        """
        Executes the chemical transformation associated with the ER domain on a given polyketide chain.
        This typically involves the reduction of a carbon-carbon double bond to a single bond.

        Args:
            chain (Mol): The current polyketide chain as an RDKit Mol object, representing the substrate for the ER domain's reaction.
            The chain should include at least one double bond conforming to the pattern 'C=CC(=O)S'.

        Returns:
            Mol: The modified polyketide chain as an RDKit Mol object, reflecting the outcome of the ER domain's reduction reaction.
        """
        assert len(chain.GetSubstructMatches(Chem.MolFromSmiles('C=CC(=O)S'),
           useChirality=True)) == 1, Chem.MolToSmiles(chain)

        # try setting CH unchanged
        rxn = AllChem.ReactionFromSmarts(('[#0:1]/[C:2]=[C:3]/[C:4](=[O:5])[S:6]>>'
                                          '[#0:1][CH2:2][CH2:3][C:4](=[O:5])[S:6]'))
                                          # '[#0:1][CH2:2][CH2:3][C:4](=[O:5])[S:6]'))

        prod = rxn.RunReactants((chain,))[0][0]
        try:
            Chem.SanitizeMol(prod)
        except ValueError:
            if self.type == 'L':
                rxn = AllChem.ReactionFromSmarts(('[#0:1]/[C:2]=[C:3]/[C:4](=[O:5])[S:6]>>'
                                                '[#0:1][CH2:2][C@@H1:3][C:4](=[O:5])[S:6]'))
                                                # '[#0:1][CH2:2][CH:3][C:4](=[O:5])[S:6]'))
            elif self.type == 'D':
                rxn = AllChem.ReactionFromSmarts(('[#0:1]/[C:2]=[C:3]/[C:4](=[O:5])[S:6]>>'
                                                '[#0:1][CH2:2][C@H1:3][C:4](=[O:5])[S:6]'))
                                                # '[#0:1][CH2:2][CH:3][C:4](=[O:5])[S:6]'))
            else:
                rxn = AllChem.ReactionFromSmarts(('[#0:1]/[C:2]=[C:3]/[C:4](=[O:5])[S:6]>>'
                                                '[#0:1][CH2:2][CH1:3][C:4](=[O:5])[S:6]'))
                                                # '[#0:1][CH2:2][CH:3][C:4](=[O:5])[S:6]'))
            prod = rxn.RunReactants((chain,))[0][0]
        return prod
        
    def reactants(self) -> List[Metabolite]:
        """
        Provides a list of reactants (excluding the polyketide chain itself) required for the ER domain's enzymatic reaction, 
        highlighting the role of NADPH and protons in the reduction process.
        Stoich: alkene_pks_product + NADPH + H+ -> alkane_pks_product + NADP+

        Returns:
            List[Metabolite]: A list of Cobrapy Metabolite objects representing the reactants for the ER domain's operation, typically including NADPH and a proton.
        """
        
        return [
            cobra.Metabolite('nadph_c', compartment='c'),
            cobra.Metabolite('h_c', compartment='c')
        ]
        
    @override
    def products(self) -> List[Metabolite]:
        return [
            cobra.Metabolite('nadp_c', compartment='c')
        ]  
    
class TE(Domain):
    """
    Represents a Thioesterase (TE) domain in a PKS module.

    For more information on this domain, see:
    Keatinge-Clay, Adrian T. "The structures of type I polyketide synthases." Natural product reports 29.10 (2012): 1050-1073.
    doi: 10.1039/c2np20019h

    Attributes:
        active (bool): See base class.
        cyclic (bool): Indicates whether the product is cyclic.
        ring (int): The size of the ring formed if the product is cyclic.
    """
    cyclic_reaction = '([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]'
    linear_reaction = '[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]'
    def __init__(self, active: bool, cyclic: bool, ring: int):
        """
        Initializes a new TE domain with specified characteristics.

        Args:
            active (bool): See base class.
            cyclic (bool): Indicates whether the product is cyclic.
            ring (int): The size of the ring formed if the product is cyclic.
        """
        super().__init__(active)
        self.cyclic = cyclic
        self.ring = ring

    @classmethod
    @override
    def designSpace(cls, module: Module = None) -> List[TE]:
        # adding False as a design type specifies that this domain is optional,
        # e.g. a PKS can exist without it
        
        # For now this returns false so it doesn't get included in designs
        # later we need to deal with terminal domains in a better way
        return [cls(active=False, cyclic=False, ring=0)]

    def operation(self, chain: Mol) -> Mol:
        """
        Executes the chemical reaction catalyzed by the Thioesterase (TE) domain on a polyketide chain.

        This method performs the final step in the polyketide synthesis process. 
        The behavior depends on the 'cyclic' attribute: if true, the method attempts to form a ring
        structure of size member variable 'ring'; otherwise, it releases the polyketide as a free acid.

        Args:
            chain (Mol): An RDKit Mol object representing the current polyketide chain attached to a thioesterase domain.

        Returns:
            Mol: An RDKit Mol object representing the polyketide after the TE domain's reaction has been applied.
                This will be either a cyclic product or a free polyketide acid, depending on the domain's configuration.

        Raises:
            AssertionError: If the input polyketide chain does not contain a thioester group that matches the expected
                            pattern for cyclization or release, indicating that the chain is not a suitable substrate.

        Note:
            The method assumes the presence of a thioester group in the input chain 
            and uses the domain's 'cyclic' and 'ring' attributes to determine the reaction pathway.
        """
        assert len(chain.GetSubstructMatches(Chem.MolFromSmiles('C(=O)S'),
                   useChirality=True)) == 1, Chem.MolToSmiles(chain)

        # Using index -1 will yield the largest ring
        index = -1
        if self.cyclic:
            rxn = AllChem.ReactionFromSmarts(TE.cyclic_reaction)
            index -= self.ring
        else:
            rxn = AllChem.ReactionFromSmarts(TE.linear_reaction)

        prod = rxn.RunReactants((chain,))[index][0]
        Chem.SanitizeMol(prod)

        return prod
          
    def reactants(self) -> List[Metabolite]:
        """
        Provides a list of external reactants required for the Thioesterase (TE) domain's activity.

        In the context of a TE-catalyzed reaction, water (H2O) is consumed to hydrolyze the thioester bond,
        releasing the free polyketide product.

        Returns:
            List[Metabolite]: A list containing the Cobrapy Metabolite object for water, indicating its requirement
                            for the hydrolysis reaction catalyzed by the TE domain.

        Note:
            The current implementation focuses on the non-cyclic product version of the TE domain's activity. 
            Future implementations should support the cyclic version.
        """
        return [cobra.Metabolite('h2o_c', compartment='c')]        
        
    @override
    def products(self) -> List[Metabolite]:
        return [cobra.Metabolite('h_c', compartment='c')]    
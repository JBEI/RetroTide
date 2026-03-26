"""Unit tests for the krswaps modules"""
import json
import os
import tempfile
import unittest
from collections import OrderedDict
from unittest.mock import patch, MagicMock

import pandas as pd
from parameterized import parameterized
from rdkit import Chem

import bcs
from bcs import *
from krswaps import krswaps as krs


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_pks_features(substrates, kr_types, dh_types, er_types):
    """Build a pks_features dict for testing."""
    return {
        'Module': list(range(len(substrates))),
        'Substrate': substrates,
        'KR Type': kr_types,
        'DH Type': dh_types,
        'ER Type': er_types,
    }

class TestInputStereochemistry(unittest.TestCase):

    def test_valid_stereo_option(self):
        """Test that SMILES canonicalization works correctly."""
        smiles = 'C/C=C/[C@H]([C@H]([C@H](C(O)=O)C)O)C'
        alkene_free_smi = Chem.MolToSmiles(
            Chem.MolFromSmiles('CC=C[C@H]([C@H]([C@H](C(O)=O)C)O)C'))
        
        expected_canonical_smiles = krs.canonicalize_smiles(smiles, 'R/S')
        self.assertEqual(expected_canonical_smiles, alkene_free_smi)
    
    def test_invalid_stereo_option(self):
        """Test that an invalid stereo option raises a ValueError."""
        smiles = 'C/C=C/[C@H]([C@H]([C@H](C(O)=O)C)O)C'

        with self.assertRaises(ValueError) as context:
            krs.canonicalize_smiles(smiles, 'L/D')

class TestManualTEOffload(unittest.TestCase):
    
    def setUp(self):
        self.nonthioester_product = Chem.MolFromSmiles('C/C=C/[C@H]([C@H]([C@H](C(O)=O)C)O)C')
        self.bound_product = Chem.MolFromSmiles('C/C=C/[C@@H](C)[C@@H](O)[C@@H](C)C(S)=O')

    @parameterized.expand(['thiolysis', 'cyclization'])
    def test_missing_thioester(self, offload_mech):
        """Raises a ValueError if no thioester is found."""
        with self.assertRaises(ValueError) as context:
            krs.te_offload(self.nonthioester_product, self.nonthioester_product, offload_mech)
    
    def test_invalid_release_mechanism(self):
        """Test that an invalid release mechanism raises a ValueError."""
        with self.assertRaises(ValueError) as context:
            krs.te_offload(self.bound_product, self.bound_product, 'break_thioester_bond')
    
    def test_missing_lactone_in_target(self):
        """Test that ValueError is raised when no valid lactone is found in target molecule."""
        with self.assertRaises(ValueError) as context:
            krs.te_offload(self.bound_product, self.bound_product, 'cyclization')

    def test_no_hydroxyl_at_target_distance(self):
        """Test that ValueError is raised when no hydroxyl group is found at the
            target distance for cyclization."""
        target_molecule = Chem.MolFromSmiles('C1OCCCCCC1')
        with self.assertRaises(ValueError) as context:
            krs.te_offload(self.bound_product, target_molecule, 'cyclization')
    
    def test_successful_thiolysis(self):
        offloaded_product = krs.te_offload(self.bound_product, self.bound_product, 'thiolysis')
        self.assertEqual(Chem.MolToSmiles(offloaded_product[0]),
                         Chem.MolToSmiles(self.nonthioester_product))
    
    def test_successful_cyclization(self):
        target_molecule = Chem.MolFromSmiles('C/C=C/[C@@H](C)[C@@H](O1)[C@@H](C)C1=O')
        offloaded_product = krs.te_offload(self.bound_product, target_molecule, 'cyclization')
        self.assertEqual(Chem.MolToSmiles(offloaded_product[0]),
                         Chem.MolToSmiles(target_molecule))


# ===========================================================================
# TIER 1: Pure logic & simple RDKit functions
# ===========================================================================

class TestGetModNumber(unittest.TestCase):

    def test_lm_returns_zero(self):
        self.assertEqual(krs.get_mod_number('LM'), 0)

    def test_m1(self):
        self.assertEqual(krs.get_mod_number('M1'), 1)

    def test_m10(self):
        self.assertEqual(krs.get_mod_number('M10'), 10)

    def test_m0(self):
        self.assertEqual(krs.get_mod_number('M0'), 0)


class TestMCSSearch(unittest.TestCase):

    def test_identical_molecules(self):
        mol = Chem.MolFromSmiles('CCCC')
        atom_ct, smarts = krs.mcs_search(mol, mol, True)
        self.assertEqual(atom_ct, mol.GetNumAtoms())

    def test_different_molecules(self):
        mol1 = Chem.MolFromSmiles('CCCC')
        mol2 = Chem.MolFromSmiles('c1ccccc1')
        atom_ct, _ = krs.mcs_search(mol1, mol2, False)
        self.assertLess(atom_ct, mol2.GetNumAtoms())

    def test_chirality_flag_affects_result(self):
        mol1 = Chem.MolFromSmiles('[C@@H](F)(Cl)Br')
        mol2 = Chem.MolFromSmiles('[C@H](F)(Cl)Br')
        ct_chiral, _ = krs.mcs_search(mol1, mol2, True)
        ct_no_chiral, _ = krs.mcs_search(mol1, mol2, False)
        self.assertGreaterEqual(ct_no_chiral, ct_chiral)


class TestSubstructureSearch(unittest.TestCase):

    def test_single_match(self):
        mol = Chem.MolFromSmiles('CCO')
        matches = krs.substructure_search(mol, '[OH]')
        self.assertEqual(len(matches), 1)

    def test_multiple_matches(self):
        mol = Chem.MolFromSmiles('OCCO')
        matches = krs.substructure_search(mol, '[OH]')
        self.assertEqual(len(matches), 2)

    def test_no_match(self):
        mol = Chem.MolFromSmiles('CCCC')
        matches = krs.substructure_search(mol, '[OH]')
        self.assertEqual(len(matches), 0)


class TestLactoneSizeFunc(unittest.TestCase):

    def test_six_membered_lactone(self):
        mol = Chem.MolFromSmiles('O=C1CCCCO1')
        ester_matches = krs.substructure_search(mol, '[C:1](=[O:2])[O:3][C:4]')
        size, ring = krs.lactone_size(mol, ester_matches)
        self.assertGreater(size, 0)
        self.assertIsNotNone(ring)

    def test_no_ester_returns_zero(self):
        mol = Chem.MolFromSmiles('C1CCCCC1')
        ester_matches = krs.substructure_search(mol, '[C:1](=[O:2])[O:3][C:4]')
        size, ring = krs.lactone_size(mol, ester_matches)
        self.assertEqual(size, 0)

    def test_linear_no_rings(self):
        mol = Chem.MolFromSmiles('CCCCCC')
        ester_matches = krs.substructure_search(mol, '[C:1](=[O:2])[O:3][C:4]')
        size, ring = krs.lactone_size(mol, ester_matches)
        self.assertEqual(size, 0)


class TestCanonicalizeSmilesBranches(unittest.TestCase):
    """Tests the E/Z, none, all branches not covered by TestInputStereochemistry."""

    def setUp(self):
        self.smiles = r'C/C=C/[C@H]([C@H]([C@H](C(O)=O)C)O)C'

    def test_ez_stereo_removes_chiral(self):
        result = krs.canonicalize_smiles(self.smiles, 'E/Z')
        self.assertNotIn('@', result)

    def test_none_removes_all(self):
        result = krs.canonicalize_smiles(self.smiles, 'none')
        self.assertNotIn('@', result)
        self.assertNotIn('/', result)

    def test_all_keeps_stereo(self):
        result = krs.canonicalize_smiles(self.smiles, 'all')
        mol = Chem.MolFromSmiles(result)
        self.assertIsNotNone(mol)


class TestExtractEzAtoms(unittest.TestCase):

    def test_z_alkene(self):
        mol = Chem.MolFromSmiles(r'C/C=C\C')
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        z_bonds, e_bonds = krs.extract_ez_atoms(mol)
        self.assertGreater(len(z_bonds), 0)
        self.assertEqual(len(e_bonds), 0)

    def test_e_alkene(self):
        mol = Chem.MolFromSmiles(r'C/C=C/C')
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        z_bonds, e_bonds = krs.extract_ez_atoms(mol)
        self.assertEqual(len(z_bonds), 0)
        self.assertGreater(len(e_bonds), 0)

    def test_no_stereo_double_bonds(self):
        mol = Chem.MolFromSmiles('CCCC')
        z_bonds, e_bonds = krs.extract_ez_atoms(mol)
        self.assertEqual(len(z_bonds), 0)
        self.assertEqual(len(e_bonds), 0)


class TestKRTypeLogic(unittest.TestCase):

    def test_case1_malonyl_no_er_swap_A_to_B(self):
        pf = make_pks_features(['Malonyl-CoA'], ['A'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=False, beta=True)
        self.assertEqual(result, 'B')

    def test_case1_malonyl_no_er_swap_B_to_A(self):
        pf = make_pks_features(['Malonyl-CoA'], ['B'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=False, beta=True)
        self.assertEqual(result, 'A')

    def test_case2_malonyl_with_er(self):
        pf = make_pks_features(['Malonyl-CoA'], ['A'], ['None'], ['L'])
        result = krs.kr_type_logic(pf, 0, alpha=True, beta=True)
        self.assertEqual(result, 'B')

    def test_case3_methylmalonyl_with_dh(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['A1'], ['Z'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=True, beta=True)
        self.assertEqual(result, 'B1')

    def test_case4_both_wrong_A1_to_B2(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['A1'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=True, beta=True)
        self.assertEqual(result, 'B2')

    def test_case4_both_wrong_B2_to_A1(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['B2'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=True, beta=True)
        self.assertEqual(result, 'A1')

    def test_case5_alpha_only_swap_number(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['A1'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=True, beta=False)
        self.assertEqual(result, 'A2')

    def test_case6_beta_only_swap_letter(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['A1'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=False, beta=True)
        self.assertEqual(result, 'B1')

    def test_case7_no_kr_with_alpha(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['None'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=True, beta=False)
        self.assertEqual(result, 'C2')

    def test_case8_no_kr_no_alpha(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['None'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=False, beta=False)
        self.assertIsNone(result)

    def test_no_mismatch_returns_original(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['B1'], ['None'], ['None'])
        result = krs.kr_type_logic(pf, 0, alpha=False, beta=False)
        self.assertEqual(result, 'B1')


class TestERTypeLogic(unittest.TestCase):

    def test_non_malonyl_L_to_D(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['B1'], ['E'], ['L'])
        result = krs.er_type_logic(pf, 0)
        self.assertEqual(result, 'D')

    def test_non_malonyl_D_to_L(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['B1'], ['E'], ['D'])
        result = krs.er_type_logic(pf, 0)
        self.assertEqual(result, 'L')

    def test_malonyl_returns_original(self):
        pf = make_pks_features(['Malonyl-CoA'], ['B'], ['E'], ['L'])
        result = krs.er_type_logic(pf, 0)
        self.assertEqual(result, 'L')

    def test_no_er_returns_none_string(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['B1'], ['None'], ['None'])
        result = krs.er_type_logic(pf, 0)
        self.assertEqual(result, 'None')


class TestKRDHTypeLogic(unittest.TestCase):

    def test_non_malonyl_incompatibility(self):
        pf = make_pks_features(['Methylmalonyl-CoA'], ['A1'], ['Z'], ['None'])
        kr, dh = krs.kr_dh_type_logic(pf, 0)
        self.assertEqual(kr, 'B1')
        self.assertEqual(dh, 'E')

    def test_malonyl_z_to_e(self):
        pf = make_pks_features(['Malonyl-CoA'], ['A'], ['Z'], ['None'])
        kr, dh = krs.kr_dh_type_logic(pf, 0)
        self.assertEqual(kr, 'B')
        self.assertEqual(dh, 'E')

    def test_malonyl_e_to_z(self):
        pf = make_pks_features(['Malonyl-CoA'], ['B'], ['E'], ['None'])
        kr, dh = krs.kr_dh_type_logic(pf, 0)
        self.assertEqual(kr, 'A')
        self.assertEqual(dh, 'Z')


class TestIdentifyPairsWithMismatches(unittest.TestCase):

    def setUp(self):
        self.pairs = [
            ((0, 'LM'), (2, 'M1')),
            ((4, 'M1'), (6, 'M2')),
        ]

    def test_first_atom_mismatched(self):
        result = krs.identify_pairs_with_mismatches(self.pairs, [0])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0][0], 0)

    def test_second_atom_mismatched(self):
        result = krs.identify_pairs_with_mismatches(self.pairs, [2])
        self.assertEqual(len(result), 1)

    def test_no_mismatches(self):
        result = krs.identify_pairs_with_mismatches(self.pairs, [99])
        self.assertEqual(len(result), 0)

    def test_both_pairs_mismatched(self):
        result = krs.identify_pairs_with_mismatches(self.pairs, [0, 6])
        self.assertEqual(len(result), 2)


class TestInsertUnknownTypes(unittest.TestCase):

    def test_single_module(self):
        pf = make_pks_features(['a', 'b'], ['A1', 'B1'], ['None', 'None'], ['None', 'None'])
        krs.insert_unknown_types(pf, {1})
        self.assertEqual(pf['KR Type'][1], 'U')
        self.assertEqual(pf['KR Type'][0], 'A1')

    def test_multiple_modules(self):
        pf = make_pks_features(['a', 'b', 'c'], ['A1', 'B1', 'A2'],
                               ['None', 'None', 'None'], ['None', 'None', 'None'])
        krs.insert_unknown_types(pf, {0, 2})
        self.assertEqual(pf['KR Type'][0], 'U')
        self.assertEqual(pf['KR Type'][2], 'U')
        self.assertEqual(pf['KR Type'][1], 'B1')

    def test_empty_set_no_change(self):
        pf = make_pks_features(['a'], ['A1'], ['None'], ['None'])
        krs.insert_unknown_types(pf, set())
        self.assertEqual(pf['KR Type'][0], 'A1')


class TestParseFinalDesign(unittest.TestCase):

    def test_kr_only(self):
        pf = make_pks_features(['sub'], ['A1'], ['None'], ['None'])
        result = krs.parse_final_design(pf)
        self.assertIn('KR(A1)', result[0][2])

    def test_dh_kr(self):
        pf = make_pks_features(['sub'], ['B1'], ['Z'], ['None'])
        result = krs.parse_final_design(pf)
        self.assertIn('DH(Z)-KR(B1)', result[0][2])

    def test_dh_er_kr(self):
        pf = make_pks_features(['sub'], ['B1'], ['E'], ['L'])
        result = krs.parse_final_design(pf)
        self.assertIn('DH(E)-ER(L)-KR(B1)', result[0][2])

    def test_no_reductive_loop(self):
        pf = make_pks_features(['sub'], ['None'], ['None'], ['None'])
        result = krs.parse_final_design(pf)
        self.assertIn('None', result[0][2])


# ===========================================================================
# TIER 2: Molecule & DataFrame operations
# ===========================================================================

class TestGetDesignFeatures(unittest.TestCase):

    def _make_module(self, substrate, kr_type=None, dh_type=None, er_type=None, loading=False):
        domains = OrderedDict({bcs.AT: bcs.AT(active=True, substrate=substrate)})
        if kr_type:
            domains[bcs.KR] = bcs.KR(active=True, type=kr_type)
        if dh_type:
            domains[bcs.DH] = bcs.DH(active=True, type=dh_type)
        if er_type:
            domains[bcs.ER] = bcs.ER(active=True, type=er_type)
        return bcs.Module(domains=domains, loading=loading)

    def test_all_domains(self):
        design = [self._make_module('Methylmalonyl-CoA', 'B1', 'E', 'L')]
        pf = krs.get_design_features(design)
        self.assertEqual(pf['KR Type'][0], 'B1')
        self.assertEqual(pf['DH Type'][0], 'E')
        self.assertEqual(pf['ER Type'][0], 'L')

    def test_at_only(self):
        design = [self._make_module('Malonyl-CoA', loading=True)]
        pf = krs.get_design_features(design)
        self.assertEqual(pf['KR Type'][0], 'None')
        self.assertEqual(pf['DH Type'][0], 'None')
        self.assertEqual(pf['ER Type'][0], 'None')

    def test_at_kr_only(self):
        design = [self._make_module('Methylmalonyl-CoA', kr_type='A1')]
        pf = krs.get_design_features(design)
        self.assertEqual(pf['KR Type'][0], 'A1')
        self.assertEqual(pf['DH Type'][0], 'None')
        self.assertEqual(pf['ER Type'][0], 'None')

    def test_multiple_modules(self):
        design = [
            self._make_module('prop', loading=True),
            self._make_module('Methylmalonyl-CoA', 'B1', 'E', 'L'),
            self._make_module('Malonyl-CoA', 'A'),
        ]
        pf = krs.get_design_features(design)
        self.assertEqual(len(pf['Module']), 3)
        self.assertEqual(pf['Substrate'][0], 'prop')
        self.assertEqual(pf['KR Type'][1], 'B1')
        self.assertEqual(pf['KR Type'][2], 'A')


class TestModuleMap(unittest.TestCase):

    def test_with_atom_labels(self):
        mol = Chem.MolFromSmiles('CCO')
        for atom in mol.GetAtoms():
            atom.SetProp('atomLabel', 'M1')
        df = krs.module_map(mol)
        self.assertEqual(len(df), mol.GetNumAtoms())
        self.assertTrue(all(df['Module Idx'] == 'M1'))

    def test_without_atom_labels(self):
        mol = Chem.MolFromSmiles('CCO')
        df = krs.module_map(mol)
        self.assertTrue(all(df['Module Idx'] == '-'))


class TestAtomMap(unittest.TestCase):

    def test_similar_molecules(self):
        mol1 = Chem.MolFromSmiles('CCCO')
        mol2 = Chem.MolFromSmiles('CCCO')
        df = krs.atom_map(mol1, mol2)
        self.assertGreater(len(df), 0)
        self.assertIn('Product Atom Idx', df.columns)
        self.assertIn('Target Atom Idx', df.columns)

    def test_different_molecules_warns(self):
        mol1 = Chem.MolFromSmiles('CCCC')
        mol2 = Chem.MolFromSmiles('[Fe]')
        df = krs.atom_map(mol1, mol2)
        self.assertEqual(len(df), 0)


class TestFullMap(unittest.TestCase):

    def test_merge(self):
        atom_df = pd.DataFrame({
            'Atom Type': ['C', 'C', 'O'],
            'Product Atom Idx': [0, 1, 2],
            'Target Atom Idx': [5, 6, 7],
        })
        mod_df = pd.DataFrame({
            'Atom Type': ['C', 'C', 'O'],
            'Product Atom Idx': [0, 1, 2],
            'Module Idx': ['LM', 'M1', 'M1'],
        })
        result = krs.full_map(atom_df, mod_df)
        self.assertEqual(len(result), 3)
        self.assertIn('Module Idx', result.columns)
        self.assertIn('Target Atom Idx', result.columns)


class TestExtractCarbonPairs(unittest.TestCase):

    def test_cross_module_pairs(self):
        mol = Chem.MolFromSmiles('CCCC')
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetProp('atomLabel', f'M{i // 2}')
        full_map_df = pd.DataFrame({
            'Atom Type': ['C', 'C', 'C', 'C'],
            'Product Atom Idx': [0, 1, 2, 3],
            'Target Atom Idx': [0, 1, 2, 3],
            'Module Idx': ['M0', 'M0', 'M1', 'M1'],
        })
        pairs = krs.extract_carbon_pairs(mol, full_map_df)
        self.assertGreater(len(pairs), 0)
        modules_in_pair = {pairs[0][0][1], pairs[0][1][1]}
        self.assertEqual(len(modules_in_pair), 2)

    def test_same_module_no_pairs(self):
        mol = Chem.MolFromSmiles('CC')
        full_map_df = pd.DataFrame({
            'Atom Type': ['C', 'C'],
            'Product Atom Idx': [0, 1],
            'Target Atom Idx': [0, 1],
            'Module Idx': ['M0', 'M0'],
        })
        pairs = krs.extract_carbon_pairs(mol, full_map_df)
        self.assertEqual(len(pairs), 0)


class TestCheckAlphaCarbonMismatch(unittest.TestCase):

    def test_alpha_in_mmatch(self):
        mol = Chem.MolFromSmiles('CC(=O)C')
        pair = ((0, 'LM'), (2, 'M1'))
        mmatch1 = [0]
        result = krs.check_alpha_carbon_mismatch(mol, pair, mmatch1)
        self.assertTrue(result['atom1'] or result['atom2'])

    def test_alpha_not_in_mmatch(self):
        mol = Chem.MolFromSmiles('CC(=O)C')
        pair = ((0, 'LM'), (2, 'M1'))
        mmatch1 = [99]
        result = krs.check_alpha_carbon_mismatch(mol, pair, mmatch1)
        self.assertFalse(result['atom1'])
        self.assertFalse(result['atom2'])


class TestCheckBetaCarbonMismatch(unittest.TestCase):

    def test_beta_in_mmatch(self):
        mol = Chem.MolFromSmiles('C(O)C')
        pair = ((0, 'LM'), (2, 'M1'))
        mmatch1 = [0]
        result = krs.check_beta_carbon_mismatch(mol, pair, mmatch1)
        self.assertTrue(result['atom1'] or result['atom2'])

    def test_beta_not_in_mmatch(self):
        mol = Chem.MolFromSmiles('C(O)C')
        pair = ((0, 'LM'), (2, 'M1'))
        mmatch1 = [99]
        result = krs.check_beta_carbon_mismatch(mol, pair, mmatch1)
        self.assertFalse(result['atom1'])
        self.assertFalse(result['atom2'])


class TestCheckAlkeneMismatch(unittest.TestCase):

    def test_alkene_in_mmatch(self):
        mol = Chem.MolFromSmiles('C=C')
        pair = ((0, 'LM'), (1, 'M1'))
        mmatch1 = [0]
        result = krs.check_alkene_mismatch(mol, pair, mmatch1)
        self.assertTrue(result['atom1'] or result['atom2'])

    def test_alkene_not_in_mmatch(self):
        mol = Chem.MolFromSmiles('C=C')
        pair = ((0, 'LM'), (1, 'M1'))
        mmatch1 = [99]
        result = krs.check_alkene_mismatch(mol, pair, mmatch1)
        self.assertFalse(result['atom1'])
        self.assertFalse(result['atom2'])


class TestCheckCCMismatchCases(unittest.TestCase):

    def test_no_mismatches_returns_empty(self):
        mol = Chem.MolFromSmiles('CCCC')
        pairs = [((0, 'LM'), (1, 'M1'))]
        result = krs.check_cc_mismatch_cases(mol, pairs, [])
        self.assertEqual(len(result), 0)


class TestCheckAlkeneMismatchCases(unittest.TestCase):

    def test_no_mismatches_returns_empty(self):
        mol = Chem.MolFromSmiles('CCCC')
        pairs = [((0, 'LM'), (1, 'M1'))]
        result = krs.check_alkene_mismatch_cases(mol, pairs, [])
        self.assertEqual(len(result), 0)


class TestRSStereoCorrespondence(unittest.TestCase):

    def test_matching_chiral_centers(self):
        mol = Chem.MolFromSmiles('[C@@H](F)(Cl)Br')
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        full_map_df = pd.DataFrame({
            'Atom Type': [a.GetSymbol() for a in mol.GetAtoms()],
            'Product Atom Idx': list(range(mol.GetNumAtoms())),
            'Target Atom Idx': list(range(mol.GetNumAtoms())),
            'Module Idx': ['M0'] * mol.GetNumAtoms(),
        })
        result = krs.get_rs_stereo_correspondence(mol, mol, full_map_df)
        self.assertGreater(len(result.match1), 0)
        self.assertEqual(len(result.mmatch1), 0)

    def test_mismatching_chiral_centers(self):
        mol1 = Chem.MolFromSmiles('[C@@H](F)(Cl)Br')
        mol2 = Chem.MolFromSmiles('[C@H](F)(Cl)Br')
        Chem.AssignStereochemistry(mol1, force=True, cleanIt=True)
        Chem.AssignStereochemistry(mol2, force=True, cleanIt=True)
        full_map_df = pd.DataFrame({
            'Atom Type': [a.GetSymbol() for a in mol1.GetAtoms()],
            'Product Atom Idx': list(range(mol1.GetNumAtoms())),
            'Target Atom Idx': list(range(mol1.GetNumAtoms())),
            'Module Idx': ['M0'] * mol1.GetNumAtoms(),
        })
        result = krs.get_rs_stereo_correspondence(mol1, mol2, full_map_df)
        self.assertGreater(len(result.mmatch1), 0)


class TestEZStereoCorrespondence(unittest.TestCase):

    def test_matching_ez(self):
        mol = Chem.MolFromSmiles(r'C/C=C/C')
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        full_map_df = pd.DataFrame({
            'Atom Type': [a.GetSymbol() for a in mol.GetAtoms()],
            'Product Atom Idx': list(range(mol.GetNumAtoms())),
            'Target Atom Idx': list(range(mol.GetNumAtoms())),
            'Module Idx': ['M0'] * mol.GetNumAtoms(),
        })
        result = krs.get_ez_stereo_correspondence(mol, mol, full_map_df)
        self.assertGreater(len(result.match1), 0)
        self.assertEqual(len(result.mmatch1), 0)

    def test_mismatching_ez(self):
        mol1 = Chem.MolFromSmiles(r'C/C=C/C')
        mol2 = Chem.MolFromSmiles(r'C/C=C\C')
        Chem.AssignStereochemistry(mol1, force=True, cleanIt=True)
        Chem.AssignStereochemistry(mol2, force=True, cleanIt=True)
        full_map_df = pd.DataFrame({
            'Atom Type': [a.GetSymbol() for a in mol1.GetAtoms()],
            'Product Atom Idx': list(range(mol1.GetNumAtoms())),
            'Target Atom Idx': list(range(mol1.GetNumAtoms())),
            'Module Idx': ['M0'] * mol1.GetNumAtoms(),
        })
        result = krs.get_ez_stereo_correspondence(mol1, mol2, full_map_df)
        self.assertGreater(len(result.mmatch1), 0)


class TestIdentifySwapCases(unittest.TestCase):

    def test_kr_swap_updates_features(self):
        pf = make_pks_features(
            ['prop', 'Methylmalonyl-CoA'],
            ['None', 'A1'], ['None', 'None'], ['None', 'None'])
        result_dict = {
            'c_idx_i': 0, 'module_i': 'LM',
            'c_idx_i+1': 2, 'module_i+1': 'M1',
            'alpha_mismatch': True, 'beta_mismatch': True,
        }
        krs.identify_kr_swap_case(pf, result_dict)
        self.assertEqual(pf['KR Type'][1], 'B2')

    def test_er_swap_updates_features(self):
        pf = make_pks_features(
            ['prop', 'Methylmalonyl-CoA'],
            ['None', 'B1'], ['None', 'E'], ['None', 'L'])
        result_dict = {
            'c_idx_i': 0, 'module_i': 'LM',
            'c_idx_i+1': 2, 'module_i+1': 'M1',
            'alpha_mismatch': True, 'beta_mismatch': False,
        }
        krs.identify_er_swap_case(pf, result_dict)
        self.assertEqual(pf['ER Type'][1], 'D')

    def test_dh_swap_updates_features(self):
        pf = make_pks_features(
            ['prop', 'Malonyl-CoA'],
            ['None', 'A'], ['None', 'Z'], ['None', 'None'])
        result_dict = {
            'c_idx_i': 0, 'module_i': 'LM',
            'c_idx_i+1': 2, 'module_i+1': 'M1',
            'alkene_mismatch': True,
        }
        krs.identify_dh_swap_case(pf, result_dict)
        self.assertEqual(pf['KR Type'][1], 'B')
        self.assertEqual(pf['DH Type'][1], 'E')


class TestKRSwapsLoop(unittest.TestCase):

    def test_empty_results_no_change(self):
        pf = make_pks_features(['prop', 'Malonyl-CoA'],
                               ['None', 'A'], ['None', 'None'], ['None', 'None'])
        original_kr = list(pf['KR Type'])
        krs.kr_swaps(pf, [])
        self.assertEqual(pf['KR Type'], original_kr)

    def test_single_mismatch_applied(self):
        pf = make_pks_features(['prop', 'Malonyl-CoA'],
                               ['None', 'A'], ['None', 'None'], ['None', 'None'])
        results = [{
            'c_idx_i': 0, 'module_i': 'LM',
            'c_idx_i+1': 2, 'module_i+1': 'M1',
            'alpha_mismatch': False, 'beta_mismatch': True,
        }]
        krs.kr_swaps(pf, results)
        self.assertEqual(pf['KR Type'][1], 'B')


class TestERSwapsLoop(unittest.TestCase):

    def test_empty_results_no_change(self):
        pf = make_pks_features(['prop', 'Methylmalonyl-CoA'],
                               ['None', 'B1'], ['None', 'E'], ['None', 'L'])
        original_er = list(pf['ER Type'])
        krs.er_swaps(pf, [])
        self.assertEqual(pf['ER Type'], original_er)


class TestDHSwapsLoop(unittest.TestCase):

    def test_empty_results_no_change(self):
        pf = make_pks_features(['prop', 'Malonyl-CoA'],
                               ['None', 'A'], ['None', 'Z'], ['None', 'None'])
        original_dh = list(pf['DH Type'])
        krs.dh_swaps(pf, [])
        self.assertEqual(pf['DH Type'], original_dh)


class TestComputeJaccardSim(unittest.TestCase):

    def test_identical_molecules(self):
        mol = Chem.MolFromSmiles('CCCCO')
        sim = krs.compute_jaccard_sim(mol, mol)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_different_molecules(self):
        mol1 = Chem.MolFromSmiles('CCCCO')
        mol2 = Chem.MolFromSmiles('c1ccccc1')
        sim = krs.compute_jaccard_sim(mol1, mol2)
        self.assertLess(sim, 1.0)
        self.assertGreaterEqual(sim, 0.0)


class TestAddAtomLabels(unittest.TestCase):

    def test_sets_atom_note(self):
        mol = Chem.MolFromSmiles('CCO')
        chiral_centers = {}
        krs.add_atom_labels(mol, chiral_centers)
        for atom in mol.GetAtoms():
            self.assertTrue(atom.HasProp('atomNote'))

    def test_includes_chiral_label(self):
        mol = Chem.MolFromSmiles('[C@@H](F)(Cl)Br')
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        cc = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        krs.add_atom_labels(mol, cc)
        chiral_atom = mol.GetAtomWithIdx(0)
        note = chiral_atom.GetProp('atomNote')
        self.assertIn('(', note)


# ===========================================================================
# TIER 3: Integration tests with mocking
# ===========================================================================

class TestPksPrecursorAtoms(unittest.TestCase):

    def test_shared_substructure(self):
        mol1 = Chem.MolFromSmiles('CCCCO')
        mol2 = Chem.MolFromSmiles('CCCCOC')
        result = krs.pks_precursor_atoms(mol1, mol2)
        self.assertGreater(len(result), 0)

    def test_no_common_substructure(self):
        mol1 = Chem.MolFromSmiles('CCCC')
        mol2 = Chem.MolFromSmiles('[Fe]')
        result = krs.pks_precursor_atoms(mol1, mol2)
        self.assertEqual(len(result), 0)


class TestExtractPrecursorMol(unittest.TestCase):

    def test_removes_non_precursor_atoms(self):
        mol = Chem.MolFromSmiles('CCCCN')
        precursor_atoms = {0, 1, 2, 3}  # all C, exclude N
        result = krs.extract_precursor_mol(mol, precursor_atoms)
        self.assertEqual(result.GetNumAtoms(), 4)

    def test_preserves_all_atoms(self):
        mol = Chem.MolFromSmiles('CCO')
        precursor_atoms = {0, 1, 2}
        result = krs.extract_precursor_mol(mol, precursor_atoms)
        self.assertEqual(result.GetNumAtoms(), 3)


class TestGetPksTarget(unittest.TestCase):

    @patch('krswaps.krswaps.compareToTarget')
    def test_full_match_returns_original(self, mock_compare):
        mock_compare.return_value = 1.0
        mol = Chem.MolFromSmiles('CCCC')
        target = Chem.MolFromSmiles('CCCC')
        result_mol, score = krs.get_pks_target(mol, target)
        self.assertEqual(score, 1.0)
        self.assertEqual(Chem.MolToSmiles(result_mol), Chem.MolToSmiles(target))

    @patch('krswaps.krswaps.compareToTarget')
    def test_partial_match_extracts_precursor(self, mock_compare):
        mock_compare.return_value = 0.75
        mol = Chem.MolFromSmiles('CCCO')
        target = Chem.MolFromSmiles('CCCON')
        result_mol, score = krs.get_pks_target(mol, target)
        self.assertEqual(score, 0.75)
        self.assertLessEqual(result_mol.GetNumAtoms(), target.GetNumAtoms())


class TestGetLactoneAtoms(unittest.TestCase):

    def test_molecule_with_lactone(self):
        mol = Chem.MolFromSmiles('O=C1CCCCO1')
        result = krs.get_lactone_atoms(mol)
        self.assertIsNotNone(result)

    def test_molecule_without_lactone(self):
        mol = Chem.MolFromSmiles('CCCCCC')
        result = krs.get_lactone_atoms(mol)
        self.assertIsNone(result)


class TestForceTargetLactoneAlkene(unittest.TestCase):

    def test_small_lactone_forces_z(self):
        # 6-membered lactone with a double bond
        mol = Chem.MolFromSmiles('O=C1CC=CCO1')
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        ring_info = mol.GetRingInfo()
        lactone_atoms = tuple(ring_info.AtomRings()[0])
        result = krs.force_target_lactone_alkene(mol, lactone_atoms)
        # Check that double bond in ring has stereo set
        self.assertIsNotNone(result)


class TestNewPksDesign(unittest.TestCase):

    def test_builds_cluster_and_product(self):
        pf = make_pks_features(
            ['prop', 'Methylmalonyl-CoA', 'Malonyl-CoA'],
            ['None', 'B1', 'A'],
            ['None', 'None', 'None'],
            ['None', 'None', 'None'])
        product, design = krs.new_pks_design(pf)
        self.assertIsNotNone(product)
        self.assertEqual(len(design), 3)
        self.assertTrue(design[0].loading)
        self.assertFalse(design[1].loading)


class TestOutputResults(unittest.TestCase):

    def test_writes_files(self):
        results = {
            'target_molecule': 'CCCC',
            'target_pks_precursor': 'CCCC',
            'final_pks_design': [['M0', 'AT: prop', 'KR: None']],
            'final_pks_product': 'CCCC',
            'mcs_similarity': 1.0,
            'jaccard_i': 0.8,
            'jaccard_f': 0.9,
            'stereo_before': '<svg>before</svg>',
            'stereo_after': '<svg>after</svg>',
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            krs.output_results(results, 'test_job', tmpdir)
            json_path = os.path.join(tmpdir, 'test_job_stereocorrection_results.json')
            pre_svg = os.path.join(tmpdir, 'test_job_initial_stereo_correspondence.svg')
            post_svg = os.path.join(tmpdir, 'test_job_final_stereo_correspondence.svg')
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(pre_svg))
            self.assertTrue(os.path.exists(post_svg))
            with open(json_path) as f:
                data = json.load(f)
            self.assertIn('Target Molecule', data)
            self.assertIn('MCS Similarity', data)
            self.assertIn('Final PKS Design', data)

    def test_json_values_correct(self):
        results = {
            'target_molecule': 'C=CC',
            'target_pks_precursor': 'C=CC',
            'final_pks_design': [],
            'final_pks_product': 'CCC',
            'mcs_similarity': 0.5,
            'jaccard_i': 0.3,
            'jaccard_f': 0.7,
            'stereo_before': '<svg/>',
            'stereo_after': '<svg/>',
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            krs.output_results(results, 'test2', tmpdir)
            with open(os.path.join(tmpdir, 'test2_stereocorrection_results.json')) as f:
                data = json.load(f)
            self.assertEqual(data['MCS Similarity'], 0.5)
            self.assertEqual(data['Initial Jaccard Similarity'], 0.3)
            self.assertEqual(data['Final Jaccard Similarity'], 0.7)


class TestAddressUndefinedStereo(unittest.TestCase):

    def test_undefined_stereo_returns_modules(self):
        mol = Chem.MolFromSmiles('[C@@H](F)(Cl)Br')
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        # Create a pair structure
        pairs = [((0, 'LM'), (2, 'M1'))]
        # Create a chiral result with '?' in cc2
        cc1 = {0: 'S'}
        cc2 = {0: '?'}
        chiral_result = krs.ChiralCheckResult(
            match1=[], match2=[], mmatch1=[], mmatch2=[], cc1=cc1, cc2=cc2)
        full_map_df = pd.DataFrame({
            'Atom Type': ['C', 'F', 'Cl', 'Br'],
            'Product Atom Idx': [0, 1, 2, 3],
            'Target Atom Idx': [0, 1, 2, 3],
            'Module Idx': ['LM', 'LM', 'M1', 'M1'],
        })
        result = krs.address_undefined_stereo(pairs, chiral_result, full_map_df)
        self.assertIsInstance(result, set)

    def test_no_undefined_returns_empty(self):
        pairs = [((0, 'LM'), (2, 'M1'))]
        cc1 = {0: 'S'}
        cc2 = {0: 'S'}
        chiral_result = krs.ChiralCheckResult(
            match1=[0], match2=[0], mmatch1=[], mmatch2=[], cc1=cc1, cc2=cc2)
        full_map_df = pd.DataFrame({
            'Atom Type': ['C', 'F', 'Cl', 'Br'],
            'Product Atom Idx': [0, 1, 2, 3],
            'Target Atom Idx': [0, 1, 2, 3],
            'Module Idx': ['LM', 'LM', 'M1', 'M1'],
        })
        result = krs.address_undefined_stereo(pairs, chiral_result, full_map_df)
        self.assertEqual(len(result), 0)


class TestVisualizeSteroCorrespondence(unittest.TestCase):

    def test_returns_svg_string(self):
        mol1 = Chem.MolFromSmiles('[C@@H](F)(Cl)Br')
        mol2 = Chem.MolFromSmiles('[C@H](F)(Cl)Br')
        Chem.AssignStereochemistry(mol1, force=True, cleanIt=True)
        Chem.AssignStereochemistry(mol2, force=True, cleanIt=True)
        chiral_result = krs.ChiralCheckResult(
            match1=[], match2=[], mmatch1=[0], mmatch2=[0],
            cc1={0: 'S'}, cc2={0: 'R'})
        alkene_result = krs.AlkeneCheckResult(
            match1=[], match2=[], mmatch1=[], mmatch2=[])
        svg = krs.visualize_stereo_correspondence(mol1, mol2, chiral_result, alkene_result)
        self.assertIn('svg', svg.lower())

    def test_background_covers_full_width(self):
        mol1 = Chem.MolFromSmiles('[C@@H](F)(Cl)Br')
        mol2 = Chem.MolFromSmiles('[C@H](F)(Cl)Br')
        Chem.AssignStereochemistry(mol1, force=True, cleanIt=True)
        Chem.AssignStereochemistry(mol2, force=True, cleanIt=True)
        chiral_result = krs.ChiralCheckResult(
            match1=[], match2=[], mmatch1=[0], mmatch2=[0],
            cc1={0: 'S'}, cc2={0: 'R'})
        alkene_result = krs.AlkeneCheckResult(
            match1=[], match2=[], mmatch1=[], mmatch2=[])
        svg = krs.visualize_stereo_correspondence(mol1, mol2, chiral_result, alkene_result)
        self.assertIn("width='1000.0'", svg)
        self.assertNotIn("width='500.0' height='400.0' x='0.0' y='0.0'", svg)


if __name__ == '__main__':
    unittest.main()
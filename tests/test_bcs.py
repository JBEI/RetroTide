import unittest
from parameterized import parameterized
import pytest
from unittest.mock import patch, MagicMock
from bcs import *
from rdkit import Chem
from cobra import Metabolite

import unittest
from unittest.mock import MagicMock, patch

class TestCluster(unittest.TestCase):
    def setUp(self):
        self.te_domain = TE(True, True, True)
        self.dh_domain = DH(True, 'E')
        self.at_domain = AT(True, substrate="Acetyl-CoA")
        self.module_at = Module(domains={AT: self.at_domain})
        self.module_te = Module(domains={TE: self.te_domain})
        self.module_dh = Module(domains={DH: self.dh_domain})
        self.structureDB = MagicMock()

    def test_initialization_without_modules(self):
        """Test the __init__ method initializes an empty list when no modules are provided."""
        cluster = Cluster()
        self.assertIsInstance(cluster.modules, list, "Modules should be initialized to an empty list.")
        self.assertEqual(len(cluster.modules), 0, "Modules list should be empty when no modules are provided.")

    def test_initialization_with_modules(self):
        """Test the __init__ method correctly assigns the modules list when provided."""
        # Create example modules
        dh_module = Module(domains=OrderedDict([(DH, DH(True, 'E'))]))
        er_module = Module(domains=OrderedDict([(ER, ER(True, 'L'))]))
        modules_list = [dh_module, er_module]

        cluster = Cluster(modules=modules_list)
        self.assertEqual(cluster.modules, modules_list, "Modules should be initialized with the provided list.")
        self.assertEqual(len(cluster.modules), 2, "Modules list should contain two items.")

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    def test_empty_cluster(self, mock_reaction):
        """Test computeProduct with no modules."""
        cluster = Cluster()
        result = cluster.computeProduct(self.structureDB)
        self.assertFalse(result)
        mock_reaction.assert_not_called()

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    def test_empty_cluster_with_chain_returns_initial_chain(self, mock_reaction):
        """Test computeProduct with no modules and an initial chain passed in. Should output the initial chain."""
        initial_chain = "Initial Chain"
        cluster = Cluster()
        result = cluster.computeProduct(self.structureDB, initial_chain)
        self.assertEqual(initial_chain, result)
        mock_reaction.assert_not_called()

    def test_starter_module_returns_its_chain(self):
        """
        Check that when we encounter an AT domain, we get its corresponding output chain from starters.
        The cluster contains a module with an AT domain.
        """
        self.structureDB = dict()

        # Call computeProduct on the cluster with no initial chain
        cluster = Cluster(modules=[self.module_at])
        result = cluster.computeProduct(self.structureDB)

        self.assertIsInstance(result, Chem.Mol)
        self.assertEqual(Chem.MolToSmiles(result), "CC(=O)[S]", "The output chain should match the AT domain's substrate from starters.")

    def test_te_domain_presence(self):
        """Test computeProduct with a TE domain present."""
        te_mock = MagicMock()
        te_mock.operation.return_value = "TE Chain"
        module_te = Module(domains={TE: te_mock})
        cluster = Cluster(modules=[module_te])
        initial_chain = "Initial Chain"
        result = cluster.computeProduct(self.structureDB, chain=initial_chain)
        self.assertIsNotNone(result)
        self.assertEqual(result, "TE Chain")
        module_te.domains[TE].operation.assert_called_once_with(initial_chain)

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.SanitizeMol')
    def test_compute_product_only_execute_last_module_when_chain_given(self, mock_sanitize, mock_reaction_from_smarts):
        """When a chain is given, ensure only the last module executes RunReactants."""
        # Setup the reaction to produce a specific output and mock SanitizeMol for final output
        # arrange
        module2 = Module(domains={AT: self.at_domain, DH: self.dh_domain})
        reaction_instance = mock_reaction_from_smarts.return_value
        mock_product = MagicMock()
        reaction_instance.RunReactants.return_value = ((mock_product,),)

        # Initial chain to be modified by the last module
        initial_chain = MagicMock()

        structureDB = {}
        structureDB[self.module_dh] = "DH Module Structure"
        structureDB[module2] = "Module 2 Structure"

        cluster = Cluster(modules=[module2, self.module_dh])

        # Act
        # Run computeProduct with an initial chain
        result = cluster.computeProduct(structureDB, chain=initial_chain)
        
        # Assert
        # Ensure that the reaction is called with the chain and the last module's structure
        reaction_instance.RunReactants.assert_called_once_with((initial_chain, structureDB[self.module_dh]))

        # Verify that SanitizeMol is called correctly
        mock_sanitize.assert_called_once_with(mock_product)
        # Check that the final product is as expected
        self.assertEqual(result, mock_product)
    
    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.SanitizeMol')
    def test_compute_products_runs_all_modules_when_no_chain_given(self, mock_sanitize, mock_reaction_from_smarts):
        """
        Test that RunReactants is called on both modules sequentially when no chain is given.
        Verify that the output of the call on the first module (AT) is passed to the second module (DH).
        """
        # Arrange
        reaction_instance = mock_reaction_from_smarts.return_value
        mock_product = MagicMock()
        reaction_instance.RunReactants.return_value = ((mock_product,),)
        
        cluster = Cluster(modules=[self.module_at, self.module_dh])
        structureDB = {self.module_dh: "DH Module Structure"}
        mock_product_mod1 = MagicMock(name='ProductMod1')
        mock_product_mod2 = MagicMock(name='ProductMod2')
        at_mod_chain = starters[self.at_domain.substrate]

        # Act
        result = cluster.computeProduct(structureDB)

        # Assert
        # Ensure RunReactants is called with correct arguments
        reaction_instance.RunReactants.assert_called_once_with((starters[self.at_domain.substrate], structureDB[self.module_dh]))

        mock_sanitize.assert_called_once_with(mock_product)
        # Check that the final product is as expected
        self.assertEqual(result, mock_product)

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.SanitizeMol')
    def test_module_with_TE_will_terminate_chain(self, mock_sanitize, mock_reaction_from_smarts):
        """
        Test that the cluster terminates processing when a TE module is encountered and does not execute any modules beyond it.
        """
        # Arrange
        te_operation_mock = MagicMock()
        self.module_te.domains[TE].operation = te_operation_mock
        te_operation_mock.return_value = 'TE Processed Product'
        cluster = Cluster(modules=[self.module_at, self.module_te, self.module_dh])

        # Create a fake structureDB, not used in this specific test as TE should terminate
        structureDB = {}

        # Act
        result = cluster.computeProduct(structureDB)

        # Assert
        # Ensure that the TE domain's operation was called with the correct product (use the starter chain)
        te_operation_mock.assert_called_once_with(starters[self.at_domain.substrate])

        # Ensure that RunReactants was never called as the TE should terminate processing
        mock_reaction_from_smarts.assert_not_called()
        mock_sanitize.assert_not_called()
        # Ensure that the result is from the TE operation
        self.assertEqual(result, 'TE Processed Product')
        # Verify that the DH module was never invoked
        self.assertNotIn(self.module_dh, structureDB)

class TestModule(unittest.TestCase):
    def setUp(self):
        # Instantiating modules for testing
        self.dh_domain = DH(True, 'E')  # Assuming DH takes an 'active' parameter
        self.er_domain = ER(True, 'L')  # Assuming ER takes an 'active' parameter
        self.module1 = Module(domains=OrderedDict([(DH, self.dh_domain)]), loading=True)
        self.module2 = Module(domains=OrderedDict([(DH, self.dh_domain)]), loading=True)
        self.module3 = Module(domains=OrderedDict([(ER, self.er_domain)]), loading=True)
        self.module4 = "Not a Module"

    def test_initialization_default_values(self):
        """Test the __init__ method with default values."""
        module = Module()
        
        self.assertEqual(module.product, '')
        self.assertEqual(module.iterations, 1)
        self.assertFalse(module.loading)
        self.assertIsInstance(module.domains, OrderedDict)
        self.assertEqual(len(module.domains), 0)

    def test_initialization_with_custom_values(self):
        """Test the __init__ method with custom values and provided domains."""
        domains = OrderedDict([
            ('Domain1', MagicMock()),
            ('Domain2', MagicMock())
        ])
        module = Module(product='TestProduct', iterations=3, domains=domains, loading=True)
        
        self.assertEqual(module.product, 'TestProduct')
        self.assertEqual(module.iterations, 3)
        self.assertTrue(module.loading)
        self.assertEqual(module.domains, domains)
        self.assertEqual(len(module.domains), 2)
    
    def test_hash_consistency(self):
        """Test that the hash of a module is consistent across calls."""
        module = Module(product='TestProduct', iterations=2, domains=OrderedDict([(DH, DH(True, 'E'))]), loading=True)
        hash1 = hash(module)
        hash2 = hash(module)
        self.assertEqual(hash1, hash2, "Hash values should be consistent across multiple invocations.")

    def test_hash_uniqueness(self):
        """Test that different domain configurations yield different hashes."""
        module1 = Module(domains=OrderedDict([(DH, DH(True, 'E'))]), loading=True)
        module2 = Module(domains=OrderedDict([(ER, ER(True, 'L'))]), loading=True)
        self.assertNotEqual(hash(module1), hash(module2), "Different domain configurations should have different hashes.")

    def test_hash_is_same_when_same_configuration(self):
        """Test that modules considered 'equal' have the same hash."""
        # Assuming equality could be defined similarly to hashing criteria
        module1 = Module(domains=OrderedDict([(DH, DH(True, 'E'))]), loading=True)
        module2 = Module(domains=OrderedDict([(DH, DH(True, 'E'))]), loading=True)
        self.assertEqual(hash(module1), hash(module2), "Equal modules should have the same hash.")

    def test_hash_changes_with_attributes(self):
        """Test that changes in attributes used in hash calculation alter the hash."""
        domain_mock = MagicMock()
        module = Module(domains=OrderedDict([('Domain', domain_mock)]), loading=True)
        initial_hash = hash(module)
        # Change an attribute
        module.loading = False
        self.assertNotEqual(initial_hash, hash(module), "Hash should change when attributes change.")

    def test_hash_empty_domains(self):
        """Test hashing behavior with empty domains."""
        module = Module(domains=OrderedDict(), loading=True)
        self.assertIsInstance(hash(module), int, "Hash should be calculable and return an integer even for empty domains.")
    
    def test_hash_produces_different_values_for_same_domains_in_different_order(self):
        """
        Test that hash values are different for modules with the same set of domains
        added in different orders, reflecting the order-sensitivity in hashing.
        """
        # Domain instances
        dh_domain = DH(True, 'E')  # Assuming DH takes an 'active' parameter
        er_domain = ER(True, 'L')  # Assuming ER takes an 'active' parameter

        # Module with domains in one order
        module1 = Module(domains=OrderedDict([(DH, dh_domain), (ER, er_domain)]), loading=True)

        # Module with domains in reverse order
        module2 = Module(domains=OrderedDict([(ER, er_domain), (DH, dh_domain)]), loading=True)

        # Compute hashes
        hash1 = hash(module1)
        hash2 = hash(module2)

        # Assert that the hashes are different
        self.assertNotEqual(hash1, hash2, "Hashes should differ for modules with the same domains in different orders.")
    
    def test_compute_product(self):
        """Test the computeProduct method processes domain operations correctly."""
        # Mock domain classes and instances
        domain1 = MagicMock()
        domain2 = MagicMock()
        domain3 = MagicMock()

        # Setup return values for each domain's operation method
        domain1.operation.return_value = 'Chain1'
        domain2.operation.return_value = 'Chain2'
        domain3.operation.return_value = 'Chain3'

        # Creating an OrderedDict of domains as it would be in a real module
        domains = OrderedDict([
            ('Domain1', domain1),
            ('Domain2', domain2),
            ('Domain3', domain3)
        ])

        # Instantiate the Module with mocked domains
        module = Module(domains=domains)

        # Execute computeProduct which should process each domain's operation in sequence
        result = module.computeProduct()

        # Check that each domain's operation method is called correctly
        domain1.operation.assert_called_once_with(None)  # First domain receives initial False
        domain2.operation.assert_called_once_with('Chain1')  # Second domain receives output of first
        domain3.operation.assert_called_once_with('Chain2')  # Third domain receives output of second

        # Verify the final output is the output of the last domain operation
        self.assertEqual(result, 'Chain3')

    def test_equals_with_same_configuration(self):
        """Test that __eq__ returns True for modules with identical configurations."""
        self.assertTrue(self.module1 == self.module2, "Modules with identical configurations should be equal.")

    def test_equals_with_different_configurations(self):
        """Test that __eq__ returns False for modules with different configurations."""
        self.assertFalse(self.module1 == self.module3, "Modules with different configurations should not be equal.")

    def test_equals_with_different_type(self):
        """Test that __eq__ returns False when compared against different types."""
        self.assertFalse(self.module1 == self.module4, "A module compared to a non-module should not be equal.")

class MockDomain(Domain):
    def __init__(self, active=True, attribute='value'):
        self.active = active
        self.attribute = attribute

class TestDomain(unittest.TestCase):
    def setUp(self):
        self.domain1 = MockDomain(active=True, attribute='value1')
        self.domain2 = MockDomain(active=False, attribute='value1')
        self.domain3 = MockDomain(active=True, attribute='value2')

    def test_design_reports_correctly(self):
        """Test that the design method reports the domain's attributes correctly."""
        expected_design = {'active': True, 'attribute': 'value1'}
        self.assertEqual(self.domain1.design(), expected_design)

    def test_equals_returns_false_for_different_data_type(self):
        """Test __eq__ returns False when comparing with different data types."""
        self.assertFalse(self.domain1 == 123)

    def test_equals_returns_false_for_different_domains(self):
        """Test __eq__ returns False for different domain configurations."""
        self.assertFalse(self.domain1 == self.domain3)

    def test_equals_returns_true_for_same_domain(self):
        """Test __eq__ returns True for identical domain configurations."""
        domain1_copy = MockDomain(active=True, attribute='value1')
        self.assertTrue(self.domain1 == domain1_copy)

    def test_repr_shows_correct_string(self):
        """Test __repr__ shows the correct string based on active status."""
        self.assertEqual(repr(self.domain1), "{'attribute': 'value1'}")
        self.assertEqual(repr(self.domain2), "{'active': False, 'attribute': 'value1'}")

    def test_hash_consistency(self):
        """Test that the hash of a domain is consistent across calls."""
        hash1 = hash(self.domain1)
        hash2 = hash(self.domain1)
        self.assertEqual(hash1, hash2, "Hash values should be consistent across multiple invocations.")

    def test_hash_uniqueness_same_type(self):
        """Test that different domain configurations yield different hashes."""
        hash1 = hash(self.domain1)
        hash2 = hash(self.domain2)
        self.assertNotEqual(hash1, hash2, "Different configurations should have different hashes.")

    def test_hash_stability_across_same_type_instances(self):
        """Test that different instances with the same configuration yield the same hash."""
        hash1 = hash(self.domain1)
        domain1_copy = MockDomain(active=True, attribute='value1')
        hash3 = hash(domain1_copy)
        self.assertEqual(hash1, hash3, "Identical configurations should yield the same hash.")

    def test_hash_changes_with_attributes(self):
        """Test that changes in attributes used in hash calculation alter the hash."""
        domain1_copy = MockDomain(active=True, attribute='value1')
        initial_hash = hash(domain1_copy)
        domain1_copy.attribute = 'value3'  # Change an attribute
        new_hash = hash(domain1_copy)
        self.assertNotEqual(initial_hash, new_hash, "Hash should change when attributes change.")

    def test_hash_changes_with_class_type(self):
        """
        Test that the __hash__ method produces different hash values for instances
        of different classes even when they have identical attribute values.
        """
        # Create instances of different domain classes with the same attribute values
        er_domain = ER(active=True, type='L')
        dh_domain = DH(active=True, type='E')
        
        # Compute hashes for each instance
        er_hash = hash(er_domain)
        dh_hash = hash(dh_domain)
        
        # Assert that hashes are different, indicating class type affects hash calculation
        self.assertNotEqual(er_hash, dh_hash, "Hashes should differ for different class types despite identical attributes.")

class TestAT(unittest.TestCase):
    def setUp(self):
        # Setup for testing AT domain with mock substrate values
        self.at_domain_malonyl = AT(active=True, substrate='Malonyl-CoA')
        self.at_domain_methylmalonyl = AT(active=True, substrate='Methylmalonyl-CoA')

    def test_init(self):
        # Test correct initialization of AT domain instances
        self.assertTrue(self.at_domain_malonyl.active)
        self.assertEqual(self.at_domain_malonyl.substrate, 'Malonyl-CoA')
        self.assertTrue(self.at_domain_methylmalonyl.active)
        self.assertEqual(self.at_domain_methylmalonyl.substrate, 'Methylmalonyl-CoA')

    def test_reactants_malonyl(self):
        # Test reactants method for Malonyl-CoA substrate
        reactants = self.at_domain_malonyl.reactants()
        self.assertEqual(len(reactants), 2)
        self.assertIsInstance(reactants[0], Metabolite)
        self.assertIsInstance(reactants[1], Metabolite)
        self.assertEqual(reactants[0].id, 'h_c')
        self.assertEqual(reactants[1].id, 'malcoa_c')

    def test_reactants_methylmalonyl(self):
        # Test reactants method for Methylmalonyl-CoA substrate
        reactants = self.at_domain_methylmalonyl.reactants()
        self.assertEqual(len(reactants), 2)
        self.assertIsInstance(reactants[0], Metabolite)
        self.assertIsInstance(reactants[1], Metabolite)
        self.assertEqual(reactants[0].id, 'h_c')
        self.assertEqual(reactants[1].id, 'mmcoa__S_c')

    def test_products(self):
        # Test products method
        products = self.at_domain_malonyl.products()
        self.assertEqual(len(products), 2)
        self.assertIsInstance(products[0], Metabolite)
        self.assertIsInstance(products[1], Metabolite)
        self.assertEqual(products[0].id, 'coa_c')
        self.assertEqual(products[1].id, 'co2_c')
    
    def test_operation_returns_molecule_when_loading_is_true_and_no_chain_provided(self):
        # at = AT(active=True, substrate='substrate')
        molecule = self.at_domain_methylmalonyl.operation(None, loading=True)
        assert isinstance(molecule, Chem.Mol)
        expected_smiles = "CCC(=O)[S]" # Methylmalonyl-CoA
        self.assertEqual(Chem.MolToSmiles(molecule), expected_smiles)
    
    def test_operation_not_loading(self):
        at_output = AT(active=True, substrate='Malonyl-CoA').operation(None, loading=False)
        assert isinstance(at_output, Chem.Mol)
        self.assertEqual(extenders['Malonyl-CoA'], at_output)

    def test_operation_unknown_substrate_throws_exception(self):
        return

    def test_operation_throw_exception_when_chain_is_passed(self):
        smiles = "CC(C)C(=O)CC(O)C"
        # Create the Mol object from SMILES
        chain = Chem.MolFromSmiles(smiles)

        self.assertRaises(NotImplementedError, self.at_domain_malonyl.operation, chain, True)
        return

    def test_designspace_loading_returns_starters(self):
        module = Module(loading=True)
        designs = AT.designSpace(module)
        assert len(designs) == len(starters)
        for design in designs:
            self.assertTrue(design.substrate in starters)        
    
    def test_designspace_not_loading_returns_extenders(self):
        module = Module(loading=False)
        designs = AT.designSpace(module)
        assert len(designs) == len(extenders)
        for design in designs:
            self.assertTrue(design.substrate in extenders)
    
    def test_operation_with_not_found_substrate(self):
        # Create an instance of the AT class with a substrate that doesn't exist in starters or extenders
        at_instance = AT(active=True, substrate="notFoundSubstrate")

        # Test the operation method for loading scenario with non-existent substrate
        # Expecting a ValueError due to the substrate not being found
        with self.assertRaises(ValueError) as context:
            at_instance.operation(chain=None, loading=True)
        
        # Optionally, you can check the message of the exception
        self.assertTrue("not found in starters" in str(context.exception))

        # Repeat the test for the non-loading (extender) scenario
        with self.assertRaises(ValueError) as context_extender:
            at_instance.operation(chain=None, loading=False)
        
        # Check the message for the extender scenario
        self.assertTrue("not found in extenders" in str(context_extender.exception))

class TestKR(unittest.TestCase):
    def setUp(self):
        # Setup for testing KR domain
        return
    
    def test_init_invalid_type(self):
        active = True
        type = 'D'

        with pytest.raises(AssertionError):
            kr = KR(active, type)
    
    def test_init_success(self):
        active = True
        type = 'B1'

        kr = KR(active, type)

        assert kr.active == active
        assert kr.type == type
    
    def test_designSpace_when_module_does_not_contain_MalonylCoA(self):
        # Arrange
        module1_domains = {
            AT: AT(active=True, substrate='Acetyl-CoA')
        }
        module = Module(loading=True, domains=module1_domains)

        # Act
        result = KR.designSpace(module)

        # Assert
        assert isinstance(result, list)
        assert all(isinstance(obj, KR) for obj in result)

        self.assertEqual(len(result), 7)
        self.assertTrue(any(design.type == 'C1' and design.active for design in result))
        self.assertTrue(any(design.type == 'B1' and design.active for design in result))
        self.assertTrue(any(design.type == 'B1' and not design.active for design in result))
    
    def test_designSpace_when_module_contains_MalonylCoA(self):
        # Arrange
        module1_domains = {
            AT: AT(active=True, substrate='Malonyl-CoA')
        }
        module = Module(loading=True, domains=module1_domains)

        # Act
        result = KR.designSpace(module)

        # Assert
        assert isinstance(result, list)
        assert all(isinstance(obj, KR) for obj in result)
        expected_designs = ['A', 'B', 'B1']

        self.assertEqual(len(result), 3)
        self.assertTrue(any(design.type == 'B' and design.active for design in result))
        self.assertTrue(any(design.type == 'B1' and not design.active for design in result))

    @parameterized.expand([
        ('B1', '[#0:1][C:2](=[O:3])[C:4][C:5](=[O:6])[S:7]>>[#0:1][C@@:2]([O:3])[C@:4][C:5](=[O:6])[S:7]'),
        ('B', '[#0:1][C:2](=[O:3])[C:4][C:5](=[O:6])[S:7]>>[#0:1][C@@:2]([O:3])[C:4][C:5](=[O:6])[S:7]'),
        ('C1', '[#0:1][C:2](=[O:3])[C:4][C:5](=[O:6])[S:7]>>[#0:1][C:2](=[O:3])[C@:4][C:5](=[O:6])[S:7]')
    ])
    def test_operation(self, kr_type, expected_smarts):
        with patch('rdkit.Chem.MolToSmiles') as mock_mol_to_smiles, \
            patch('rdkit.Chem.AllChem.ReactionFromSmarts') as mock_reaction_from_smarts, \
            patch('rdkit.Chem.SanitizeMol') as mock_sanitize:
            # Set up the mock for ReactionFromSmarts to return a mock object
            mock_mol_to_smiles.return_value = 'mocked SMILES string'
            reaction_mock = MagicMock()
            mock_reaction_from_smarts.return_value = reaction_mock
            
            # Configure the mock object to have a RunReactants method
            # which also needs to return a mock to simulate the reaction products
            reaction_product_mock = MagicMock()
            reaction_mock.RunReactants.return_value = ((reaction_product_mock,),)
            
            # Instantiate KR and call the operation method with a mock chain
            chain_mock = MagicMock()
            chain_mock.GetSubstructMatches.return_value = ((1,),)
            kr_instance = KR(True, kr_type)
            kr_instance.operation(chain_mock)
            
            # Assertions to verify that the methods were called with expected parameters
            mock_reaction_from_smarts.assert_called_once_with(expected_smarts)
            reaction_mock.RunReactants.assert_called_once_with((chain_mock,))
            mock_sanitize.assert_called_once_with(reaction_product_mock)

    def test_kr_operation_requires_correct_substructure(self):
        kr = KR(True, 'B1')
        chain = Chem.MolFromSmiles('C(=O)')
        with pytest.raises(AssertionError):
            kr.operation(chain)

    def test_reactants_returns_list_of_metabolites(self):
        kr = KR(active=True, type='B1')
        reactants = kr.reactants()
        assert len(reactants) == 2
        assert all(isinstance(obj, Metabolite) for obj in reactants)
        assert reactants[0].id == 'nadph_c'
        assert reactants[1].id == 'h_c'
    
    def test_products(self):
        kr = KR(active=True, type='B1')
        reactants = kr.products()
        assert len(reactants) == 1
        assert all(isinstance(obj, Metabolite) for obj in reactants)
        assert reactants[0].id == 'nadp_c'

class TestDH(unittest.TestCase):
    def test_init_invalid_type(self):
        active = True
        type = 'D'

        with pytest.raises(AssertionError):
            dh = DH(active, type)
    
    def test_init_success(self):
        active = True
        type = 'E'
        dh = DH(active, type)

        assert dh.active == active
        assert dh.type == type
        
    def test_dh_design_space_without_module(self):
        # Act
        result = DH.designSpace()
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(obj, DH) for obj in result)
        assert any(obj.active for obj in result)
        assert any(not obj.active for obj in result)
    
    def test_dh_design_space_with_module_lacking_kr(self):
        # Arrange
        module = Module(loading=True, domains={AT: AT(active=True, substrate='Acetyl-CoA')})

        # Act
        result = DH.designSpace(module=module)

        # Assert
        assert len(result) == 1
        assert all(not obj.active for obj in result)

    def test_dh_design_space_with_inactive_kr(self):
        # Arrange
        module = Module(loading=True, domains={KR: KR(active=False, type='B')})

        # Act
        result = DH.designSpace(module=module)

        # Assert
        assert len(result) == 1
        assert all(not obj.active for obj in result)

    @parameterized.expand(['A', 'B', 'B1'])
    def test_dh_design_space_with_active_kr_of_type_a_b_b1(self, kr_type):
        # Arrange
        module = Module(loading=True, domains={KR: KR(active=True, type=kr_type)})

        # Act
        result = DH.designSpace(module=module)
    
        # Assert
        assert len(result) == 2
        assert any(obj.active for obj in result)
        assert any(not obj.active for obj in result)
    
    @parameterized.expand([
        ('Z','[#0:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>[#0:1]/[CH1:2]=[CH1:4]\[C:6](=[O:7])[S:8].[O:3]'),
        ('E','[#0:1][C:2]([O:3])[C:4][C:6](=[O:7])[S:8]>>[#0:1]/[CH1:2]=[CH1:4]/[C:6](=[O:7])[S:8].[O:3]')
    ])
    def test_dh_operation_success(self, dh_type, expected_smarts):
        with patch('rdkit.Chem.MolToSmiles') as mock_mol_to_smiles, \
            patch('rdkit.Chem.AllChem.ReactionFromSmarts') as mock_reaction_from_smarts, \
            patch('rdkit.Chem.SanitizeMol') as mock_sanitize:

            # Setup mocks
            mock_mol_to_smiles.return_value = 'mocked SMILES string'
            reaction_mock = MagicMock()
            mock_reaction_from_smarts.return_value = reaction_mock

            reaction_product_mock = MagicMock()
            reaction_mock.RunReactants.return_value = ((reaction_product_mock,),)

            chain_mock = MagicMock()
            chain_mock.GetSubstructMatches.return_value = ((1,),)  # Matches the assert condition

            # Instantiate DH and call the operation method
            dh_instance = DH(True, dh_type)
            result = dh_instance.operation(chain_mock)

            # Verify the behavior and interactions
            mock_reaction_from_smarts.assert_called_once_with(expected_smarts)  # Since specific SMARTS pattern checking might be too rigid
            reaction_mock.RunReactants.assert_called_once_with((chain_mock,))
            mock_sanitize.assert_called_once_with(reaction_product_mock)
            assert result is reaction_product_mock  # Check if the product is as expected

    @parameterized.expand(['Z', 'E'])
    def test_dh_operation_retry_after_value_error(self, dh_type):
        with patch('rdkit.Chem.MolToSmiles') as mock_mol_to_smiles, \
            patch('rdkit.Chem.AllChem.ReactionFromSmarts') as mock_reaction_from_smarts, \
            patch('rdkit.Chem.SanitizeMol') as mock_sanitize:

            # Setup mocks for initial failure
            mock_mol_to_smiles.side_effect = lambda x: 'mocked SMILES string'
            reaction_mock = MagicMock()
            mock_reaction_from_smarts.return_value = reaction_mock

            reaction_product_mock_first_try = MagicMock()
            reaction_product_mock_second_try = MagicMock()
            reaction_mock.RunReactants.side_effect = [((reaction_product_mock_first_try,),), ((reaction_product_mock_second_try,),)]

            # Simulate ValueError on first try
            mock_sanitize.side_effect = [ValueError, None]

            chain_mock = MagicMock()
            chain_mock.GetSubstructMatches.return_value = ((1,),)  # Matches the assert condition twice

            # Instantiate DH and call the operation method
            dh_instance = DH(True, dh_type)
            result = dh_instance.operation(chain_mock)

            # Verify the behavior and interactions
            assert mock_reaction_from_smarts.call_count == 2  # Called twice with different reactions
            assert mock_sanitize.call_count == 2  # Called twice, second after catching ValueError
            assert result is reaction_product_mock_second_try  # The result should be the second product

    @parameterized.expand(['Z', 'E'])
    def test_dh_reactants(self, dh_type):
        dh = DH(True, dh_type)
        reactants = dh.reactants()
        # Verify the reactants list is empty
        assert isinstance(reactants, list), "The reactants should be a list."
        assert len(reactants) == 0, "The reactants list should be empty."

    @parameterized.expand(['Z', 'E'])
    def test_dh_products(self, dh_type):
        dh = DH(True, dh_type)
        products = dh.products()
        # Verify the products list contains the correct metabolites
        assert isinstance(products, list), "The products should be a list."
        assert len(products) == 1, "The products list should contain exactly one metabolite."
        h2o = products[0]
        assert isinstance(h2o, cobra.Metabolite), "The product should be an instance of cobra.Metabolite."
        assert h2o.id == 'h2o_c', "The metabolite ID should be 'h2o_c'."
        assert h2o.compartment == 'c', "The metabolite compartment should be 'c'."

class TestER(unittest.TestCase):
    def test_init_invalid_type(self):
        active = True
        type = 'J'

        with pytest.raises(AssertionError):
            er = ER(active, type)
    
    def test_init_success(self):
        active = True
        type = 'L'
        er = ER(active, type)

        assert er.active == active
        assert er.type == type

    def test_design_space_without_module(self):
        """Test that designSpace returns both active and inactive ER when no module is provided."""
        result = ER.designSpace()
        self.assertEqual(len(result), 3)
        self.assertTrue(any(er.active for er in result))
        self.assertTrue(any(not er.active for er in result))

    def test_design_space_with_module_lacking_dh(self):
        """Test that designSpace returns only inactive ER when DH is not in module domains."""
        module = MagicMock()
        module.domains = {}
        result = ER.designSpace(module=module)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].active)

    def test_design_space_with_inactive_dh(self):
        """Test that designSpace returns only inactive ER when DH in module domains is inactive."""
        dh_instance = DH(active=False, type='E')
        module = MagicMock()
        module.domains = {DH: dh_instance}
        result = ER.designSpace(module=module)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].active)

    def test_design_space_with_active_dh(self):
        """Test that designSpace returns both active and inactive ER when DH in module domains is active."""
        dh_instance = DH(active=True, type='E')
        module = MagicMock()
        module.domains = {DH: dh_instance}
        result = ER.designSpace(module=module)
        self.assertEqual(len(result), 3)
        self.assertTrue(any(er.active for er in result))
        self.assertTrue(any(not er.active for er in result))

    @parameterized.expand(['L', 'D'])
    def test_er_operation_success(self, er_type):
        """Test ER operation success without handling ValueError."""
        with patch('rdkit.Chem.AllChem.ReactionFromSmarts') as mock_reaction_from_smarts, \
            patch('rdkit.Chem.MolToSmiles') as mock_mol_to_smiles, \
            patch('rdkit.Chem.SanitizeMol') as mock_sanitize:

            # Setup mocks
            mock_mol_to_smiles.return_value = 'mocked SMILES string'
            reaction_mock = MagicMock()
            mock_reaction_from_smarts.return_value = reaction_mock
            reaction_product_mock = MagicMock()
            reaction_mock.RunReactants.return_value = ((reaction_product_mock,),)

            chain_mock = MagicMock()
            chain_mock.GetSubstructMatches.return_value = ((1,),)

            # Instantiate ER and call the operation method
            er_instance = ER(True, er_type)
            result = er_instance.operation(chain_mock)

            # Assertions
            mock_reaction_from_smarts.assert_called_once()
            reaction_mock.RunReactants.assert_called_once_with((chain_mock,))
            mock_sanitize.assert_called_once_with(reaction_product_mock)
            self.assertEqual(result, reaction_product_mock)

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.MolToSmiles')
    @patch('rdkit.Chem.SanitizeMol')
    def test_er_operation_assertion_failure(self, mock_sanitize, mock_mol_to_smiles, mock_reaction_from_smarts):
        """Test that the ER operation raises an AssertionError when the substructure match condition fails."""
        mock_mol_to_smiles.return_value = 'mocked SMILES string'
        reaction_mock = MagicMock()
        mock_reaction_from_smarts.return_value = reaction_mock

        chain_mock = MagicMock()
        chain_mock.GetSubstructMatches.return_value = ()  # No matches, should fail the assertion

        er_instance = ER(True, 'L')

        # Expecting an AssertionError due to the assertion failing
        with self.assertRaises(AssertionError) as context:
            er_instance.operation(chain_mock)

        # Check that the assertion error message contains the expected SMILES string
        self.assertIn('mocked SMILES string', str(context.exception))

    @parameterized.expand(['L', 'D'])
    def test_er_operation_handle_value_error(self, er_type):
        """Test ER operation retries the reaction after ValueError."""
        with patch('rdkit.Chem.AllChem.ReactionFromSmarts') as mock_reaction_from_smarts, \
             patch('rdkit.Chem.MolToSmiles') as mock_mol_to_smiles, \
             patch('rdkit.Chem.SanitizeMol', side_effect=[ValueError, None]) as mock_sanitize:

            # Setup mocks
            mock_mol_to_smiles.return_value = 'mocked SMILES string'
            reaction_mock = MagicMock()
            mock_reaction_from_smarts.return_value = reaction_mock
            reaction_product_mock_first_try = MagicMock()
            reaction_product_mock_second_try = MagicMock()
            reaction_mock.RunReactants.side_effect = [((reaction_product_mock_first_try,),), ((reaction_product_mock_second_try,),)]

            chain_mock = MagicMock()
            chain_mock.GetSubstructMatches.return_value = ((1,),)

            # Instantiate ER and call the operation method
            er_instance = ER(True, er_type)
            result = er_instance.operation(chain_mock)

            # Assertions
            self.assertEqual(mock_reaction_from_smarts.call_count, 2)
            self.assertEqual(reaction_mock.RunReactants.call_count, 2)
            self.assertEqual(mock_sanitize.call_count, 1)
            self.assertEqual(result, reaction_product_mock_second_try)

    @parameterized.expand(['L', 'D'])
    def test_er_reactants(self, er_type):
        """Test that ER reactants are correctly returned."""
        er = ER(True, er_type)
        reactants = er.reactants()
        self.assertEqual(len(reactants), 2, "There should be two reactants.")
        # Check that each metabolite is an instance of cobra.Metabolite and has the correct ID and compartment
        expected_ids = {'nadph_c', 'h_c'}
        for metabolite in reactants:
            self.assertIsInstance(metabolite, cobra.Metabolite)
            self.assertIn(metabolite.id, expected_ids)
            self.assertEqual(metabolite.compartment, 'c')
            expected_ids.remove(metabolite.id)  # Remove to ensure both are unique and correct

    @parameterized.expand(['L', 'D'])
    def test_er_products(self, er_type):
        """Test that ER products are correctly returned."""
        er = ER(True, er_type)
        products = er.products()
        self.assertEqual(len(products), 1, "There should be one product.")
        product = products[0]
        self.assertIsInstance(product, cobra.Metabolite)
        self.assertEqual(product.id, 'nadp_c')
        self.assertEqual(product.compartment, 'c')

class TestTE(unittest.TestCase):
    def setUp(self):
        # Setup for testing TE domain
        self.chain = Chem.MolFromSmiles('CC(=O)SC')
        self.te_active_cyclic = TE(active=True, cyclic=True, ring=3)
        self.te_active_linear = TE(active=True, cyclic=False, ring=0)

    def test_init(self):
        # Test initialization
        self.assertTrue(self.te_active_cyclic.active)
        self.assertTrue(self.te_active_cyclic.cyclic)
        self.assertEqual(self.te_active_cyclic.ring, 3)

    def test_design_space(self):
        """Test the designSpace class method."""
        result = TE.designSpace()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].active)
        self.assertFalse(result[0].cyclic)
        self.assertEqual(result[0].ring, 0)

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.SanitizeMol')
    def test_operation_cyclic(self, mock_sanitize, mock_reaction_from_smarts):
        # Setup the mock for ReactionFromSmarts to return a mock object
        reaction_mock = MagicMock()
        mock_reaction_from_smarts.return_value = reaction_mock
        te_cyclic = TE(active=True, cyclic=True, ring=1)
        
        # Configure the mock object to have a RunReactants method
        # which also needs to return a mock to simulate the reaction products
        reaction_product_mock1 = MagicMock()
        reaction_product_mock2 = MagicMock()
        reaction_mock.RunReactants.return_value = ((reaction_product_mock1,),(reaction_product_mock2,))
        
        # Execute the operation method
        product = te_cyclic.operation(self.chain)
        
        # Assertions to verify that the methods were called with expected parameters
        mock_reaction_from_smarts.assert_called_once_with(TE.cyclic_reaction)
        reaction_mock.RunReactants.assert_called_once_with((self.chain,))
        mock_sanitize.assert_called_once_with(reaction_product_mock1)
        self.assertEqual(product, reaction_product_mock1)

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.SanitizeMol')
    def test_operation_linear(self, mock_sanitize, mock_reaction_from_smarts):
        # Setup the mock for ReactionFromSmarts similar to the cyclic test
        reaction_mock = MagicMock()
        mock_reaction_from_smarts.return_value = reaction_mock
        
        # Configure the mock object to have a RunReactants method
        reaction_product_mock = MagicMock()
        reaction_mock.RunReactants.return_value = ((reaction_product_mock,),)
        
        # Execute the operation method
        product = self.te_active_linear.operation(self.chain)
        
        # Assertions to verify that the methods were called with expected parameters
        mock_reaction_from_smarts.assert_called_once_with(TE.linear_reaction)
        reaction_mock.RunReactants.assert_called_once_with((self.chain,))
        mock_sanitize.assert_called_once_with(reaction_product_mock)
        self.assertEqual(product, reaction_product_mock)

    def test_te_operation_requires_correct_substructure(self):
        kr = KR(True, 'B1')
        chain = Chem.MolFromSmiles('C(=O)')
        with pytest.raises(AssertionError):
            kr.operation(chain)

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.MolToSmiles')
    @patch('rdkit.Chem.SanitizeMol')
    def test_operation_assertion_error(self, mock_sanitize, mock_mol_to_smiles, mock_reaction_from_smarts):
        """Test that an AssertionError is raised when the substructure match condition fails."""
        # Setting up the mock for MolToSmiles to return a specific SMILES string
        mock_mol_to_smiles.return_value = 'mocked SMILES string of chain'
        reaction_mock = MagicMock()
        mock_reaction_from_smarts.return_value = reaction_mock
        
        # Setting up the chain mock to simulate zero matching substructures
        chain_mock = MagicMock()
        chain_mock.GetSubstructMatches.return_value = ()  # No matches found

        te = TE(active=True, cyclic=False, ring=0)

        # Expecting an AssertionError due to failing the substructure match condition
        with self.assertRaises(AssertionError) as context:
            te.operation(chain_mock)

        # Check that the assertion error message contains the expected SMILES string
        self.assertIn('mocked SMILES string of chain', str(context.exception))
        self.assertEqual(str(context.exception), "mocked SMILES string of chain")

        # Verify that no further reactions were attempted after the assertion failed
        mock_reaction_from_smarts.assert_not_called()
        mock_sanitize.assert_not_called()

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.MolToSmiles')
    @patch('rdkit.Chem.SanitizeMol')
    def test_operation(self, mock_sanitize, mock_mol_to_smiles, mock_reaction_from_smarts):
        """Test the operation method."""
        mock_mol_to_smiles.return_value = 'mocked SMILES string'
        reaction_mock = MagicMock()
        mock_reaction_from_smarts.return_value = reaction_mock
        reaction_product_mock = MagicMock()
        reaction_mock.RunReactants.return_value = ((reaction_product_mock,),)

        chain_mock = MagicMock()
        chain_mock.GetSubstructMatches.return_value = ((1,),)  # Simulate one matching substructure

        te = TE(active=True, cyclic=False, ring=0)
        result = te.operation(chain_mock)

        mock_reaction_from_smarts.assert_called_once()
        reaction_mock.RunReactants.assert_called_once_with((chain_mock,))
        mock_sanitize.assert_called_once_with(reaction_product_mock)
        self.assertEqual(result, reaction_product_mock)

    def test_reactants(self):
        """Test the reactants method."""
        te = TE(active=True, cyclic=False, ring=0)
        reactants = te.reactants()
        self.assertEqual(len(reactants), 1)
        self.assertIsInstance(reactants[0], cobra.Metabolite)
        self.assertEqual(reactants[0].id, 'h2o_c')
        self.assertEqual(reactants[0].compartment, 'c')

    def test_products(self):
        """Test the products method."""
        te = TE(active=True, cyclic=False, ring=0)
        products = te.products()
        self.assertEqual(len(products), 1)
        self.assertIsInstance(products[0], cobra.Metabolite)
        self.assertEqual(products[0].id, 'h_c')
        self.assertEqual(products[0].compartment, 'c')

if __name__ == '__main__':
    unittest.main()
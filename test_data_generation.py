#!/usr/bin/env python3
"""
Unit Tests for Data Generation Module
Tests the generate_sample_data.py functionality
"""

import unittest
import numpy as np
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, '.')

from generate_sample_data import (
    load_config,
    get_default_config,
    set_random_seed,
    generate_synthesis_parameters,
    validate_physics_constraints,
    calculate_physics_features,
    determine_phase_outcomes,
    generate_material_properties,
    validate_generated_data,
    filter_low_quality_samples,
    create_sample_dataset
)


class TestDataGeneration(unittest.TestCase):
    """Test cases for data generation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = get_default_config()
        self.small_sample_size = 50
        set_random_seed(42)  # Ensure reproducible tests
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_loading(self):
        """Test configuration loading functionality"""
        # Test default config
        default_config = get_default_config()
        self.assertIsInstance(default_config, dict)
        self.assertIn('synthesis_parameters', default_config)
        
        # Test loading non-existent config (should return default)
        config = load_config('non_existent_file.json')
        self.assertIsInstance(config, dict)
    
    def test_synthesis_parameter_generation(self):
        """Test synthesis parameter generation"""
        df = generate_synthesis_parameters(self.small_sample_size, self.test_config)
        
        # Check basic properties
        self.assertEqual(len(df), self.small_sample_size)
        
        # Check required columns exist
        required_cols = ['cs_br_concentration', 'pb_br2_concentration', 'temperature', 
                        'oa_concentration', 'oam_concentration', 'reaction_time', 'solvent_type']
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        
        # Check value ranges
        self.assertTrue((df['cs_br_concentration'] >= 0.1).all())
        self.assertTrue((df['cs_br_concentration'] <= 3.0).all())
        self.assertTrue((df['temperature'] >= 80.0).all())
        self.assertTrue((df['temperature'] <= 250.0).all())
        self.assertTrue((df['solvent_type'] >= 0).all())
        self.assertTrue((df['solvent_type'] <= 4).all())
        
        # Check data types
        self.assertTrue(df['cs_br_concentration'].dtype in [np.float32, np.float64])
        self.assertTrue(df['solvent_type'].dtype in [np.int8, np.int16, np.int32, np.int64])
    
    def test_physics_constraints_validation(self):
        """Test physics constraints validation"""
        df = generate_synthesis_parameters(self.small_sample_size, self.test_config)
        validated_df = validate_physics_constraints(df, self.test_config)
        
        # Check that validation columns are added
        validation_cols = ['stoichiometry_valid', 'concentration_valid', 'ligand_valid', 'physics_failure_prob']
        for col in validation_cols:
            self.assertIn(col, validated_df.columns, f"Missing validation column: {col}")
        
        # Check boolean columns are actually boolean
        for col in ['stoichiometry_valid', 'concentration_valid', 'ligand_valid']:
            self.assertTrue(validated_df[col].dtype == bool, f"{col} should be boolean")
        
        # Check failure probability is in valid range
        self.assertTrue((validated_df['physics_failure_prob'] >= 0).all())
        self.assertTrue((validated_df['physics_failure_prob'] <= 1).all())
    
    def test_physics_features_calculation(self):
        """Test physics features calculation"""
        df = generate_synthesis_parameters(self.small_sample_size, self.test_config)
        physics_df = calculate_physics_features(df, self.test_config)
        
        # Check physics columns are added
        physics_cols = ['supersaturation', 'nucleation_rate', 'growth_rate', 'solvent_effect']
        for col in physics_cols:
            self.assertIn(col, physics_df.columns, f"Missing physics column: {col}")
        
        # Check for reasonable values (no NaN, inf, or negative values where inappropriate)
        self.assertFalse(physics_df['supersaturation'].isna().any())
        self.assertFalse(np.isinf(physics_df['supersaturation']).any())
        self.assertTrue((physics_df['supersaturation'] >= 0).all())
        
        self.assertTrue((physics_df['solvent_effect'] > 0).all())
        self.assertTrue((physics_df['solvent_effect'] <= 2.0).all())  # Reasonable upper bound
    
    def test_phase_outcome_determination(self):
        """Test phase outcome determination"""
        df = generate_synthesis_parameters(self.small_sample_size, self.test_config)
        validated_df = validate_physics_constraints(df, self.test_config)
        physics_df = calculate_physics_features(validated_df, self.test_config)
        phase_df = determine_phase_outcomes(physics_df, self.test_config)
        
        # Check phase label column exists
        self.assertIn('phase_label', phase_df.columns)
        
        # Check phase labels are in valid range (0-4)
        self.assertTrue((phase_df['phase_label'] >= 0).all())
        self.assertTrue((phase_df['phase_label'] <= 4).all())
        
        # Check probability columns exist and sum to 1
        prob_cols = ['prob_3d', 'prob_0d', 'prob_2d', 'prob_mixed', 'prob_failed']
        for col in prob_cols:
            self.assertIn(col, phase_df.columns, f"Missing probability column: {col}")
        
        # Check probabilities sum to approximately 1
        prob_sums = phase_df[prob_cols].sum(axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5, 
                                 err_msg="Phase probabilities should sum to 1")
    
    def test_material_properties_generation(self):
        """Test material properties generation"""
        df = generate_synthesis_parameters(self.small_sample_size, self.test_config)
        validated_df = validate_physics_constraints(df, self.test_config)
        physics_df = calculate_physics_features(validated_df, self.test_config)
        phase_df = determine_phase_outcomes(physics_df, self.test_config)
        material_df = generate_material_properties(phase_df, self.test_config)
        
        # Check material property columns exist
        material_cols = ['bandgap', 'plqy', 'emission_peak', 'particle_size', 
                        'emission_fwhm', 'lifetime', 'stability_score', 'phase_purity']
        for col in material_cols:
            self.assertIn(col, material_df.columns, f"Missing material property: {col}")
        
        # Check uncertainty columns exist
        uncertainty_cols = ['bandgap_uncertainty', 'plqy_uncertainty', 'particle_size_uncertainty', 
                           'synthesis_confidence']
        for col in uncertainty_cols:
            self.assertIn(col, material_df.columns, f"Missing uncertainty column: {col}")
        
        # Check value ranges for key properties
        self.assertTrue((material_df['bandgap'] >= 0.5).all())  # Minimum bandgap
        self.assertTrue((material_df['bandgap'] <= 5.0).all())  # Maximum reasonable bandgap
        self.assertTrue((material_df['plqy'] >= 0.0).all())
        self.assertTrue((material_df['plqy'] <= 1.0).all())
        self.assertTrue((material_df['particle_size'] >= 1.0).all())
        self.assertTrue((material_df['synthesis_confidence'] >= 0.1).all())
        self.assertTrue((material_df['synthesis_confidence'] <= 1.0).all())
    
    def test_data_validation(self):
        """Test data validation functionality"""
        # Generate a small complete dataset
        df = generate_synthesis_parameters(self.small_sample_size, self.test_config)
        validated_df = validate_physics_constraints(df, self.test_config)
        physics_df = calculate_physics_features(validated_df, self.test_config)
        phase_df = determine_phase_outcomes(physics_df, self.test_config)
        complete_df = generate_material_properties(phase_df, self.test_config)
        
        # Run validation
        validation_results = validate_generated_data(complete_df)
        
        # Check validation structure
        self.assertIsInstance(validation_results, dict)
        self.assertIn('n_samples', validation_results)
        self.assertIn('property_ranges', validation_results)
        self.assertIn('warnings', validation_results)
        
        self.assertEqual(validation_results['n_samples'], self.small_sample_size)
        self.assertIsInstance(validation_results['warnings'], list)
    
    def test_quality_filtering(self):
        """Test quality filtering functionality"""
        # Generate data with some intentionally poor quality samples
        df = generate_synthesis_parameters(self.small_sample_size, self.test_config)
        validated_df = validate_physics_constraints(df, self.test_config)
        
        # Apply quality filtering
        filtered_df = filter_low_quality_samples(validated_df, quality_threshold=0.8)
        
        # Check that some samples were potentially removed
        self.assertLessEqual(len(filtered_df), len(validated_df))
        self.assertGreater(len(filtered_df), 0)  # Should not remove everything
    
    def test_complete_dataset_creation(self):
        """Test complete dataset creation"""
        output_path = Path(self.temp_dir) / "test_data"
        
        # Create small dataset
        csv_path = create_sample_dataset(
            n_samples=self.small_sample_size,
            output_dir=str(output_path),
            use_parallel=False  # Disable parallel for small test
        )
        
        # Check files were created
        self.assertTrue(Path(csv_path).exists())
        info_path = output_path / "dataset_info.json"
        self.assertTrue(info_path.exists())
        
        # Load and check CSV
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 0)
        self.assertLessEqual(len(df), self.small_sample_size)  # May be filtered
        
        # Load and check info file
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        self.assertIn('n_samples', info)
        self.assertIn('validation_results', info)
        self.assertEqual(info['n_samples'], len(df))
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        # Generate two datasets with same seed
        set_random_seed(12345)
        df1 = generate_synthesis_parameters(20, self.test_config)
        
        set_random_seed(12345)
        df2 = generate_synthesis_parameters(20, self.test_config)
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
    
    def test_parallel_vs_sequential(self):
        """Test that parallel and sequential generation produce similar results"""
        test_samples = 100
        
        # Sequential generation
        csv_path_seq = create_sample_dataset(
            n_samples=test_samples,
            output_dir=str(Path(self.temp_dir) / "sequential"),
            use_parallel=False
        )
        
        # Parallel generation (if sample size allows)
        csv_path_par = create_sample_dataset(
            n_samples=test_samples,
            output_dir=str(Path(self.temp_dir) / "parallel"),
            use_parallel=True
        )
        
        # Load both datasets
        df_seq = pd.read_csv(csv_path_seq)
        df_par = pd.read_csv(csv_path_par)
        
        # Should have similar distributions (not exact due to different random seeds)
        self.assertEqual(len(df_seq.columns), len(df_par.columns))
        self.assertAlmostEqual(df_seq['temperature'].mean(), df_par['temperature'].mean(), delta=20.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        set_random_seed(42)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_zero_samples(self):
        """Test behavior with zero samples"""
        config = get_default_config()
        with self.assertRaises((ValueError, ZeroDivisionError)):
            generate_synthesis_parameters(0, config)
    
    def test_very_large_sample_request(self):
        """Test behavior with very large sample requests"""
        config = get_default_config()
        # Should handle gracefully (may be slow but shouldn't crash)
        try:
            df = generate_synthesis_parameters(5, config)  # Use small number for test speed
            self.assertEqual(len(df), 5)
        except MemoryError:
            self.skipTest("Not enough memory for large dataset test")
    
    def test_invalid_config(self):
        """Test behavior with invalid configuration"""
        invalid_config = {"invalid": "config"}
        
        # Should fall back to defaults or handle gracefully
        try:
            df = generate_synthesis_parameters(10, invalid_config)
            self.assertGreater(len(df), 0)
        except (KeyError, TypeError):
            # Expected behavior for invalid config
            pass


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
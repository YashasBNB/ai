import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging

from config import Config, get_default_config
from scripts.data_processor import DataProcessor
from scripts.feature_extractor import AdvancedFeatureExtractor
from scripts.advanced_pattern_learner import PatternLearner
from scripts.pattern_analyzer import PatternAnalyzer
from utils.synthetic_data import generate_test_data

class TestUnsupervisedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create temporary directories
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = Path(cls.temp_dir) / "data"
        cls.model_dir = Path(cls.temp_dir) / "models"
        cls.results_dir = Path(cls.temp_dir) / "results"
        
        for directory in [cls.data_dir, cls.model_dir, cls.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Generate synthetic test data
        cls.test_data = generate_test_data(
            output_dir=str(cls.data_dir),
            num_symbols=3,
            timeframe='M15',
            num_days=5
        )
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.temp_dir)
        
    def setUp(self):
        """Set up test configuration"""
        self.config = get_default_config()
        self.config.data.data_dir = str(self.data_dir)
        self.config.model_dir = str(self.model_dir)
        self.config.results_dir = str(self.results_dir)
        
        # Reduce model size and training time for tests
        self.config.model.hidden_dims = [32, 16]
        self.config.model.epochs = 2
        self.config.model.batch_size = 32
        
    def test_data_processor(self):
        """Test data processing functionality"""
        processor = DataProcessor(self.config.data.data_dir)
        
        # Test data loading
        data_dict = processor.load_data(timeframe='M15')
        self.assertGreater(len(data_dict), 0, "No data loaded")
        
        # Test data alignment
        aligned_data = processor.align_data(data_dict)
        self.assertEqual(
            len(set(df.index[0] for df in aligned_data.values())),
            1,
            "Data not properly aligned"
        )
        
    def test_feature_extraction(self):
        """Test feature extraction functionality"""
        # Load data
        processor = DataProcessor(self.config.data.data_dir)
        data_dict = processor.load_data(timeframe='M15')
        
        # Extract features
        extractor = AdvancedFeatureExtractor()
        symbol = list(data_dict.keys())[0]
        features, names = extractor.fit_transform(
            data_dict[symbol],
            window_size=self.config.data.window_size
        )
        
        self.assertIsInstance(features, np.ndarray, "Features not numpy array")
        self.assertGreater(len(names), 0, "No feature names generated")
        
    def test_pattern_learner(self):
        """Test pattern learning functionality"""
        # Generate sample features
        features = np.random.randn(100, 20)
        
        # Initialize and train model
        learner = PatternLearner(
            input_dim=features.shape[1],
            latent_dim=8,
            hidden_dims=[32, 16]
        )
        
        learner.train(
            data=features,
            epochs=2,
            batch_size=32
        )
        
        # Test pattern detection
        patterns = learner.detect_patterns(features)
        
        self.assertIn('clusters', patterns, "No clusters in patterns")
        self.assertIn('reconstruction_errors', patterns, "No reconstruction errors")
        
    def test_pattern_analyzer(self):
        """Test pattern analysis functionality"""
        # Generate sample data
        features = np.random.randn(100, 20)
        patterns = {
            'clusters': np.random.randint(0, 3, 100),
            'latent_pca': np.random.randn(100, 2),
            'reconstruction_errors': np.random.rand(100),
            'anomalies': np.random.choice([True, False], 100)
        }
        feature_names = [f"feature_{i}" for i in range(20)]
        timestamps = pd.date_range('2023-01-01', periods=100, freq='15T')
        
        # Initialize analyzer
        analyzer = PatternAnalyzer(self.config.results_dir)
        
        # Run analysis
        results = analyzer.analyze_patterns(
            features=features,
            patterns=patterns,
            feature_names=feature_names,
            timestamps=timestamps
        )
        
        self.assertIsInstance(results, dict, "Analysis results not dictionary")
        self.assertGreater(len(results), 0, "No analysis results generated")
        
    def test_complete_pipeline(self):
        """Test complete pipeline integration"""
        from main import UnsupervisedTradingPipeline
        
        # Initialize pipeline
        pipeline = UnsupervisedTradingPipeline(self.config)
        
        try:
            # Run pipeline
            results = pipeline.run()
            
            # Check results
            self.assertIsNotNone(results, "Pipeline returned no results")
            
        except Exception as e:
            self.fail(f"Pipeline failed with error: {str(e)}")
            
    def test_gpu_support(self):
        """Test GPU support if available"""
        if torch.cuda.is_available():
            # Initialize model on GPU
            features = torch.randn(100, 20).cuda()
            learner = PatternLearner(
                input_dim=20,
                latent_dim=8,
                hidden_dims=[32, 16]
            )
            
            # Verify model is on GPU
            self.assertTrue(
                next(learner.model.parameters()).is_cuda,
                "Model not on GPU"
            )
            
            # Test forward pass
            output = learner.model(features)
            self.assertTrue(
                output[0].is_cuda,
                "Model output not on GPU"
            )

if __name__ == '__main__':
    unittest.main()

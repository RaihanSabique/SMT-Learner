#!/usr/bin/env python3
"""
Quick test script to verify the baseline comparison and ablation study setup
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from STCRL.BaselineModels import create_baseline_models
        print("✓ BaselineModels imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import BaselineModels: {e}")
        return False
    
    try:
        from STCRL.EvaluationFramework import ComprehensiveComparison, EvaluationMetrics
        print("✓ EvaluationFramework imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import EvaluationFramework: {e}")
        return False
    
    try:
        from STCRL.STCRLDataset import STCRLModelFittingDataset
        print("✓ STCRLDataset imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import STCRLDataset: {e}")
        return False
    
    try:
        from STCRL.TransformerEncoder import STCRLTransformer
        print("✓ TransformerEncoder imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TransformerEncoder: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if baseline models can be created"""
    print("\nTesting model creation...")
    
    try:
        from STCRL.BaselineModels import create_baseline_models
        models = create_baseline_models(input_dim=3, hidden_dim=64, embedding_dim=32)
        print(f"✓ Created {len(models)} baseline models")
        
        for name, model in models.items():
            print(f"  - {name}: {model.__class__.__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create models: {e}")
        return False

def test_dummy_data():
    """Test dummy data creation and dataset"""
    print("\nTesting dummy data creation...")
    
    try:
        # Create dummy data
        np.random.seed(42)
        dummy_data = []
        
        for i in range(10):  # Small dataset for testing
            seq_len = 512
            x = np.cumsum(np.random.randn(seq_len) * 0.1)
            y = np.cumsum(np.random.randn(seq_len) * 0.1)
            t = np.linspace(0, 1, seq_len)
            
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            
            trajectory = np.column_stack([x, y, t])
            
            dummy_data.append({
                'normalized_trajectory': trajectory,
                'completion_time': np.random.exponential(2.0) + 0.5,
                'rmsd': np.random.gamma(2, 0.1),
                'is_success': np.random.choice([0, 1]),
                'task_type': np.random.choice([0, 1]),
                'participant_id': np.random.randint(1, 5)
            })
        
        df = pd.DataFrame(dummy_data)
        print(f"✓ Created dummy dataset with {len(df)} samples")
        
        # Test dataset creation
        from STCRL.STCRLDataset import STCRLModelFittingDataset
        dataset = STCRLModelFittingDataset(df)
        print(f"✓ Created dataset with {len(dataset)} items")
        
        # Test data loading
        traj, metadata = dataset[0]
        print(f"✓ Sample trajectory shape: {traj.shape}")
        print(f"✓ Metadata keys: {list(metadata.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create dummy data: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")
    
    try:
        from STCRL.EvaluationFramework import EvaluationMetrics
        evaluator = EvaluationMetrics()
        
        # Create dummy trajectories
        batch_size = 4
        seq_len = 512
        input_dim = 3
        
        original = torch.randn(batch_size, seq_len, input_dim)
        reconstructed = original + torch.randn_like(original) * 0.1
        
        # Test reconstruction metrics
        mse = evaluator.trajectory_reconstruction_quality(original, reconstructed)
        endpoint_err = evaluator.endpoint_error(original, reconstructed)
        curvature_err = evaluator.curvature_error(original, reconstructed)
        
        print(f"✓ Reconstruction MSE: {mse:.6f}")
        print(f"✓ Endpoint error: {endpoint_err:.6f}")
        print(f"✓ Curvature error: {curvature_err:.6f}")
        
        # Test temporal correlation
        embeddings = np.random.randn(10, 64)
        metadata = {
            'completion_time': np.random.exponential(2.0, 10),
            'rmsd': np.random.gamma(2, 0.1, 10),
            'is_success': np.random.choice([0, 1], 10)
        }
        
        correlations = evaluator.temporal_correlation(embeddings, metadata)
        print(f"✓ Temporal correlations: {correlations}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test evaluation metrics: {e}")
        return False

def test_device_setup():
    """Test device setup"""
    print("\nTesting device setup...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("ℹ CUDA not available, using CPU")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("STCRL EVALUATION FRAMEWORK TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Dummy Data", test_dummy_data),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Device Setup", test_device_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The evaluation framework is ready to use.")
        print("\nNext steps:")
        print("1. Run: python GenerateRebuttalResults.py --quick")
        print("2. Check results in ./rebuttal_results/")
        print("3. Use full mode for actual rebuttal: python GenerateRebuttalResults.py")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install torch pandas matplotlib seaborn scikit-learn")
        print("2. Check Python path and module imports")
        print("3. Verify STCRL module structure")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

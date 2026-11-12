# -*- coding: utf-8 -*-
"""
Test script to verify the modular SAMBA system works correctly
"""

import torch
import numpy as np
from config import ModelArgs, TrainingConfig
from models import SAMBA
from utils import init_seed, print_model_parameters


def test_model_creation():
    """Test that the model can be created and initialized"""
    print("Testing model creation...")
    
    # Create model arguments
    model_args = ModelArgs(
        d_model=32,
        n_layer=2,
        vocab_size=10,
        seq_in=5,
        seq_out=1
    )
    
    # Create SAMBA model
    model = SAMBA(
        model_args,
        hidden=32,
        inp=5,
        out=1,
        embed=10,
        cheb_k=3
    )
    
    print("✓ Model created successfully")
    return model


def test_forward_pass():
    """Test that the model can perform forward pass"""
    print("Testing forward pass...")
    
    # Create model
    model = test_model_creation()
    
    # Create dummy input
    batch_size = 2
    seq_len = 5
    num_nodes = 10
    
    input_tensor = torch.randn(batch_size, seq_len, num_nodes)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"✓ Forward pass successful. Input shape: {input_tensor.shape}, Output shape: {output.shape}")
    return output


def test_config():
    """Test configuration classes"""
    print("Testing configuration...")
    
    # Test ModelArgs
    model_args = ModelArgs(
        d_model=64,
        n_layer=3,
        vocab_size=20,
        seq_in=10,
        seq_out=2
    )
    
    print(f"✓ ModelArgs created: d_inner={model_args.d_inner}, dt_rank={model_args.dt_rank}")
    
    # Test TrainingConfig
    config = TrainingConfig(
        lag=5,
        horizon=1,
        num_nodes=20,
        epochs=100
    )
    
    config_dict = config.to_dict()
    print(f"✓ TrainingConfig created with {len(config_dict)} parameters")
    
    return model_args, config


def test_utilities():
    """Test utility functions"""
    print("Testing utilities...")
    
    # Test seed initialization
    init_seed(42)
    print("✓ Seed initialization successful")
    
    # Test model parameter printing
    model = test_model_creation()
    print_model_parameters(model, only_num=True)
    print("✓ Model parameter printing successful")


def main():
    """Run all tests"""
    print("=" * 50)
    print("SAMBA Stock Price Forecasting System Test")
    print("=" * 50)
    
    try:
        # Test configuration
        test_config()
        print()
        
        # Test utilities
        test_utilities()
        print()
        
        # Test forward pass
        test_forward_pass()
        print()
        
        print("=" * 50)
        print("✓ All tests passed! SAMBA stock forecasting system is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

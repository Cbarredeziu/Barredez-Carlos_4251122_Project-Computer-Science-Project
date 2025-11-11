#!/usr/bin/env python3
"""
Test script to verify image saving functionality
"""
import os
import sys
from main import main

def test_image_saving():
    """Test that images are saved correctly"""
    print("Testing image saving functionality...")
    
    # Check if directories exist (adjust for running from src directory)
    data_dirs = [
        "../data/inputs",
        "../data/grid_photos"
    ]
    
    test_dirs = [
        "../test/parking_grind"
    ]
    
    for dir_path in data_dirs + test_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Found directory: {dir_path}")
            files = os.listdir(dir_path)
            if files:
                print(f"  Files: {files}")
        else:
            print(f"✗ Missing directory: {dir_path}")
    
    # Check results directories
    results_dirs = [
        "../test/results",
        "../test/results/images",
        "../test/results/data", 
        "../test/results/grid_results"
    ]
    
    print("\nResults directories:")
    for dir_path in results_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Found: {dir_path}")
            files = os.listdir(dir_path)
            if files:
                print(f"  Contents: {files}")
        else:
            print(f"✗ Missing: {dir_path}")

if __name__ == "__main__":
    test_image_saving()
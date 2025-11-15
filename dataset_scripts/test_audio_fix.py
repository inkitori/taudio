#!/usr/bin/env python3
"""
Minimal test to verify audio shape handling is correct.
"""

import numpy as np
from datasets import Dataset, Audio, Features, Value

def test_audio_shapes():
    """Test different audio array shapes to identify the issue."""
    
    print("Testing audio shape handling...")
    print("-" * 40)
    
    # Test Case 1: Correct 1D mono audio
    print("\nTest 1: 1D mono audio (CORRECT)")
    sample_rate = 44100
    duration = 2  # 2 seconds
    mono_audio = np.random.randn(sample_rate * duration).astype(np.float32)
    
    example1 = {
        'audio': {
            'array': mono_audio,
            'sampling_rate': sample_rate
        },
        'label': 'mono'
    }
    
    try:
        features1 = Features({
            'audio': Audio(),
            'label': Value('string')
        })
        dataset1 = Dataset.from_list([example1], features=features1)
        print(f"✓ Success! Audio shape: {mono_audio.shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test Case 2: Wrong shape (2D with wrong axis order)
    print("\nTest 2: 2D stereo audio (may fail)")
    stereo_audio = np.random.randn(sample_rate * duration, 2).astype(np.float32)
    
    example2 = {
        'audio': {
            'array': stereo_audio,
            'sampling_rate': sample_rate
        },
        'label': 'stereo_wrong'
    }
    
    try:
        features2 = Features({
            'audio': Audio(),
            'label': Value('string')
        })
        dataset2 = Dataset.from_list([example2], features=features2)
        print(f"✓ Success! Audio shape: {stereo_audio.shape}")
    except Exception as e:
        print(f"✗ Failed (expected): {e}")
        print("  This is the error we're seeing!")
    
    # Test Case 3: Convert stereo to mono
    print("\nTest 3: Convert stereo to mono (SOLUTION)")
    stereo_audio = np.random.randn(sample_rate * duration, 2).astype(np.float32)
    mono_converted = np.mean(stereo_audio, axis=1).astype(np.float32)
    
    example3 = {
        'audio': {
            'array': mono_converted,
            'sampling_rate': sample_rate
        },
        'label': 'mono_converted'
    }
    
    try:
        features3 = Features({
            'audio': Audio(),
            'label': Value('string')
        })
        dataset3 = Dataset.from_list([example3], features=features3)
        print(f"✓ Success! Converted audio shape: {mono_converted.shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 40)
    print("Solution: Always ensure audio is 1D float32 array!")
    print("=" * 40)


if __name__ == "__main__":
    test_audio_shapes()
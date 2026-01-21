import torch
import sys

print("=" * 60)
print("CUDA 12.6 Verification Test")
print("=" * 60)

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version in PyTorch: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"\n✓ GPU is available!")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")

    # Test tensor operations on GPU
    print("\nTesting GPU tensor operations...")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ Successfully performed matrix multiplication on GPU")
        print(f"  Result tensor device: {z.device}")
        print(f"  Result tensor shape: {z.shape}")
    except Exception as e:
        print(f"✗ Error during GPU operation: {e}")
else:
    print("\n✗ WARNING: CUDA not available!")
    print("\nTroubleshooting steps:")
    print("1. Check if you installed CUDA-enabled PyTorch:")
    print(
        "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
    )
    print("2. Verify NVIDIA drivers with: nvidia-smi")
    print("3. Check if torch.__version__ shows '+cu126' (not '+cpu')")

print("\n" + "=" * 60)

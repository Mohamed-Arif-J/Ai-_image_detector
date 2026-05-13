import torch
import sys

print("===== PYTORCH GPU CHECK =====")
print("Python version:", sys.version.split()[0])
print("PyTorch version:", torch.__version__)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    try:
        gpu_name = torch.cuda.get_device_name(0)
        print("GPU detected:", gpu_name)
        print("CUDA runtime version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
    except Exception as e:
        print("Error while accessing GPU:", e)
else:
    print("No GPU detected by PyTorch!")

import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())  # Will be False - that's OK!
print("CPU available:", torch.device('cpu'))

# Create a simple tensor
x = torch.rand(3, 3)
print("\nSample tensor:")
print(x)

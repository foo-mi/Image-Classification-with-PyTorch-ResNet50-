import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request

print("Loading pre-trained ResNet50 model...")
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Using  local test image
print("Loading test image...")
# Image is already in the folder as test_cat.jpg

# Load and process image
img = Image.open("test_cat1.jpg")
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Run inference
print("Running inference...")
with torch.no_grad():  # Don't calculate gradients (faster)
    output = model(img_tensor)

# Get prediction
_, predicted_idx = torch.max(output, 1)
print(f"\nPredicted class index: {predicted_idx.item()}")

# Download ImageNet labels
print("\nDownloading class labels...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt"
)

# Load labels
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Prediction: {labels[predicted_idx.item()]}")
print("\n✅ Success! Your first AI inference is complete!")

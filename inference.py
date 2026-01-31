import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


# 1. Setup Device and Labels 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_list = ["empty", "person", "vehicle"]

# 2. Consistent transforms for images
pic_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# 3. Model Architecture
def get_shipWatcher():
    model = models.resnet18(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False

    # Change fc layer
    input_size = model.fc.in_features
    model.fc = nn.Linear(input_size,3)
    return model


# 4. The Prediction Engine
@torch.no_grad()
def predict(image_path, model_path):
    # Initialise and load model with saved state_dict
    shipWatcher = get_shipWatcher()
    shipWatcher.load_state_dict(torch.load(model_path))
    shipWatcher.to(device)
    shipWatcher.eval()

    # Load and process the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = pic_transform(img).unsqueeze(0).to(device)  # Converts to (B, C, H, W) format

    # Run inference
    output = shipWatcher(img_tensor)
        
    # Calculate probabilities (Confidence %)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, index = torch.max(probabilities, 0)

    return class_list[index], confidence.item()


# 5. Execution Block
if __name__ == "__main__":
    # Define paths of test_image, shipWatcher model
    shipWatcher_pth = "shipWatcher.pth" 
    test_image_pth = "/Users/kimjimin/Desktop/test_image.jpg"

    print("Processing image...")
    
    try:
        # Run prediction engine
        class_label, confidence = predict(test_image_pth, shipWatcher_pth)
        
        # Display results
        print(f"Target Identified: {class_label}")
        print(f"Confidence Score:  {confidence * 100:.2f}%")
        print(f"----------------------------------")
        
    except FileNotFoundError as f:
        print("Error: Could not find the model or test image.")
        print("Check file paths!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")












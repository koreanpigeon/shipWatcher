import torch
import torch.nn as nn


# Device & class labels
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_list = ["empty", "person", "vehicle"]


# shipWatcher model architecture
# Initialise ResNet-18 with pre-trained weights as a feature extractor
# Modify full connected(fc) layer with nn.Linear for 3-class classification
def get_shipWatcher():
    model = models.resnet18(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False

    # Change fc layer
    input_size = model.fc.in_features
    model.fc = nn.Linear(input_size,3)
    return model
  

# Transforms for images
pic_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])







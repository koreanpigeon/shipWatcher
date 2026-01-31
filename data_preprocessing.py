from pathlib import Path
from PIL import Image


train_images = []
train_labels = []
val_images = []
val_labels = []

empty_train_folder = Path("/Users/kimjimin/Desktop/Empty_train")
person_train_folder = Path("/Users/kimjimin/Desktop/Person_train")
vehicle_train_folder = Path("/Users/kimjimin/Desktop/Vehicle_train")
empty_val_folder = Path("/Users/kimjimin/Desktop/Empty_val")
person_val_folder = Path("/Users/kimjimin/Desktop/Person_val")
vehicle_val_folder = Path("/Users/kimjimin/Desktop/Vehicle_val")


for class_index, folder in enumerate([empty_train_folder, person_train_folder, vehicle_train_folder]):
    for file in folder.iterdir():
        if file.is_file():
            try:
                with Image.open(file) as img:
                    train_images.append(file)
                    train_labels.append(class_index)
            except(Image.UnidentifiedImageError, IOError):
                continue

for class_index, folder in enumerate([empty_val_folder, person_val_folder, vehicle_val_folder]):
    for file in folder.iterdir():
        if file.is_file():
            try:
                with Image.open(file) as img:
                    val_images.append(file)
                    val_labels.append(class_index)
            except(Image.UnidentifiedImageError, IOError):
                continue

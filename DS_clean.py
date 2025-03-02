import os
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
dataset_path = r"C:\Users\ashwi\GUVI_Projects\PlantDisease\Dataset\train"
output_path = r"C:\Users\ashwi\GUVI_Projects\PlantDisease\Dataset\balanced"

# Step 1: Copy Dataset to Balanced Folder
if not os.path.exists(output_path):
    os.makedirs(output_path)

class_counts_before = {class_folder: len(os.listdir(os.path.join(dataset_path, class_folder))) for class_folder in os.listdir(dataset_path)}
print("Image counts before balancing:", class_counts_before)

for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    save_path = os.path.join(output_path, class_folder)
    os.makedirs(save_path, exist_ok=True)
    
    for img_name in os.listdir(class_path):
        shutil.copy(os.path.join(class_path, img_name), save_path)

# Step 2: Balance the Dataset
max_images = max(class_counts_before.values())
target_images = int(max_images * 0.52)

def augment_image(image_path, save_path, count):
    """Generate augmented images."""
    img = load_img(image_path)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    datagen = ImageDataGenerator(rotation_range=30, zoom_range=0.2, horizontal_flip=True)
    
    i = 0
    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=save_path, save_prefix="aug", save_format="jpg"):
        i += 1
        if i >= count:
            break

class_counts_after_balancing = {}
for class_folder in os.listdir(output_path):
    class_path = os.path.join(output_path, class_folder)
    images = os.listdir(class_path)
    num_images = len(images)
    
    if num_images > target_images:
        # Undersampling
        images_to_remove = random.sample(images, num_images - target_images)
        for img_name in images_to_remove:
            os.remove(os.path.join(class_path, img_name))
    elif num_images < target_images:
        # Oversampling (Augmentation)
        images_needed = target_images - num_images
        sample_images = random.choices(images, k=images_needed)
        for img_name in sample_images:
            augment_image(os.path.join(class_path, img_name), class_path, 1)
    
    class_counts_after_balancing[class_folder] = len(os.listdir(class_path))

print("Image counts after balancing:", class_counts_after_balancing)
print("Dataset balancing complete. Balanced dataset is stored in:", output_path)

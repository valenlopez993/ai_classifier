import os
from pathlib import Path
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

root_folder_path = Path(__file__).parent.parent
root_folder_path = os.path.join(root_folder_path, "..")
images_path = os.path.join(root_folder_path, "images")

element = "tornillos"

# Initialize ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

for i in os.listdir(os.path.join(images_path, element)):
    # Choose an image to augment
    img_path = os.path.join(images_path, f"{element}/{i}")
    img = load_img(img_path)  # Load image

    # Convert image to numpy array
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Reshape to (1, height, width, channels)

    # Generate augmented images and save to directory
    save_dir = os.path.join(images_path, f"{element}_augmentadas")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate augmented images
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=element, save_format='jpg'):
        i += 1
        if i >= 3:  # Generate 20 augmented images
            break
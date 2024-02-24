import os 
from pathlib import Path

root_folder_path = Path(__file__).parent.parent
root_folder_path = os.path.join(root_folder_path, "..")

images_path = os.path.join(root_folder_path, "images")
elements = ["clavos", "tornillos", "tuercas", "arandelas"]
for element in elements:
    element_path = os.path.join(images_path, element)
    imgs = os.listdir(element_path)
    for count, img in enumerate(imgs):
        img_path = os.path.join(element_path, img)
        os.rename(img_path, os.path.join(element_path, f"{element}{count}.jpg"))
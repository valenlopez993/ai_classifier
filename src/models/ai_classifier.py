from abc import ABC, abstractmethod
import os
import cv2
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops

class AIClassifier(ABC):

    root_folder_path = Path(__file__).parent.parent
    root_folder_path = os.path.join(root_folder_path, "..")

    def __init__(self):
        # Load images parameters
        self.img_crop_size = 2160

        # Preprocess parameters
        self.kernel_close_size = 7
        self.kernel_open_size = 7
        self.kernel_close = np.ones((self.kernel_close_size, self.kernel_close_size), np.uint8)
        self.kernel_open = np.ones((self.kernel_open_size, self.kernel_open_size), np.uint8)

        # Relation cm/px
        self.relation_cm_px = 8.8/4032

        # Elements to classify
        self.elements = {
            "tuercas" : 0, 
            "tornillos" : 0, 
            "arandelas" : 0, 
            "clavos" : 0
        }

        # Data for plots
        self.image_props_label = [
            "Excentricidad", 
            "Momento de Hu 1", 
            "Momento de Hu 2",
            "Momento de Hu 3",
            "Momento de Hu 4",
            "Momento de Hu 5",
            "Momento de Hu 6",
            "Momento de Hu 7"
        ]

        self.plot_colors = {
            "Nuevo Objeto" : "red",
            "Tuercas" : "blue",
            "Tornillos" : "green",
            "Arandelas" : "black",
            "Clavos" : "orange",
        }

    @abstractmethod
    def fit(self, train_images, train_labels, clusters_tags):
        pass
    
    @abstractmethod
    def predict(self, imgs, k : int = 3):
        pass

    def euclidean_distance(self, img, train_img):
        return np.sqrt(np.sum((img - train_img)**2))

    # Method to calculate the length of "nails" and "screws"
    def calculate_length(self, img, orientation):
        # Rotate the image to put the object vertically
        angle = math.degrees(-orientation)
        scale = 1.0
        (height, width) = img.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_img = cv2.warpAffine(img, matrix, (width, height))

        # Labeling to get the bbox of the biggest object
        label_image = label(rotated_img)
        regions = regionprops(label_image)
        area = 0
        for props in regions:
            if props.area > area:
                area = props.area
                minr, minc, maxr, maxc = props.bbox
                length = maxr - minr

        length = round(length * self.relation_cm_px, 2) # cm
        return f"{length} cm"
    
    # Method to crop and resize the images to always have the same size
    def preprocess_image(self, imgs):
        imgs_resized = []
        for img in imgs:
            # crop and resize image
            y_size, x_size = img.shape
            img_cropped = img[
                round(y_size/2)-round(self.img_crop_size/2) : round(y_size/2)+round(self.img_crop_size/2), 
                round(x_size/2)-round(self.img_crop_size/2) : round(x_size/2)+round(self.img_crop_size/2)
            ]
            img_resized = cv2.resize(img_cropped, (self.img_crop_size, self.img_crop_size))
            imgs_resized.append(img_resized)
        return imgs_resized

    def load_images(self, elements):
        # load images
        train_data = []
        train_labels = []

        images_path = os.path.join(AIClassifier.root_folder_path, "images")

        for element in elements:
            imgs = os.listdir(f"{images_path}/{element}")
            self.elements[element] = len(imgs)
            for img in imgs:
                img_new = cv2.imread(
                    f"{images_path}/{element}/{img}",
                    cv2.IMREAD_GRAYSCALE
                )
                if img_new is not None:
                    img_resized = self.preprocess_image([img_new])
                    train_data.append(np.array(img_resized[0]))
                    train_labels.append(elements.index(element))

                else:
                    print(f"Error: It is not possible to read the image {element}/{img}")

        # convert to numpy array
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        return train_data, train_labels

    # Method to convert the images to a vectors representation
    def img_to_vec(self, images):
        # Preprocess train images

        # Create an array to store the following properties of the images:
        # - Eccentricity
        # - 7 Hu moments
        img_vec = np.empty([1, 8])
        
        for img in images:

            # Binarize image
            thresh = threshold_triangle(img)
            image = img > thresh
            image_bw = image.astype(np.uint8) * 255

            # Closing
            image_close = cv2.erode(cv2.dilate(image_bw, self.kernel_close, iterations=1), self.kernel_close, iterations=1)
            # Opening
            image_open = cv2.dilate(cv2.erode(image_close, self.kernel_open, iterations=1), self.kernel_open, iterations=1)

            # Labeling (identify objects)
            label_image = label(image_open)
            # Get properties of objects
            regions = regionprops(label_image)

            # Get the biggest object and its properties
            area = 0
            for props in regions:
                if props.area > area:
                    main_label = props.label
                    orientation = props.orientation
                    area = props.area
                    eccentricity = props.eccentricity 
                    moments_hu = props.moments_hu

            # Get just the biggest object from the image
            label_image[label_image != main_label] = 0

            new_row = np.array([
                eccentricity, 
                *moments_hu
            ])

            img_vec = np.append(img_vec, [new_row], axis=0)

        return img_vec[1:], orientation, image_bw, image_close, image_open, label_image
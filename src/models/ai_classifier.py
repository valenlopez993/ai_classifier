from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import threshold_yen, threshold_mean, threshold_triangle, rank
from skimage.measure import label, regionprops

class AIClassifier(ABC):

    root_folder_path = Path(__file__).parent.parent
    root_folder_path = os.path.join(root_folder_path, "..")

    @abstractmethod
    def fit(self, train_images, train_labels, clusters_tags):
        pass
    
    @abstractmethod
    def predict(self, imgs, k : int = 3):
        pass

    def euclidean_distance(self, img, train_img):
        return np.sqrt(np.sum((img - train_img)**2))

    def load_images(self, elements):
        # load images
        train_data = []
        train_labels = []

        images_path = f"{AIClassifier.root_folder_path}/images"

        for element in elements:

            for img in os.listdir(f"{images_path}/{element}"):
                img_new = cv2.imread(
                    f"{images_path}/{element}/{img}",
                    cv2.IMREAD_GRAYSCALE
                )
                if img_new is not None:
                    img_resized = cv2.resize(img_new, (500, 500))
                    train_data.append(np.array(img_resized))
                    train_labels.append(elements.index(element))

                else:
                    print(f"Error: It is not possible to read the image {element}/{img}")

        # convert to numpy array
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        return train_data, train_labels

    def preprocess(self, images):

        # Preprocess train images
        img_vec = np.empty([1, 10])
        for img in images:
            
            # Apply filter
            gamma_corrected = exposure.adjust_sigmoid(img)

            # Binarize image
            thresh = threshold_triangle(img)
            image = img > thresh
            image_bw = image.astype(np.uint8) * 255

            # Closing
            image_close = cv2.erode(cv2.dilate(image_bw, self.kernel, iterations=1), self.kernel, iterations=1)
            # Opening
            image_open = cv2.dilate(cv2.erode(image_close, self.kernel, iterations=1), self.kernel, iterations=1)

            # Labeling (identify objects)
            label_image = label(image_open)
            # Get properties of objects
            regions = regionprops(label_image)

            # Get the biggest object and its properties
            area = 0
            for props in regions:
                if props.area > area:
                    main_label = props.label
                    endpoints = props.bbox
                    area = props.area
                    perimeter = props.perimeter
                    eccentricity = props.eccentricity 
                    moments_hu = props.moments_hu

            # Remove the biggest object from the image
            label_image[label_image != main_label] = 0

            new_row = np.array([
                area, 
                perimeter,
                eccentricity, 
                *moments_hu
            ])

            img_vec = np.append(img_vec, [new_row], axis=0)

        return img_vec[1:], endpoints, gamma_corrected, image_bw, image_close, image_open, label_image
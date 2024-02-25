from abc import ABC, abstractmethod
import os
import cv2
import math
import numpy as np
from pathlib import Path
from skimage.filters import threshold_triangle
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

        return length
    
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

            for img in os.listdir(f"{images_path}/{element}"):
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
        img_vec = np.empty([1, 10])
        for img in images:

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
                    orientation = props.orientation
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

        return img_vec[1:], orientation, image_bw, image_close, image_open, label_image
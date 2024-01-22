import os
import cv2
import numpy as np
from pathlib import Path
import logging
from skimage.filters import threshold_yen, threshold_mean, threshold_triangle
from skimage.measure import label, regionprops

class KNNClassifier:

    main_logger = logging.Logger("KNNClassifier")

    root_folder_path = Path(__file__).parent.parent
    root_folder_path = os.path.join(root_folder_path, "..")
        
    def __init__(self):
        
        # Preprocess parameters
        self.kernel_size = 5
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        elements = ["tuercas", "tornillos", "arandelas", "clavos"]
        train_data, train_labels = self.load_images(elements)
        self.fit(train_data, train_labels, elements)


    def load_images(self, elements):
        # load images
        train_data = []
        train_labels = []

        images_path = f"{KNNClassifier.root_folder_path}/images/Fotos"

        for element in elements:

            for img in os.listdir(f"{images_path}/{element}"):
                img_new = cv2.imread(
                    f"{images_path}/{element}/{img}",
                    cv2.IMREAD_GRAYSCALE
                )
                if img_new is not None:
                    img_resized = cv2.resize(img_new, (500, 500))
                    train_data.append(np.array(img_resized).flatten())
                    train_labels.append(elements.index(element))

                else:
                    print(f"Error: It is not possible to read the iamge {element}/{img}")

        # convert to numpy array
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        return train_data, train_labels

    def __preprocess(self, images):

        # Preprocess train images
        img_vec = np.empty([1, 10])
        for img in images:
            # apply threshold to a gray image
            thresh = threshold_triangle(img)
            #image = np.bitwise_not(gray > yen)
            image = img > thresh
            image = image.astype(np.uint8) * 255

            # Opening
            #image = cv2.dilate(cv2.erode(image, kernel, iterations=1), kernel, iterations=1)
            # Closing
            image = cv2.erode(cv2.dilate(image, self.kernel, iterations=1), self.kernel, iterations=1)

            # Labeling (identify objects)
            label_image = label(image)
            # Get properties of objects
            regions = regionprops(label_image)

            # Get the biggest object and its properties
            area = 0
            for props in regions:
                if props.area > area:
                    area = props.area
                    perimeter = props.perimeter
                    eccentricity = props.eccentricity 
                    moments_hu = props.moments_hu

            new_row = np.array([
                area, 
                perimeter, 
                eccentricity, 
                *moments_hu
            ])

            img_vec = np.append(img_vec, [new_row], axis=0)

        return img_vec[1:]

    def fit(
        self, 
        train_images : np.ndarray, 
        train_labels : np.ndarray,
        clusters_tags : list = None
    ):
        if (not isinstance(train_images, np.ndarray) or not isinstance(train_images, np.ndarray)):
            raise TypeError("train_images and train_labels must be numpy arrays")
        if (train_images.shape[0] != train_labels.shape[0]):
            raise Exception("Shape mistmatch: train_images and train_labels must have the same number of elements")
        if train_images.ndim != 2:
            raise Exception("train_images must have 2 dimensions")
        if train_labels.ndim != 1:
            raise Exception("train_labels must have 1 dimension")

        self.train_images = self.__preprocess(train_images)
        self.train_labels = train_labels
        self.categories = clusters_tags

    def __euclidean_distance(self, img, train_img):
        return np.sqrt(np.sum((img - train_img)**2))

    def predict(
        self, 
        imgs,
        k : int = 3
    ):
        # if (not isinstance(imgs, np.ndarray)):
        #     raise TypeError("img must be a numpy array")

        imgs = self.__preprocess(imgs)
        
        predictions = []
        for img in imgs:
            distances = [
                self.__euclidean_distance(img, train_img)
                for train_img in self.train_images
            ]

            neighbors_index = np.argsort(distances)[:k]
            neighbors = self.train_labels[neighbors_index]

            most_common = np.bincount(neighbors).argmax()
            predictions.append(self.categories[most_common])
        
        return predictions
import numpy as np
import logging

from models.ai_classifier import AIClassifier

class KNNClassifier(AIClassifier):

    main_logger = logging.Logger("KNNClassifier")
        
    def __init__(self):
        
        # Preprocess parameters
        self.kernel_size = 5
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        elements = ["tuercas", "tornillos", "arandelas", "clavos"]
        train_data, train_labels = self.load_images(elements)
        self.fit(train_data, train_labels, elements)

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
        if train_labels.ndim != 1:
            raise Exception("train_labels must have 1 dimension")

        self.train_images, _, _, _, _, _ = self.preprocess(train_images)
        self.train_labels = train_labels
        self.categories = clusters_tags

    def predict(
        self, 
        imgs,
        k : int = 3
    ):

        imgs_vec, endpoints, image_thresh, image_close, image_open, label_image = self.preprocess(imgs)
        
        predictions = []
        for img_vec in imgs_vec:
            distances = [
                self.euclidean_distance(img_vec, train_img)
                for train_img in self.train_images
            ]

            neighbors_index = np.argsort(distances)[:k]
            neighbors = self.train_labels[neighbors_index]

            most_common = np.bincount(neighbors).argmax()
            predictions.append(self.categories[most_common])
        
        return (
            img_vec, 
            endpoints, 
            predictions, 
            {
                "threshold": image_thresh,
                "closing": image_close,
                "opening": image_open,
                "label_image": label_image
            })
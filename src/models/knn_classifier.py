import numpy as np
from ai_classifier_logger import AIClassifierLogger

from models.ai_classifier import AIClassifier

class KNNClassifier(AIClassifier):
        
    def __init__(self):
        
        self.logger = AIClassifierLogger("KNNClassifier")

        # Load images parameters
        self.img_crop_size = 2000

        # Preprocess parameters
        self.kernel_size = 5
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        # Relation cm/px
        self.relation_cm_px = 8.8/4032

        elements = ["tuercas", "tornillos", "arandelas", "clavos"]
        
        self.logger.debug(f"Loading dataset")
        self.train_data, train_labels = self.load_images(elements)
        self.fit(self.train_data, train_labels, elements)
        self.logger.info(f"Dataset loaded")

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

        self.train_images, _, _, _, _, _ = self.img_to_vec(train_images)
        self.train_labels = train_labels
        self.categories = clusters_tags

    def predict(
        self, 
        imgs,
        k : int = 3
    ):

        self.logger.info(f"Predicting")

        # Preprocess and vectorize the image
        self.logger.info(f"Preprocessing images")
        imgs_resized = self.preprocess_image(imgs)
        imgs_vec, orientation, image_bw, image_close, image_open, label_image = self.img_to_vec(imgs_resized)
        
        self.logger.info(f"Running algorithm")
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

            # Calculate the length of the object if it is a "nail" or a "screw"
            objects_length = []
            for prediction in predictions:
                if prediction in ["clavos", "tornillos"]:
                    self.logger.warning(f"Calculating length")
                    objects_length.append(
                        f"{round(self.calculate_length(image_open, orientation) * self.relation_cm_px, 2)} cm"
                    )
                else:
                    objects_length.append(None)

        self.logger.info(f"Predicting done")
            
        return (
            img_vec,
            predictions, 
            objects_length,
            {
                "Escala de grises": imgs_resized[0],	
                "Binarizada": image_bw,
                "Cierre": image_close,
                "Apertura": image_open,
                "label_image": label_image
            })
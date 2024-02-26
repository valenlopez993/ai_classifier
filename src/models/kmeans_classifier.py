import numpy as np
from ai_classifier_logger import AIClassifierLogger

from models.ai_classifier import AIClassifier

class KMeansClassifier(AIClassifier):
        
    def __init__(self):
        super().__init__()

        self.logger = AIClassifierLogger("KMeansClassifier")

        # K-means parameters
        self.k_means_iterations = 5
        self.k = len(self.elements)

        self.logger.debug(f"Loading dataset")
        self.train_data, train_labels = self.load_images(self.elements)

        self.train_images, _, _, _, _, _ = self.img_to_vec(self.train_data)
        self.train_labels = train_labels
        self.categories = self.elements
        self.logger.info(f"Dataset loaded")

    def fit():
        pass

    def predict(
        self, 
        imgs
    ):

        self.logger.info(f"Predicting")
        # Preprocess and vectorize the image
        imgs_resized = self.preprocess_image(imgs)
        img_vec, orientation, image_bw, image_close, image_open, label_image = self.img_to_vec(imgs_resized)
        
        # Build a vector with the images to predict in the 0th position and the train images
        imgs_vec = np.concatenate([img_vec, self.train_images])

        # KMeans algorithm
        self.logger.info(f"Running algorithm")
        # Repeat the algorithm "self.k_means_iterations" times and select the centroids with the lowest variance
        centroids_variance_old = np.array([np.inf for i in range(self.k)])
        for it in range(self.k_means_iterations):

            # Initialize the centroids with random images from the train set
            random_args = np.random.randint(0, self.train_images.shape[0], self.k)
            centroids = self.train_images[random_args]

            # Repeat the algorithm until the centroids do not change more than the "centroids_error"
            old_centroids = np.copy(centroids)
            centroids_error = 0.01
            while (abs(np.sum(old_centroids - centroids)) < centroids_error):

                # Calculate the distance of each image to each centroid
                distances = [
                    [
                        self.euclidean_distance(img, centroid)
                        for centroid in centroids
                    ]
                    for img in imgs_vec
                ]

                # Assign the closest centroid to each image
                closest_centroids = np.argmin(distances, axis=1)

                # Update the centroids
                old_centroids = np.copy(centroids)
                for i in range(self.k):
                    new_centroid = np.mean(imgs_vec[closest_centroids == i], axis=0)
                    if not np.isnan(new_centroid).any():
                        centroids[i] = new_centroid

            # After the algorithm converges, calculate the variance of the centroids
            centroids_variance = []
            for i in range(centroids.shape[0]):
                centroids_variance_by_dim = np.var(imgs_vec[closest_centroids == i], axis=0)
                if not np.isnan(centroids_variance_by_dim).any():
                    centroids_variance.append(np.var(centroids_variance_by_dim))
                else:
                    centroids_variance.append(np.max(distances))

            # If the variance of the centroids is lower than the previous one, 
            # save the centroids, the variance and the prediction
            if (sum(centroids_variance) < sum(centroids_variance_old)):
                centroids_variance_old = centroids_variance.copy()
                final_centroids = centroids.copy()

                # Get the closest centroid to the image to predict that is in the 0th position
                prediction = self.categories[closest_centroids[0]]

                # Calculate the length of the object if it is a "nail" or a "screw"
                if prediction in ["clavos", "tornillos"]:
                    self.logger.warning(f"Calculating length")
                    object_length = (
                        self.calculate_length(image_open, orientation)
                    )
                else:
                    object_length = None
        
        self.logger.info(f"Predicting done")

        return (
            img_vec,
            final_centroids,
            prediction,
            object_length, 
            {
                "Escala de grises": imgs_resized[0],	
                "Binarizacion": image_bw,
                "Cierre": image_close,
                "Apertura": image_open,
                "label_image": label_image
            })
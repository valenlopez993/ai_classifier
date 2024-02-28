import numpy as np
import matplotlib.pyplot as plt
from ai_classifier_logger import AIClassifierLogger

from models.ai_classifier import AIClassifier

class KMeansClassifier(AIClassifier):
        
    def __init__(self):
        super().__init__()

        self.logger = AIClassifierLogger("KMeansClassifier")

        # K-means parameters
        self.k = len(list(self.elements.keys()))

        self.logger.debug(f"Loading dataset")
        self.train_data, train_labels = self.load_images(list(self.elements.keys()))

        self.train_images, _, _, _, _, _ = self.img_to_vec(self.train_data)
        self.train_labels = train_labels.copy()
        self.categories = list(self.elements.keys())
        self.logger.info(f"Dataset loaded")

    def fit():
        pass

    def identify_clusters_and_get_prediction(
        self, 
        centroids,
        closest_centroid,
        knn_k=6
    ):
        # KNN algorithm
        knn_k = 6

        # Calculate the distance of each image to each centroid
        distances = [
            [
                self.euclidean_distance(centroid, img)
                for img in self.train_images
            ]
            for centroid in centroids
        ]

        # Get the k-nearest neighbors of each centroid
        neighbors_index = np.argpartition(distances, knn_k, axis=1)[:, :knn_k]
        neighbors = self.train_labels[neighbors_index]

        # Get the most common category of the k-nearest neighbors
        most_commons = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=neighbors)
        
        # Create an aux array to store the new order full of "inf"
        self.kmeans_labels = np.copy(closest_centroid)
        self.kmeans_labels[:] = self.k
        # Order the centroids by the same order of the categories
        final_centroids = []
        for category in self.categories:
            # Get the index of the category in the "most_commons" array
            centroid_category_index = np.argwhere(most_commons == self.categories.index(category))[0]
            # Update the closest centroid with the new centroids's order
            self.kmeans_labels[closest_centroid == centroid_category_index] = self.categories.index(category)
            # Reorder the centroids
            final_centroids.append(centroids[centroid_category_index][0])
        # Convert the list to a NumPy array    
        final_centroids = np.vstack([final_centroids[0], final_centroids[1], final_centroids[2], final_centroids[3]])

        # Get the category of the closest centroid to the new datapoint
        prediction = self.categories[self.kmeans_labels[0]]

        return final_centroids, prediction

    def predict(
        self, 
        img
    ):
        self.logger.info(f"Predicting")
        # Preprocess and vectorize the image
        img_resized = self.preprocess_image(img)
        img_vec, orientation, image_bw, image_close, image_open, label_image = self.img_to_vec([img_resized])
        img_vec = img_vec[0]
        
        # Build a vector with the images to predict in the 0th position and the train images
        imgs_vec = np.vstack([img_vec, self.train_images])

        # KMeans algorithm
        self.logger.info(f"Running algorithm")

        # Initialize the centroids with random images from the train set
        random_args = np.random.randint(0, self.train_images.shape[0], self.k)
        centroids = self.train_images[random_args]

        # Repeat the algorithm until the centroids do not change more than the "centroids_error"
        old_centroids = np.copy(centroids)
        centroids_error = 0.001
        while (True):

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

            # Break the loop if the centroids do not change more than the "centroids_error"
            if (abs(np.sum(old_centroids - centroids)) < centroids_error): break

        # Indentify the cluster of each centroid and
        # get the closest centroid to the image to predict that is in the 0th position
        final_centroids, prediction = self.identify_clusters_and_get_prediction(
            centroids=centroids,
            closest_centroid=closest_centroids
        )
        
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
                "Escala de grises": img_resized,	
                "Binarizacion": image_bw,
                "Cierre": image_close,
                "Apertura": image_open,
                "label_image": label_image
            })

    def generate_data_plots(
        self, 
        img_vec, 
        centroids,
        main_prop_for_plot="Excentricidad"
    ):
        # Select the main property for the plot
        main_image_prop = self.image_props_label.index(main_prop_for_plot)
        other_image_props = list(range(len(self.image_props_label)))
        other_image_props.remove(main_image_prop)

        fig = plt.figure(figsize=(8, 8))
        figs_np = {}
        for prop2 in other_image_props:

            x_label = self.image_props_label[main_image_prop]
            y_label = self.image_props_label[prop2]

            tuercas_x = self.train_images[self.kmeans_labels[1:] == self.categories.index("tuercas"), main_image_prop]
            tuercas_y = self.train_images[self.kmeans_labels[1:] == self.categories.index("tuercas"), prop2]

            tornillos_x = self.train_images[self.kmeans_labels[1:] == self.categories.index("tornillos"), main_image_prop]
            tornillos_y = self.train_images[self.kmeans_labels[1:] == self.categories.index("tornillos"), prop2]

            arandelas_x = self.train_images[self.kmeans_labels[1:] == self.categories.index("arandelas"), main_image_prop]
            arandelas_y = self.train_images[self.kmeans_labels[1:] == self.categories.index("arandelas"), prop2]

            clavos_x = self.train_images[self.kmeans_labels[1:] == self.categories.index("clavos"), main_image_prop]
            clavos_y = self.train_images[self.kmeans_labels[1:] == self.categories.index("clavos"), prop2]

            centroids_x = centroids[:, main_image_prop]
            centroids_y = centroids[:, prop2]

            # Create the scatter plot for the dataset
            plt.scatter(tuercas_x, tuercas_y, c=self.plot_colors["Tuercas"], label="Tuercas")
            plt.scatter(centroids_x[0], centroids_y[0], c=self.plot_colors["Tuercas"], label="Centroide - Tuercas", marker="D", s=100, linewidths=4)

            plt.scatter(tornillos_x, tornillos_y, c=self.plot_colors["Tornillos"], label="Tornillos")
            plt.scatter(centroids_x[1], centroids_y[1], c=self.plot_colors["Tornillos"], label="Centroide - Tornillos", marker="D", s=100, linewidths=4)

            plt.scatter(arandelas_x, arandelas_y, c=self.plot_colors["Arandelas"], label="Arandelas")
            plt.scatter(centroids_x[2], centroids_y[2], c=self.plot_colors["Arandelas"], label="Centroide - Arandelas", marker="D", s=100, linewidths=4)

            plt.scatter(clavos_x, clavos_y, c=self.plot_colors["Clavos"], label="Clavos")
            plt.scatter(centroids_x[3], centroids_y[3], c=self.plot_colors["Clavos"], label="Centroide - Clavos", marker="D", s=100, linewidths=4)

            # Create the scatter plot for the new datapoint
            new_obj_x = img_vec[main_image_prop]
            new_obj_y = img_vec[prop2]
            plt.scatter(new_obj_x, new_obj_y, c=self.plot_colors["Nuevo Objeto"], label="Nuevo Objeto", marker="x", s=100, linewidths=4)
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True)

            # Convert the plot to a NumPy array
            canvas = fig.canvas
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

            figs_np[f"plot_{y_label} vs {x_label}"] = image_np

            fig.clear()

        return figs_np
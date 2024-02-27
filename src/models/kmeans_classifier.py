import numpy as np
import matplotlib.pyplot as plt
from ai_classifier_logger import AIClassifierLogger

from models.ai_classifier import AIClassifier

class KMeansClassifier(AIClassifier):
        
    def __init__(self):
        super().__init__()

        self.logger = AIClassifierLogger("KMeansClassifier")

        # K-means parameters
        self.k_means_iterations = 5
        self.k = len(list(self.elements.keys()))

        self.logger.debug(f"Loading dataset")
        self.train_data, train_labels = self.load_images(list(self.elements.keys()))

        self.train_images, _, _, _, _, _ = self.img_to_vec(self.train_data)
        self.train_labels = train_labels
        self.categories = list(self.elements.keys())
        self.logger.info(f"Dataset loaded")

    def fit():
        pass

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
            for i in range(self.k):
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

        num_tuercas = self.elements["tuercas"]
        num_tornillos = self.elements["tornillos"]
        num_arandelas = self.elements["arandelas"]
        num_clavos = self.elements["clavos"]

        fig = plt.figure(figsize=(8, 8))
        figs_np = {}
        for prop2 in other_image_props:

            x_label = self.image_props_label[main_image_prop]
            y_label = self.image_props_label[prop2]

            tuercas_x = self.train_images[0:num_tuercas, main_image_prop]
            tuercas_y = self.train_images[0:num_tuercas, prop2]

            tornillos_x = self.train_images[num_tuercas:num_tuercas+num_tornillos, main_image_prop]
            tornillos_y = self.train_images[num_tuercas:num_tuercas+num_tornillos, prop2]

            arandelas_x = self.train_images[num_tuercas+num_tornillos:num_tuercas+num_tornillos+num_arandelas, main_image_prop]
            arandelas_y = self.train_images[num_tuercas+num_tornillos:num_tuercas+num_tornillos+num_arandelas, prop2]

            clavos_x = self.train_images[num_tuercas+num_tornillos+num_arandelas:, main_image_prop]
            clavos_y = self.train_images[num_tuercas+num_tornillos+num_arandelas:, prop2]

            centroids_x = centroids[:, main_image_prop]
            centroids_y = centroids[:, prop2]

            # Create the scatter plot for the new datapoint
            new_obj_x = img_vec[main_image_prop]
            new_obj_y = img_vec[prop2]
            plt.scatter(new_obj_x, new_obj_y, c=self.plot_colors["Nuevo Objeto"], label="Nuevo Objeto", marker="x")
            
            # Create the scatter plot for the dataset
            plt.scatter(tuercas_x, tuercas_y, c=self.plot_colors["Tuercas"], label="Tuercas")
            plt.scatter(centroids_x[0], centroids_y[0], c=self.plot_colors["Tuercas"], label="Tuercas", marker="D")

            plt.scatter(tornillos_x, tornillos_y, c=self.plot_colors["Tornillos"], label="Tornillos")
            plt.scatter(centroids_x[1], centroids_y[1], c=self.plot_colors["Tornillos"], label="Tornillos", marker="D")

            plt.scatter(arandelas_x, arandelas_y, c=self.plot_colors["Arandelas"], label="Arandelas")
            plt.scatter(centroids_x[2], centroids_y[2], c=self.plot_colors["Arandelas"], label="Arandelas", marker="D")

            plt.scatter(clavos_x, clavos_y, c=self.plot_colors["Clavos"], label="Clavos")
            plt.scatter(centroids_x[3], centroids_y[3], c=self.plot_colors["Clavos"], label="Clavos", marker="D")

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
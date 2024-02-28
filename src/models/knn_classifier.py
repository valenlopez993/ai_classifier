import numpy as np
import matplotlib.pyplot as plt
from ai_classifier_logger import AIClassifierLogger

from models.ai_classifier import AIClassifier

class KNNClassifier(AIClassifier):
        
    def __init__(self):
        super().__init__()
        
        self.logger = AIClassifierLogger("KNNClassifier")
        
        self.logger.debug(f"Loading dataset")
        self.train_data, train_labels = self.load_images(list(self.elements.keys()))
        self.fit(self.train_data, train_labels, list(self.elements.keys()))
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
        self.train_labels = train_labels.copy()
        self.categories = clusters_tags

    def predict(
        self, 
        img,
        k : int = 3
    ):

        self.logger.info(f"Predicting")

        # Preprocess and vectorize the image
        self.logger.info(f"Preprocessing images")
        img_resized = self.preprocess_image(img)
        img_vec, orientation, image_bw, image_close, image_open, label_image = self.img_to_vec([img_resized])
        img_vec = img_vec[0]
        
        self.logger.info(f"Running algorithm")
        
        # Calculate the distance of each image to the new datapoint
        distances = [
            self.euclidean_distance(img_vec, train_img)
            for train_img in self.train_images
        ]

        # Get the k-nearest neighbors
        neighbors_index = np.argsort(distances)[:k]
        neighbors = self.train_labels[neighbors_index]

        # Get the most common category of the k-nearest neighbors
        most_common = np.bincount(neighbors).argmax()
        prediction = self.categories[most_common]

        # Calculate the length of the object if it is a "nail" or a "screw"
        objects_length = []
        if prediction in ["clavos", "tornillos"]:
            self.logger.warning(f"Calculating length")
            objects_length = (
                self.calculate_length(image_open, orientation)
            )
        else:
            objects_length = None

        self.logger.info(f"Predicting done")
            
        return (
            img_vec,
            prediction, 
            objects_length,
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

            tuercas_x = self.train_images[self.train_labels == self.categories.index("tuercas"), main_image_prop]
            tuercas_y = self.train_images[self.train_labels == self.categories.index("tuercas"), prop2]

            tornillos_x = self.train_images[self.train_labels == self.categories.index("tornillos"), main_image_prop]
            tornillos_y = self.train_images[self.train_labels == self.categories.index("tornillos"), prop2]

            arandelas_x = self.train_images[self.train_labels == self.categories.index("arandelas"), main_image_prop]
            arandelas_y = self.train_images[self.train_labels == self.categories.index("arandelas"), prop2]

            clavos_x = self.train_images[self.train_labels == self.categories.index("clavos"), main_image_prop]
            clavos_y = self.train_images[self.train_labels == self.categories.index("clavos"), prop2]
            
            # Create the scatter plot for the dataset
            plt.scatter(tuercas_x, tuercas_y, c=self.plot_colors["Tuercas"], label="Tuercas")
            plt.scatter(tornillos_x, tornillos_y, c=self.plot_colors["Tornillos"], label="Tornillos")
            plt.scatter(arandelas_x, arandelas_y, c=self.plot_colors["Arandelas"], label="Arandelas")
            plt.scatter(clavos_x, clavos_y, c=self.plot_colors["Clavos"], label="Clavos")

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
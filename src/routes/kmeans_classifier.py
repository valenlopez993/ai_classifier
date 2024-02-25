import cv2
import numpy as np
from flask_restful import Resource
from flask import request, send_file, Response
from models.kmeans_classifier import KMeansClassifier

import zipfile

from io import BytesIO
from PIL import Image 

class KMeansClassifierRoute(Resource):
    kmeans_classifier = KMeansClassifier()

    def post(self):
        try:
            image = request.files['image']

            image = np.frombuffer(image.read(), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            
            # img_crop_size = 500
            # y_size, x_size = image.shape
            # image = image[round(y_size/2)-img_crop_size : round(y_size/2)+img_crop_size, round(x_size/2)-img_crop_size : round(x_size/2)+img_crop_size]
            image = cv2.resize(image, (500, 500))

            img_vec, endpoints, centroids, category, images_dict = KMeansClassifierRoute.kmeans_classifier.predict([image])
            print(img_vec)
            # Delete the label image because cannot be serialized
            del images_dict["label_image"]

            # Add the grayscale image to the dictionary
            images_dict = {
                "grayscale": image,
                **images_dict
            }

           # Save in memory the category
            categoryTxt = BytesIO()
            categoryTxt.write(f"{category}\n".encode())

            # Create a BytesIO object to store the ZIP file in memory
            zip_buffer = BytesIO()  

            # Create a ZipFile object
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:

                # Add the category to the ZIP file
                zip_file.writestr(f'category.txt', categoryTxt.getvalue())

                for img_name in images_dict:

                    # Convert NumPy array to PIL Image
                    pil_image = Image.fromarray(images_dict[img_name])

                    # Create an in-memory file-like object
                    image_buffer = BytesIO()
                    pil_image.save(image_buffer, format='JPEG')  # You can adjust the format as needed

                    # Add the in-memory file to the ZIP file
                    zip_file.writestr(f'{img_name}.jpg', image_buffer.getvalue())

            # Seek to the beginning of the buffer
            zip_buffer.seek(0)

            # Return the ZIP file as a response
            return send_file(zip_buffer, download_name="images.zip", as_attachment=True)
        
        except Exception as e:
            print(e)
            return {"error": str(e)}, 500

import cv2
import numpy as np
from flask_restful import Resource
from flask import request, send_file, Response
from models.knn_classifier import KNNClassifier

import zipfile

from requests_toolbelt.multipart.encoder import MultipartEncoder
from io import BytesIO
from PIL import Image 

class KNNClassifierRoute(Resource):
    knn_classifier = KNNClassifier()

    def post(self):
        try:
            image = request.files['image']

            image = np.frombuffer(image.read(), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (500, 500))

            category, image_thresh, image_close, label_image = KNNClassifierRoute.knn_classifier.predict([image], k=6)

            zip_name = f"{category}.zip"
            images_dict = {
                "grayscale": image,
                "thresh": image_thresh,
                "close": image_close
            }

           # Save in memory the category
            categoryTxt = BytesIO()
            categoryTxt.write(f"{category[0]}\n".encode())

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
            return send_file(zip_buffer, download_name=zip_name, as_attachment=True)
        
        except Exception as e:
            print(e)
            return {"error": str(e)}, 500

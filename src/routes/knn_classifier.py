import cv2
import numpy as np
from flask_restful import Resource
from flask import request
from models.knn_classifier import KNNClassifier

class KNNClassifierRoute(Resource):
    knn_classifier = KNNClassifier()

    def post(self):
        
        try:
            image = request.files['image']

            image = np.frombuffer(image.read(), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (500, 500))
            image_flatten = np.array(image).flatten()

            category = KNNClassifierRoute.knn_classifier.predict([image_flatten], k=6)

            return {"category": category[0]}, 200
        
        except Exception as e:
            print(e)
            return {"error": str(e)}, 500

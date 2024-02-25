from flask import Flask
from flask_restful import Api

from routes.knn_classifier import KNNClassifierRoute
from routes.kmeans_classifier import KMeansClassifierRoute

app = Flask(__name__)
api = Api(app)
api.add_resource(KNNClassifierRoute, '/knn_classifier')
api.add_resource(KMeansClassifierRoute, '/kmeans_classifier')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

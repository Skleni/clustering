import umap
import hdbscan
import os

from flask import Flask, request, jsonify

port = int(os.environ.get("PORT", 5000))

app = Flask(__name__)

@app.route('/dimensions/<int:dimensions>', methods=['POST'])
def reduce_dimensions(dimensions):
    vectors = request.get_json()
    
    embedding = umap.UMAP(
        n_neighbors=10,
        min_dist=0.0,
        n_components=dimensions,
        random_state=42,
    ).fit_transform(vectors)

    return jsonify(embedding.tolist())

@app.route('/label', methods=['POST'])
def label():
    vectors = request.get_json()

    labels = hdbscan.HDBSCAN(
        min_samples=1,
        min_cluster_size=5,
    ).fit_predict(vectors)

    return jsonify(labels.tolist())

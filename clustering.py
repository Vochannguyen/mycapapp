from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
import umap
from umap import UMAP


def kmeans_cluster(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)

    return kmeans.labels
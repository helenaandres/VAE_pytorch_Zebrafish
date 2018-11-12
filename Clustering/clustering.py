from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import numpy as np
import math


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from keras.models import load_model

num_clusters = 5
n_components = 50

def PCA_dim_reduction(train_data, input_data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(train_data)

    return pca.transform(input_data)


def Spectral_embedding(input_data, n_components):
    embedding = SpectralEmbedding(n_components=n_components, affinity='rbf').fit_transform(input_data)
    return embedding


def TSNE_embedding(input_data, n_components):
    embedding = TSNE(n_components=n_components).fit_transform(input_data)
    return embedding


def AE_dim_reduction(input_data, encoder_filename):
    encoder = load_model(encoder_filename)
    return encoder.predict(input_data)

def compute_clustering_gaussian(data, num_clusters, embedding_type):
    if embedding_type is 'TSNE':
        embedding = TSNE_embedding(data, n_components=2)
    else:
        embedding = data

    gaussian = GaussianMixture(n_components=num_clusters).fit(embedding)
    gaussian_labels = gaussian.predict(embedding)
    #gaussian_score = adjusted_rand_score(labels, gaussian_labels)
    return gaussian_labels


def compute_clustering_kmeans(data, num_clusters, embedding_type):
    if embedding_type is 'TSNE':
        embedding = TSNE_embedding(data, n_components=2)
    else:
        embedding = data
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(embedding)
    #kmeans_score = adjusted_rand_score(labels, kmeans)

    return kmeans


def compute_results_with_standard_error(data, labels, num_clusters, embedding_type, clustering_type):
    results = []

    for i in range(50):
        print (i)
        if clustering_type == 'k-means':
            score = compute_clustering_scores_kmeans(data, labels, num_clusters, embedding_type)
        else:
            print ("gaussian")
            score = compute_clustering_scores_gaussian(data, labels, num_clusters, embedding_type)
        results.append(score)
    return np.array(results)
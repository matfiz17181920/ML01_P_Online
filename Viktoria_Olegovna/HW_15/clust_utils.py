import numpy as np
import pickle

from sklearn.metrics import silhouette_score, v_measure_score
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt


def group_mean(X, col):
    col_uniq = np.sort(np.unique(col))
    res = np.array([np.mean(X[col == col_i], axis=0) for col_i in col_uniq])

    return res


def plot_faces(data, shape, suptitle, interpolation='bicubic'):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(hspace=0.2, wspace=0.2, top=0.94)
    fig.suptitle(suptitle, fontsize=14)
    plt.gray()

    for i in range(10):
        ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])    
        ax.imshow(data[i].reshape(shape), interpolation=interpolation)


def get_metrics(low, upper, method, data, target):
    # homogeneity = []
    silhouette = []
    v_ms = []

    for n_clusters in tqdm(range(low, upper + 1)):
        if method.__name__ == 'KMeans':
            clusterer = method(n_clusters=n_clusters, n_init=100)
        else:
            clusterer = method(n_clusters=n_clusters)

        preds = clusterer.fit_predict(data)

        # homogeneity.append(homogeneity_score(target, preds))
        silhouette.append(silhouette_score(data, preds))
        v_ms.append(v_measure_score(target, preds))

    return silhouette, v_ms


def load_metrics():
    silhouette_km = pickle.load(open("metrics/silhouette_km.pkl", "rb"))
    homogeneity_km = pickle.load(open("metrics/homogeneity_km.pkl", "rb"))
    completeness_km = pickle.load(open("metrics/completeness_km.pkl", "rb"))
    v_ms_km = pickle.load(open("metrics/v_ms_km.pkl", "rb"))

    silhouette_agc = pickle.load(open("metrics/silhouette_agc.pkl", "rb"))
    homogeneity_agc = pickle.load(open("metrics/homogeneity_agc.pkl", "rb"))
    completeness_agc = pickle.load(open("metrics/completeness_agc.pkl", "rb"))
    v_ms_agc = pickle.load(open("metrics/v_ms_agc.pkl", "rb"))


def plot_scores(data, hue, style):    
    plt.figure(figsize=(9, 6))
    sns.set(style="whitegrid")
    sns.lineplot(
    x="Количество кластеров",
    y="Значения",
    hue=hue,
    style=style,
    dashes=False,
    markers=True,
    palette="tab10",
    data=data)


def dim_reduction(meth_reduce, meth_clust, data, target):
    num_comp = [2, 5, 10, 20]
    silhouette = []
    v_ms = []

    for i in num_comp:
        reducer = meth_reduce(n_components=i, random_state=42)
        data_reduced = reducer.fit_transform(data)

        if meth_clust=='KMeans':
            clusterer = meth_clust(n_clusters=10, n_init=100)
        else:
            clusterer = meth_clust(n_clusters=10)

        preds = clusterer.fit_predict(data_reduced)

        silhouette.append(silhouette_score(data_reduced, preds))
        v_ms.append(v_measure_score(target, preds))

    return silhouette, v_ms

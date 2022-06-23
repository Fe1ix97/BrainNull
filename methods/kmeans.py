import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import auc, roc_curve, classification_report, zero_one_loss, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler



def kmns(x_train, x_test, y_train, y_test, infoPrint):
    # DECISIONE TREE CLASSIFIER
    if infoPrint:
        print("\n-- K-MEANS CLASSIFIER")

    kmeans_model =KMeans(n_clusters=1)
    kmeans_model.fit(x_train, y_train)
    y_pred_dct = kmeans_model.predict(x_test)
    kms_fpr, kms_tpr, thr = roc_curve(y_test, y_pred_dct)
    accuracy = auc(kms_fpr, kms_tpr)
    if infoPrint:
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred_dct))

    return kmeans_model, kms_fpr, kms_tpr

def kmns_k_search(x,y):
    dataframe = pd.DataFrame(x, y)
    print(dataframe.head())
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(dataframe)
    scaled_dataframe = pd.DataFrame(scaled_array, columns=dataframe.columns)
    kmeans_model = KMeans(n_clusters=1)
    kmeans_model.fit(scaled_array)
    scaled_dataframe["cluster"] = kmeans_model.labels_
    k_to_test = range(2, 25, 1)  # [2,3,4, ..., 24]
    silhouette_scores = {}

    for k in k_to_test:
        model_kmeans_k = KMeans(n_clusters=k)
        model_kmeans_k.fit(scaled_dataframe.drop("cluster", axis=1))
        labels_k = model_kmeans_k.labels_
        score_k = metrics.silhouette_score(scaled_dataframe.drop("cluster", axis=1), labels_k)
        silhouette_scores[k] = score_k
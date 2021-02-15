import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import streamlit as st


def get_dataset(dataset_name):
    dataset = None
    if dataset_name == "Iris":
        dataset = datasets.load_iris()
    elif dataset_name == "Wine":
        dataset = datasets.load_wine()
    else:
        dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    return X, y


def add_param_sidebar(classifier):
    params = {}
    if classifier == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif classifier == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
    return params


def get_model(classifier, params):
    model = None
    if classifier == "SVM":
        model = SVC(C=params["C"])
    elif classifier == "KNN":
        model = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        model = RandomForestClassifier(n_estimators=params["n_estimators"],
                                       max_depth=params["max_depth"],
                                       random_state=60)
    return model


# Main
st.title("ML Classification App")
st.write("""### Apply and explore different classifiers on datasets!""")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Wine", "Breast Cancer"))
st.write(f"## {dataset_name} Dataset")
classifier_name = st.sidebar.selectbox("Select classifier", ("KNN", "SVM", "Random Forest"))

X, y = get_dataset(dataset_name)
st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(np.unique(y)))
params = add_param_sidebar(classifier_name)
clf = get_model(classifier_name, params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
st.write("Model: ", clf)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Classifier: ", classifier_name)
st.write("Accuracy: ", accuracy)

#  PCA Dimensionality Reduction for 2D plot
pca = PCA(2)
X_projected = pca.fit_transform(X_test)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

# 2D Plot for predictions
fig = plt.figure()
plt.scatter(x1, x2, c=y_pred, alpha=0.8, cmap="viridis")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar()
st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="K-Means Clustering App", layout="wide")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Load trained model
with open("kmeans_iris_model.pkl", "rb") as file:
    kmeans = pickle.load(file)

# Predict cluster labels
labels = kmeans.predict(X)

# PCA for 2D plotting
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

# Create DataFrame
df = pd.DataFrame({
    'PCA1': X_projected[:, 0],
    'PCA2': X_projected[:, 1],
    'Cluster': labels
})

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
for cluster in np.unique(labels):
    subset = df[df['Cluster'] == cluster]
    ax.scatter(subset['PCA1'], subset['PCA2'], label=f"Cluster {cluster}")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Clusters (2D PCA Projection)")
ax.legend()

# Show plot in Streamlit
st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="K-Means Clustering App", layout="wide")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Sidebar slider to select number of clusters
n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# Train KMeans model with selected number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# PCA for 2D plotting
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

# Create DataFrame
df = pd.DataFrame({
    'PCA1': X_projected[:, 0],
    'PCA2': X_projected[:, 1],
    'Cluster': labels
})

# Plot clusters
fig, ax = plt.subplots(figsize=(8, 6))
for cluster in np.unique(labels):
    subset = df[df['Cluster'] == cluster]
    ax.scatter(subset['PCA1'], subset['PCA2'], label=f"Cluster {cluster}")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title(f"K-Means Clustering with k = {n_clusters}")
ax.legend()

# Display plot
st.pyplot(fig)

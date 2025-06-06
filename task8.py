import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv(r'C:\Users\Ajhay\internship\task8\Mall_Customers.csv')

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

df_numeric = df.drop(['CustomerID', 'Gender'], axis=1)

sns.pairplot(df_numeric)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_numeric)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt.grid(True)
plt.show()

optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(df_numeric)

df['Cluster'] = cluster_labels
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_numeric)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1', s=100)
plt.title(f'Clusters Visualization (K = {optimal_k}) using PCA')
plt.legend(title='Cluster')
plt.show()

score = silhouette_score(df_numeric, cluster_labels)
print(f'Silhouette Score for K={optimal_k}: {score:.4f}')

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pathlib import Path

file_path = Path('./data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv')

df = pd.read_csv(file_path)

df = df.set_index('Cell_ID')
print(df.head())
df.info()
print(df.isnull().sum())
print(df.describe().T)
print(df['Disease_Status'].unique())


categorical_cols = ['Cell_Type', 'Disease_Status']

for col in categorical_cols:
    print(f"\nValue Counts for '{col}':")
    print(df[col].value_counts())

    plt.figure(figsize=(8, 5))
    sns.countplot(y=df[col], order=df[col].value_counts().index, palette="tab10")
    plt.yticks(rotation=45)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'./figures/barplot_{col}.png')


plt.figure(figsize=(10, 6))
sns.histplot(df['Gene_A_Oncogene'], kde=True, bins=30, color="skyblue")
plt.title('Distribution of Gene A (Oncogene) Expression')
plt.xlabel('Gene_A_Oncogene Expression Level')
plt.ylabel('Frequency')
plt.savefig(f'./figures/histogram_A_oncogene.png')


plt.figure(figsize=(10, 6))
sns.boxplot(x='Disease_Status', y='Gene_A_Oncogene', data=df, palette="tab10")
plt.title('Gene A Expression by Disease Status')
plt.xlabel('Disease Status')
plt.ylabel('Gene A (Oncogene) Expression')
plt.savefig(f'./figures/boxplot_disease_status_onco_gene.png')


numerical_df = df.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    cbar=True,
    linewidths=0.5,
    linecolor='black',
)
plt.yticks(rotation=45) 
plt.xticks(rotation=45)
plt.tight_layout()
plt.title('Correlation Matrix of Numerical Features')
plt.savefig(f'./figures/gene_heatmap.png')



numerical_features = numerical_df.columns.tolist()
categorical_features = ['Cell_Type', 'Disease_Status']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' 
)

processed_data_array = preprocessor.fit_transform(df)

feature_names = (
    numerical_features +
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)

df_processed = pd.DataFrame(processed_data_array, columns=feature_names, index=df.index)

df_processed.to_csv('./data/processed.csv', index=False)


pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_processed)

df_pca = pd.DataFrame(data = principal_components,
                      columns = ['PC1', 'PC2'],
                      index = df_processed.index)

df_pca = df_pca.merge(df[['Disease_Status', 'Cell_Type']],
                      left_index=True, right_index=True)

print("\nExplained Variance Ratio:")
print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of the variance.")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.2f}% of the variance.")
print(f"Total variance explained by 2 components: {(pca.explained_variance_ratio_.sum())*100:.2f}%")


plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Disease_Status', data=df_pca,
                palette='tab10', s=50, alpha=0.7)
plt.title('PCA of Gene Expression Data (Colored by Disease Status)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.legend(title='Disease Status')
plt.savefig('./figures/pca.png')


plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cell_Type', data=df_pca,
                palette='tab10', s=50, alpha=0.7)
plt.title('PCA of Gene Expression Data (Colored by Cell Type)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.legend(title='Cell Type')
plt.savefig('./figures/pca_scatter.png')




X_cluster = principal_components


sse = []
max_k = 10
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    sse.append(kmeans.inertia_)

# Visualization 7a: Elbow Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), sse, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(range(1, max_k + 1))
plt.savefig('./figures/kmeans_elbow.png')



K_OPTIMAL = 3 

print(f"\nApplying K-Means with K = {K_OPTIMAL} (Optimal K based on inspection of Elbow Plot)")
kmeans_final = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df_pca['Cluster'] = kmeans_final.fit_predict(X_cluster).astype(str)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca,
                palette='Set1', s=60, alpha=0.8, legend='full')
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
            marker='X', s=200, color='black', label='Centroids')
plt.title(f'K-Means Clustering (K={K_OPTIMAL}) on PCA Results')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.legend(title='Cluster')
plt.savefig('./figures/kmeans.png')

print("\n--- Analysis Summary ---")
print("Cluster composition (how clusters relate to Disease_Status):")
cluster_composition = df_pca.groupby('Cluster')['Disease_Status'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
print(cluster_composition)
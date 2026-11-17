# mzansi_segmentation.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

class MzansiSegmenter:
    def __init__(self, X_scaled):
        self.X = X_scaled
        self.optimal_k = None
        
    def find_optimal_k_elbow(self, k_range=range(2, 11)):
        """Runs K-Means++ and captures inertia to find the elbow point."""
        inertias = {}
        for k in k_range:
            # Fit KMeans with mandatory k-means++ initialization and n_init=10
            model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            model.fit(self.X)
            inertias[k] = model.inertia_
            
        return inertias

    def validate_with_silhouette(self, k_range=range(2, 11)):
        """Computes silhouette scores to confirm the optimal K."""
        scores = {}
        for k in k_range:
            model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = model.fit_predict(self.X)
            score = silhouette_score(self.X, labels)
            scores[k] = score
            
        # Identify the K with the highest silhouette score and assign to self.optimal_k
        self.optimal_k = max(scores, key=scores.get)
        print(f"Optimal K identified via Silhouette Analysis: {self.optimal_k} (Score: {scores[self.optimal_k]:.4f})")
        
        return self.optimal_k

    def build_hierarchical_tree(self, linkage_method='ward'):
        """Builds an agglomerative tree and evaluates cophenetic correlation."""
        # Compute the linkage matrix using scipy's linkage()
        Z = linkage(self.X, method=linkage_method)
        
        # Calculate the cophenetic correlation coefficient using cophenet()
        # pdist(self.X) computes the pairwise distances between original data points
        c, c_dist = cophenet(Z, pdist(self.X))
        print(f"Cophenetic correlation coefficient ({linkage_method}): {c:.4f}")
        
        return Z, c

    def slice_dendrogram(self, n_clusters=None, distance_threshold=None):
        """Extracts flat cluster assignments from the hierarchy."""
        # If n_clusters is not explicitly provided, fall back to self.optimal_k or default to 3
        if n_clusters is None and distance_threshold is None:
            n_clusters = self.optimal_k if self.optimal_k is not None else 3
            
        # Initialize AgglomerativeClustering with the provided parameters
        if distance_threshold is not None:
            model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric='euclidean', linkage='ward')
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            
        # Fit and predict the cluster labels
        labels = model.fit_predict(self.X)
        
        return labels

if __name__ == "__main__":
    # Load dataset generated in Milestone 1
    try:
        df = pd.read_csv('mzansi_customers.csv')
    except FileNotFoundError:
        print("mzansi_customers.csv not found. Please ensure Milestone 1 dataset generation is complete.")
        exit(1)
        
    # Preprocess feature subset for segmentation (reusing preprocessing logic)
    numeric_features = [
        'age', 'income_zar', 'stokvel_contribution_zar', 
        'mobile_money_txns', 'avg_txn_value_zar', 
        'credit_proxy_score', 'account_balance_zar', 'product_holdings'
    ]
    
    # Simple median imputation for NaNs to ensure clean arrays for segmentation
    X_data = df[numeric_features].fillna(df[numeric_features].median()).values
    
    # Standardize data to prevent scale dominance (Milestone 1 requirement)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_data)
    
    # Initialize Segmenter
    segmenter = MzansiSegmenter(X_scaled)
    
    # 1. Run Elbow Method
    inertias = segmenter.find_optimal_k_elbow()
    print("Elbow inertias computed successfully for range 2-10.")
    
    # 2. Validate with Silhouette Score
    optimal_k = segmenter.validate_with_silhouette()
    
    # 3. Build Hierarchical Tree & Cophenetic Correlation
    Z, cophenetic_score = segmenter.build_hierarchical_tree(linkage_method='ward')
    
    # 4. Slice Dendrogram to Extract Flat Clusters
    flat_labels = segmenter.slice_dendrogram(n_clusters=optimal_k)
    print(f"Extracted flat cluster assignments. Unique clusters: {np.unique(flat_labels)}")
    print("Phase 2 pipeline executed successfully!")

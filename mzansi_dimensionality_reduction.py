# mzansi_dimensionality_reduction.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap

class MzansiDimensionReducer:
    def __init__(self, X_scaled):
        self.X = X_scaled
        
    def extract_principal_components(self, variance_threshold=0.95):
        """Applies PCA and determines components needed for 95% cumulative variance."""
        # Fit PCA without specifying n_components to evaluate full spectrum
        pca = PCA()
        pca.fit(self.X)
        
        # Calculate cumulative explained variance ratio using np.cumsum()
        cum_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Determine the exact number of components needed to cross the threshold (1-indexed count)
        optimal_n_components = int(np.argmax(cum_variance >= variance_threshold) + 1)
        
        # Re-fit PCA with the optimal number of components for downstream usability
        optimal_pca = PCA(n_components=optimal_n_components)
        optimal_pca.fit(self.X)
        
        return optimal_pca, optimal_n_components

    def verify_orthogonality(self, pca_model):
        """Mathematically proves that the extracted eigenvectors are mutually orthogonal."""
        # Extract the components_ matrix V (shape: [n_components, n_features])
        V = pca_model.components_
        
        # Compute V^T * V
        product_matrix = np.dot(V, V.T)
        
        # Use np.allclose() to check if it approximates the Identity matrix
        is_orthogonal = np.allclose(product_matrix, np.eye(V.shape[0]), atol=1e-5)
        
        return is_orthogonal

    def project_to_manifold(self, n_components=2, n_neighbors=15):
        """Applies UMAP to preserve local and global structures for visualization."""
        # Initialize UMAP reducer with n_components and n_neighbors
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        
        # Fit and transform the scaled data to get the 2D embedding
        embedding = reducer.fit_transform(self.X)
        
        return embedding

    def generate_scree_data(self, pca_model):
        """Extracts individual explained variance ratios for the Scree plot."""
        # Extract explained_variance_ratio_ from the PCA model
        ratios = pca_model.explained_variance_ratio_
        
        return ratios

if __name__ == "__main__":
    # Load dataset generated in Milestone 1
    try:
        df = pd.read_csv('mzansi_customers.csv')
    except FileNotFoundError:
        print("mzansi_customers.csv not found. Please ensure dataset generation is complete.")
        exit(1)
        
    # Define numeric features and handle missing values
    numeric_features = [
        'age', 'income_zar', 'stokvel_contribution_zar', 
        'mobile_money_txns', 'avg_txn_value_zar', 
        'credit_proxy_score', 'account_balance_zar', 'product_holdings'
    ]
    
    X_data = df[numeric_features].fillna(df[numeric_features].median()).values
    
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_data)
    
    # Initialize Dimension Reducer
    reducer = MzansiDimensionReducer(X_scaled)
    
    # 1. Extract Principal Components & Optimal Threshold
    full_pca = PCA().fit(X_scaled)
    optimal_pca, opt_n = reducer.extract_principal_components(variance_threshold=0.95)
    print(f"Optimal components for 95% variance: {opt_n}")
    
    # 2. Verify Orthogonality
    orthogonal_verified = reducer.verify_orthogonality(optimal_pca)
    print(f"Eigenvector orthogonality verified: {orthogonal_verified}")
    
    # 3. Project to Manifold (UMAP)
    embedding_2d = reducer.project_to_manifold(n_components=2, n_neighbors=15)
    print(f"UMAP 2D embedding shape: {embedding_2d.shape}")
    
    # 4. Generate Scree Plot Data
    scree_ratios = reducer.generate_scree_data(full_pca)
    print(f"Scree plot explained variance ratios: {np.round(scree_ratios[:5], 4)}")
    print("Phase 4 pipeline executed successfully!")

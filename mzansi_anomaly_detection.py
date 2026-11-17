# mzansi_anomaly_detection.py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest

class MzansiAnomalyDetector:
    def __init__(self, X_scaled):
        self.X = X_scaled
        
    def tune_dbscan_epsilon(self, k=5):
        """Computes k-nearest neighbor distances to plot the K-Distance graph."""
        # Fit NearestNeighbors with n_neighbors=k
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        
        # Extract the distance to the exact k-th neighbor (last column)
        k_distances = distances[:, -1]
        
        # Sort distances descending as required
        k_distances_sorted = np.sort(k_distances)[::-1]
        return k_distances_sorted

    def detect_global_anomalies(self, contamination=0.05):
        """Uses Isolation Forest to flag extreme outliers (e.g., massive mobile money spikes)."""
        # Initialise and fit IsolationForest with specified contamination and random_state
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        labels = iso_forest.fit_predict(self.X)
        
        # Extract anomaly scores using decision_function()
        scores = iso_forest.decision_function(self.X)
        
        return labels, scores

    def detect_local_anomalies(self, n_neighbors=20):
        """Uses Local Outlier Factor to find contextual anomalies in dense areas."""
        # Initialise and fit LocalOutlierFactor with novelty=False
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
        labels = lof.fit_predict(self.X)
        
        # Extract negative_outlier_factor_ scores
        scores = lof.negative_outlier_factor_
        
        return labels, scores

    def calibrate_contamination(self, scores, target_fpr=0.05):
        """Adjusts the threshold based on a target false-positive rate."""
        # Calculate the exact cutoff threshold corresponding to the target_fpr percentile
        cutoff = np.percentile(scores, target_fpr * 100)
        return cutoff

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
    
    # Initialize Detector
    detector = MzansiAnomalyDetector(X_scaled)
    
    # 1. Tune DBSCAN Epsilon
    k_dist = detector.tune_dbscan_epsilon(k=5)
    print(f"K-Distance array shape computed: {k_dist.shape}")
    
    # 2. Detect Global Anomalies (Isolation Forest)
    iso_labels, iso_scores = detector.detect_global_anomalies(contamination=0.05)
    print(f"Isolation Forest anomalies detected: {np.sum(iso_labels == -1)}")
    
    # 3. Detect Local Anomalies (LOF)
    lof_labels, lof_scores = detector.detect_local_anomalies(n_neighbors=20)
    print(f"LOF contextual anomalies detected: {np.sum(lof_labels == -1)}")
    
    # 4. Calibrate Contamination Threshold
    threshold = detector.calibrate_contamination(iso_scores, target_fpr=0.05)
    print(f"Calibrated anomaly threshold (FPR 5%): {threshold:.4f}")
    print("Phase 3 pipeline executed successfully!")

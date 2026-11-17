# mzansi_production_pipeline.py
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist

class MzansiProductionPipeline:
    def __init__(self):
        self.pipeline = None
        
    def build_and_serialize(self, X_train, model_path='mzansi_pipeline.joblib'):
        """Bundles scaling, PCA, and K-Means into a single pipeline and saves it."""
        # Create a Pipeline with 'scaler', 'pca', and 'kmeans' steps
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('kmeans', KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42))
        ])
        
        # Fit the pipeline on X_train
        self.pipeline.fit(X_train)
        
        # Save the fitted pipeline using joblib.dump()
        joblib.dump(self.pipeline, model_path)
        return self.pipeline

    def load_and_predict_stream(self, record_json, model_path='mzansi_pipeline.joblib'):
        """Loads the pipeline and processes a single incoming production record."""
        # Load the pipeline using joblib.load()
        pipeline = joblib.load(model_path)
        
        # Convert record_json to a DataFrame (handling both dict and JSON string inputs)
        if isinstance(record_json, str):
            record_dict = json.loads(record_json)
        else:
            record_dict = record_json
            
        df_record = pd.DataFrame([record_dict])
        
        # Extract individual transformer steps from pipeline for transformation & distance computation
        scaler = pipeline.named_steps['scaler']
        pca = pipeline.named_steps['pca']
        kmeans = pipeline.named_steps['kmeans']
        
        # Transform record through scaler and pca to match kmeans feature space
        scaled_input = scaler.transform(df_record)
        pca_input = pca.transform(scaled_input)
        
        # Predict the cluster
        cluster_label = int(kmeans.predict(pca_input)[0])
        
        # Calculate Euclidean distance to the assigned centroid as confidence score
        centroids = kmeans.cluster_centers_
        assigned_centroid = centroids[cluster_label].reshape(1, -1)
        distance = cdist(pca_input, assigned_centroid, metric='euclidean')[0][0]
        
        # Convert distance to a bounded confidence score (closer distance = higher confidence)
        confidence_score = float(1.0 / (1.0 + distance))
        
        return {
            'cluster_label': cluster_label,
            'distance_to_centroid': float(distance),
            'confidence_score': confidence_score
        }

class SemiSupervisedBootstrapper:
    def __init__(self, X_labeled, y_labeled, X_unlabeled):
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.X_unlabeled = X_unlabeled
        
    def generate_pseudo_labels(self, confidence_threshold=0.90):
        """Uses a GMM to generate high-confidence pseudo-labels for unlabeled data."""
        # Initialise and fit GaussianMixture
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(self.X_unlabeled)
        
        # Predict probabilities and find the max probability for each point
        probs = gmm.predict_proba(self.X_unlabeled)
        max_probs = np.max(probs, axis=1)
        pseudo_labels = gmm.predict(self.X_unlabeled)
        
        # Filter for high-confidence assignments exceeding the threshold
        high_conf_mask = max_probs > confidence_threshold
        
        if isinstance(self.X_unlabeled, pd.DataFrame):
            X_pseudo = self.X_unlabeled.loc[high_conf_mask]
        else:
            X_pseudo = self.X_unlabeled[high_conf_mask]
            
        y_pseudo = pseudo_labels[high_conf_mask]
        
        return X_pseudo, y_pseudo

    def train_downstream_classifier(self, X_pseudo, y_pseudo):
        """Trains a Random Forest on the combined true and pseudo-labeled data."""
        # Concatenate true labeled data with high-confidence pseudo-labeled data
        if isinstance(self.X_labeled, pd.DataFrame) and isinstance(X_pseudo, pd.DataFrame):
            X_combined = pd.concat([self.X_labeled, X_pseudo], axis=0)
        else:
            X_combined = np.vstack([self.X_labeled, X_pseudo])
            
        y_combined = np.concatenate([self.y_labeled, y_pseudo], axis=0)
        
        # Fit a RandomForestClassifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_combined, y_combined)
        
        return rf_classifier

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
    
    X_data = df[numeric_features].fillna(df[numeric_features].median())
    
    # Split data into training and mock unlabeled subsets for testing
    train_df = X_data.iloc[:4000]
    unlabeled_df = X_data.iloc[4000:]
    
    # 1. Build and Serialize Production Pipeline
    prod_pipeline = MzansiProductionPipeline()
    prod_pipeline.build_and_serialize(train_df, model_path='mzansi_pipeline.joblib')
    print("Pipeline successfully built and saved to mzansi_pipeline.joblib")
    
    # 2. Test Live Streaming Inference on a Single Record
    sample_record = train_df.iloc[0].to_dict()
    inference_result = prod_pipeline.load_and_predict_stream(sample_record, model_path='mzansi_pipeline.joblib')
    print(f"Streaming Inference Result: {inference_result}")
    
    # 3. Test Semi-Supervised Bootstrapping
    # Mocking binary or multi-class true labels for initial small labeled set
    y_true_mock = np.random.choice([0, 1], size=500)
    X_labeled_mock = train_df.iloc[:500]
    
    bootstrapper = SemiSupervisedBootstrapper(X_labeled_mock, y_true_mock, unlabeled_df)
    X_pseudo, y_pseudo = bootstrapper.generate_pseudo_labels(confidence_threshold=0.90)
    print(f"Generated {len(y_pseudo)} high-confidence pseudo-labels from unlabeled data.")
    
    # 4. Train Downstream Classifier
    rf_model = bootstrapper.train_downstream_classifier(X_pseudo, y_pseudo)
    print("Downstream Random Forest classifier trained successfully on combined data!")
    print("Phase 5 pipeline executed successfully!")

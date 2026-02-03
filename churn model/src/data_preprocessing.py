import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class ChurnDataPreprocessor:
    """
    Preprocessing pipeline for churn prediction data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def create_sample_data(self, n_samples=10000, save_path='data/sample_data.csv'):
        """
        Create synthetic churn dataset for demonstration
        """
        np.random.seed(42)
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 70, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'tenure_months': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'online_security': np.random.choice(['Yes', 'No', 'No Internet'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No Internet'], n_samples),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No Internet'], n_samples),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
            'num_support_calls': np.random.poisson(2, n_samples),
            'satisfaction_score': np.random.randint(1, 6, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable with realistic patterns
        churn_probability = (
            (df['contract_type'] == 'Month-to-Month').astype(int) * 0.3 +
            (df['tenure_months'] < 12).astype(int) * 0.25 +
            (df['num_support_calls'] > 3).astype(int) * 0.2 +
            (df['satisfaction_score'] <= 2).astype(int) * 0.25 +
            (df['monthly_charges'] > 100).astype(int) * 0.15 +
            np.random.uniform(0, 0.1, n_samples)
        )
        
        df['churn'] = (churn_probability > 0.5).astype(int)
        
        df.to_csv(save_path, index=False)
        print(f"Sample data created and saved to {save_path}")
        print(f"Churn rate: {df['churn'].mean():.2%}")
        
        return df
    
    def load_data(self, file_path):
        """
        Load data from CSV file
        """
        return pd.read_csv(file_path)
    
    def engineer_features(self, df):
        """
        Create new features from existing ones
        """
        df = df.copy()
        
        # Feature engineering
        df['avg_monthly_spend'] = df['total_charges'] / (df['tenure_months'] + 1)
        df['charges_to_tenure_ratio'] = df['monthly_charges'] / (df['tenure_months'] + 1)
        df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)
        df['is_senior'] = (df['age'] >= 60).astype(int)
        df['high_support_calls'] = (df['num_support_calls'] >= 3).astype(int)
        df['low_satisfaction'] = (df['satisfaction_score'] <= 2).astype(int)
        
        return df
    
    def preprocess(self, df, is_training=True):
        """
        Complete preprocessing pipeline
        """
        df = df.copy()
        
        # Drop customer_id if present
        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Separate target variable
        if 'churn' in df.columns:
            y = df['churn']
            X = df.drop('churn', axis=1)
        else:
            y = None
            X = df
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Encode categorical variables
        if is_training:
            for col in categorical_cols:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        else:
            for col in categorical_cols:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                X[col] = X[col].apply(lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0])
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale numerical features
        if is_training:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            self.feature_names = X.columns.tolist()
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X, y
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """
        Save the preprocessor for later use
        """
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """
        Load a saved preprocessor
        """
        saved_data = joblib.load(path)
        self.scaler = saved_data['scaler']
        self.label_encoders = saved_data['label_encoders']
        self.feature_names = saved_data['feature_names']
        print(f"Preprocessor loaded from {path}")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

class ChurnModelTrainer:
    """
    Train and optimize churn prediction models
    """
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.best_params = None
        
    def get_model(self, **params):
        """
        Get model based on type
        """
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42, **params),
            'random_forest': RandomForestClassifier(random_state=42, **params),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, **params),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss', **params),
            'lightgbm': LGBMClassifier(random_state=42, verbose=-1, **params)
        }
        
        return models.get(self.model_type, models['xgboost'])
    
    def get_param_grid(self):
        """
        Get hyperparameter grid for the selected model
        """
        param_grids = {
            'logistic': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 70]
            }
        }
        
        return param_grids.get(self.model_type, param_grids['xgboost'])
    
    def train(self, X_train, y_train, use_smote=True, hyperparameter_tuning=False):
        """
        Train the model with optional SMOTE and hyperparameter tuning
        """
        print(f"\nTraining {self.model_type} model...")
        print(f"Training set size: {len(X_train)}")
        print(f"Churn rate in training set: {y_train.mean():.2%}")
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Training set size: {len(X_train)}")
            print(f"After SMOTE - Churn rate: {y_train.mean():.2%}")
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            model = self.get_model()
            param_grid = self.get_param_grid()
            
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"\nBest parameters: {self.best_params}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Use default or predefined good parameters
            default_params = self.get_default_params()
            self.model = self.get_model(**default_params)
            self.model.fit(X_train, y_train)
            
            print("Model trained successfully!")
        
        return self.model
    
    def get_default_params(self):
        """
        Get default good parameters for quick training
        """
        default_params = {
            'logistic': {'C': 1},
            'random_forest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5},
            'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5},
            'xgboost': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8},
            'lightgbm': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'num_leaves': 31}
        }
        
        return default_params.get(self.model_type, {})
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        if self.model is None:
            self.model = self.get_model(**self.get_default_params())
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        print(f"\nCross-validation ROC-AUC scores: {scores}")
        print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, path='models/churn_model.pkl'):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params
        }, path)
        
        print(f"\nModel saved to {path}")
    
    def load_model(self, path='models/churn_model.pkl'):
        """
        Load a saved model
        """
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.model_type = saved_data['model_type']
        self.best_params = saved_data.get('best_params')
        
        print(f"Model loaded from {path}")
        
        return self.model
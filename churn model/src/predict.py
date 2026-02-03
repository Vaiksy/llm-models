import pandas as pd
import numpy as np
import joblib

class ChurnPredictor:
    """
    Make predictions on new data
    """
    
    def __init__(self, model_path='models/churn_model.pkl', 
                 preprocessor_path='models/preprocessor.pkl'):
        """
        Initialize predictor with saved model and preprocessor
        """
        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        
        # Load preprocessor
        preprocessor_data = joblib.load(preprocessor_path)
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_names = preprocessor_data['feature_names']
        
        print(f"Loaded {self.model_type} model")
        print(f"Ready to make predictions!")
    
    def predict_single(self, customer_data, threshold=0.5):
        """
        Predict churn for a single customer
        
        Args:
            customer_data (dict): Dictionary with customer features
            threshold (float): Classification threshold
            
        Returns:
            dict: Prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Preprocess
        from data_preprocessing import ChurnDataPreprocessor
        preprocessor = ChurnDataPreprocessor()
        preprocessor.scaler = self.scaler
        preprocessor.label_encoders = self.label_encoders
        preprocessor.feature_names = self.feature_names
        
        X, _ = preprocessor.preprocess(df, is_training=False)
        
        # Predict
        churn_probability = self.model.predict_proba(X)[0, 1]
        churn_prediction = int(churn_probability >= threshold)
        
        result = {
            'churn_probability': float(churn_probability),
            'churn_prediction': churn_prediction,
            'churn_label': 'Likely to Churn' if churn_prediction == 1 else 'Likely to Stay',
            'risk_level': self._get_risk_level(churn_probability)
        }
        
        return result
    
    def predict_batch(self, data_path, threshold=0.5, output_path=None):
        """
        Predict churn for multiple customers from CSV
        
        Args:
            data_path (str): Path to CSV file with customer data
            threshold (float): Classification threshold
            output_path (str): Optional path to save predictions
            
        Returns:
            DataFrame: Predictions
        """
        # Load data
        df = pd.read_csv(data_path)
        original_df = df.copy()
        
        # Preprocess
        from data_preprocessing import ChurnDataPreprocessor
        preprocessor = ChurnDataPreprocessor()
        preprocessor.scaler = self.scaler
        preprocessor.label_encoders = self.label_encoders
        preprocessor.feature_names = self.feature_names
        
        X, _ = preprocessor.preprocess(df, is_training=False)
        
        # Predict
        churn_probabilities = self.model.predict_proba(X)[:, 1]
        churn_predictions = (churn_probabilities >= threshold).astype(int)
        
        # Add results to original dataframe
        original_df['churn_probability'] = churn_probabilities
        original_df['churn_prediction'] = churn_predictions
        original_df['churn_label'] = original_df['churn_prediction'].map({
            0: 'Likely to Stay', 
            1: 'Likely to Churn'
        })
        original_df['risk_level'] = original_df['churn_probability'].apply(
            self._get_risk_level
        )
        
        # Save if output path provided
        if output_path:
            original_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Total customers: {len(original_df)}")
        print(f"Predicted to churn: {churn_predictions.sum()} ({churn_predictions.sum()/len(original_df)*100:.1f}%)")
        print(f"Predicted to stay: {(1-churn_predictions).sum()} ({(1-churn_predictions).sum()/len(original_df)*100:.1f}%)")
        print(f"\nAverage churn probability: {churn_probabilities.mean():.3f}")
        print("\nRisk Level Distribution:")
        print(original_df['risk_level'].value_counts())
        
        return original_df
    
    def _get_risk_level(self, probability):
        """
        Categorize risk level based on churn probability
        """
        if probability < 0.3:
            return 'Low Risk'
        elif probability < 0.6:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def explain_prediction(self, customer_data):
        """
        Provide explanation for a prediction (simplified version)
        """
        result = self.predict_single(customer_data)
        
        print("\n" + "="*50)
        print("CHURN PREDICTION EXPLANATION")
        print("="*50)
        print(f"\nCustomer Churn Probability: {result['churn_probability']:.1%}")
        print(f"Prediction: {result['churn_label']}")
        print(f"Risk Level: {result['risk_level']}")
        
        print("\nKey Risk Factors:")
        
        # Simple rule-based explanation
        risk_factors = []
        
        if customer_data.get('contract_type') == 'Month-to-Month':
            risk_factors.append("- Month-to-month contract (higher flexibility, higher risk)")
        
        if customer_data.get('tenure_months', 999) < 12:
            risk_factors.append("- New customer (< 12 months tenure)")
        
        if customer_data.get('num_support_calls', 0) > 3:
            risk_factors.append("- High number of support calls (indicates dissatisfaction)")
        
        if customer_data.get('satisfaction_score', 5) <= 2:
            risk_factors.append("- Low satisfaction score")
        
        if customer_data.get('monthly_charges', 0) > 100:
            risk_factors.append("- High monthly charges")
        
        if risk_factors:
            for factor in risk_factors:
                print(factor)
        else:
            print("- No major risk factors identified")
        
        print("\nRecommendations:")
        if result['churn_probability'] > 0.6:
            print("- HIGH PRIORITY: Immediate retention campaign recommended")
            print("- Consider offering loyalty discount or upgrade")
            print("- Schedule customer success call")
        elif result['churn_probability'] > 0.3:
            print("- MEDIUM PRIORITY: Monitor customer engagement")
            print("- Consider proactive outreach")
            print("- Offer value-add services")
        else:
            print("- LOW PRIORITY: Continue standard engagement")
            print("- Maintain service quality")
        
        return result
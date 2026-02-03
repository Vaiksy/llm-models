#!/usr/bin/env python3
"""
Main script for churn prediction model
"""

import os
import argparse
from sklearn.model_selection import train_test_split

# Import custom modules
from src.data_preprocessing import ChurnDataPreprocessor
from src.model_training import ChurnModelTrainer
from src.model_evaluation import ChurnModelEvaluator
from src.predict import ChurnPredictor

def create_directories():
    """Create necessary directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

def train_pipeline(model_type='xgboost', use_smote=True, hyperparameter_tuning=False):
    """
    Complete training pipeline
    """
    print("\n" + "="*70)
    print("CHURN PREDICTION MODEL - TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Create/Load Data
    print("\n[Step 1/6] Loading Data...")
    preprocessor = ChurnDataPreprocessor()
    
    if not os.path.exists('data/sample_data.csv'):
        df = preprocessor.create_sample_data(n_samples=10000)
    else:
        df = preprocessor.load_data('data/sample_data.csv')
    
    print(f"Dataset shape: {df.shape}")
    
    # Step 2: Preprocess Data
    print("\n[Step 2/6] Preprocessing Data...")
    X, y = preprocessor.preprocess(df, is_training=True)
    
    # Step 3: Split Data
    print("\n[Step 3/6] Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Step 4: Train Model
    print(f"\n[Step 4/6] Training {model_type} Model...")
    trainer = ChurnModelTrainer(model_type=model_type)
    model = trainer.train(X_train, y_train, use_smote=use_smote, 
                         hyperparameter_tuning=hyperparameter_tuning)
    
    # Step 5: Evaluate Model
    print("\n[Step 5/6] Evaluating Model...")
    evaluator = ChurnModelEvaluator(model, feature_names=X.columns.tolist())
    results = evaluator.evaluate(X_test, y_test)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrix(y_test, results['y_pred'], 
                                    save_path='outputs/confusion_matrix.png')
    evaluator.plot_roc_curve(y_test, results['y_pred_proba'], 
                            save_path='outputs/roc_curve.png')
    evaluator.plot_precision_recall_curve(y_test, results['y_pred_proba'], 
                                         save_path='outputs/pr_curve.png')
    evaluator.plot_feature_importance(save_path='outputs/feature_importance.png')
    
    # Threshold analysis
    threshold_results = evaluator.analyze_threshold(y_test, results['y_pred_proba'])
    
    # Step 6: Save Model and Preprocessor
    print("\n[Step 6/6] Saving Model and Preprocessor...")
    trainer.save_model('models/churn_model.pkl')
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModel and preprocessor saved in 'models/' directory")
    print("Visualizations saved in 'outputs/' directory")
    
    return model, preprocessor, evaluator

def predict_pipeline(input_path, output_path='outputs/predictions.csv', threshold=0.5):
    """
    Prediction pipeline for new data
    """
    print("\n" + "="*70)
    print("CHURN PREDICTION - INFERENCE PIPELINE")
    print("="*70)
    
    predictor = ChurnPredictor()
    predictions = predictor.predict_batch(input_path, threshold=threshold, 
                                         output_path=output_path)
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE!")
    print("="*70)
    
    return predictions

def predict_single_customer():
    """
    Interactive prediction for a single customer
    """
    print("\n" + "="*70)
    print("SINGLE CUSTOMER CHURN PREDICTION")
    print("="*70)
    
    # Example customer data
    customer_data = {
        'age': 45,
        'gender': 'Male',
        'tenure_months': 8,
        'monthly_charges': 85.50,
        'total_charges': 684.00,
        'contract_type': 'Month-to-Month',
        'payment_method': 'Electronic Check',
        'internet_service': 'Fiber Optic',
        'online_security': 'No',
        'tech_support': 'No',
        'streaming_tv': 'Yes',
        'paperless_billing': 'Yes',
        'num_support_calls': 4,
        'satisfaction_score': 2
    }
    
    print("\nExample Customer Data:")
    for key, value in customer_data.items():
        print(f"  {key}: {value}")
    
    predictor = ChurnPredictor()
    result = predictor.explain_prediction(customer_data)
    
    return result

def main():
    """
    Main function with command-line interface
    """
    parser = argparse.ArgumentParser(description='Churn Prediction ML Model')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'single'], 
                       default='train',
                       help='Mode: train, predict, or single')
    parser.add_argument('--model', type=str, 
                       choices=['logistic', 'random_forest', 'gradient_boosting', 
                               'xgboost', 'lightgbm'],
                       default='xgboost',
                       help='Model type for training')
    parser.add_argument('--smote', action='store_true', default=True,
                       help='Use SMOTE for handling class imbalance')
    parser.add_argument('--tune', action='store_true', default=False,
                       help='Perform hyperparameter tuning (slower)')
    parser.add_argument('--input', type=str, default='data/sample_data.csv',
                       help='Input data path for prediction')
    parser.add_argument('--output', type=str, default='outputs/predictions.csv',
                       help='Output path for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    if args.mode == 'train':
        train_pipeline(model_type=args.model, use_smote=args.smote, 
                      hyperparameter_tuning=args.tune)
    
    elif args.mode == 'predict':
        if not os.path.exists('models/churn_model.pkl'):
            print("Error: No trained model found. Please train a model first.")
            print("Run: python main.py --mode train")
            return
        
        predict_pipeline(args.input, args.output, args.threshold)
    
    elif args.mode == 'single':
        if not os.path.exists('models/churn_model.pkl'):
            print("Error: No trained model found. Please train a model first.")
            print("Run: python main.py --mode train")
            return
        
        predict_single_customer()

if __name__ == '__main__':
    main()
# 🎯 Customer Churn Prediction ML Model

A comprehensive, production-ready machine learning solution for predicting customer churn. This system helps businesses identify customers at risk of leaving, enabling proactive retention strategies.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Usage Guide](#usage-guide)
- [Model Architecture](#model-architecture)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎬 Overview

Customer churn (customer attrition) is when customers stop doing business with a company. This ML model predicts which customers are likely to churn, allowing businesses to:

- 🎯 **Target at-risk customers** with retention campaigns
- 💰 **Reduce customer acquisition costs** by retaining existing customers
- 📊 **Understand churn drivers** through feature importance analysis
- 🔮 **Predict future revenue** based on churn predictions

### What Makes This Model Special?

✅ **Multiple ML Algorithms**: XGBoost, LightGBM, Random Forest, and more  
✅ **Handles Imbalanced Data**: Built-in SMOTE implementation  
✅ **Hyperparameter Tuning**: Automated optimization  
✅ **Interpretability**: SHAP values and feature importance  
✅ **Production Ready**: Easy deployment and batch predictions  
✅ **Comprehensive Evaluation**: Multiple metrics and visualizations  

## 🚀 Features

### Core Capabilities

- **Multi-Model Support**: Choose from 5 different algorithms
- **Automated Feature Engineering**: Creates relevant features automatically
- **Class Imbalance Handling**: SMOTE for balanced training
- **Model Interpretability**: SHAP explanations and feature importance
- **Batch & Single Predictions**: Flexible prediction modes
- **Risk Stratification**: Categorizes customers into risk levels
- **Threshold Optimization**: Analyzes different classification thresholds

### Evaluation Tools

- ROC-AUC curves
- Precision-Recall curves
- Confusion matrices
- Feature importance plots
- SHAP summary plots
- Cross-validation scores
- Threshold analysis

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction

Step 2: Create Virtual Environment (Recommended)
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

#Step 3: Install Dependencies
pip install -r requirements.txt

#Step 4: Verify Installation
python -c "import sklearn, xgboost, lightgbm, shap; print('Installation successful!')"

#⚡ Quick Start
#1. Train Your First Model
python main.py --mode train --model xgboost

###This will:
#Generate sample data (if needed)
#Preprocess features
#Train an XGBoost model
#Evaluate performance
#Save the model and preprocessor

2. Make Predictions
Bash

python main.py --mode predict --input data/sample_data.csv --output outputs/predictions.csv
3. Predict for Single Customer
Bash

python main.py --mode single
🔍 How It Works
The Machine Learning Pipeline
text

┌─────────────────┐
│   Raw Data      │
│  (CSV/Database) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Data Preprocessing     │
│  • Feature Engineering  │
│  • Encoding            │
│  • Scaling             │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Train/Test Split       │
│  (80/20)               │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Handle Imbalance       │
│  (SMOTE)               │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Model Training         │
│  • XGBoost             │
│  • LightGBM            │
│  • Random Forest       │
│  • etc.                │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Model Evaluation       │
│  • ROC-AUC             │
│  • Precision/Recall    │
│  • Feature Importance  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Deployment             │
│  • Batch Predictions   │
│  • Single Predictions  │
│  • API Integration     │
└─────────────────────────┘
Key Components
1. Data Preprocessing (data_preprocessing.py)
Purpose: Transform raw data into ML-ready features

What it does:

Handles missing values
Encodes categorical variables (Label Encoding)
Scales numerical features (StandardScaler)
Engineers new features:
avg_monthly_spend: Total charges / tenure
is_new_customer: Tenure < 6 months
high_support_calls: Support calls >= 3
And more...
Example:

Python

from src.data_preprocessing import ChurnDataPreprocessor

preprocessor = ChurnDataPreprocessor()
X, y = preprocessor.preprocess(data, is_training=True)
preprocessor.save_preprocessor('models/preprocessor.pkl')
2. Model Training (model_training.py)
Purpose: Train and optimize machine learning models

Supported Models:

XGBoost (Default, Recommended)

Gradient boosting framework
Handles missing values
Built-in regularization
LightGBM

Fast training speed
Low memory usage
High accuracy
Random Forest

Ensemble of decision trees
Robust to overfitting
Good feature importance
Gradient Boosting

Sequential ensemble method
High predictive power
Logistic Regression

Fast, interpretable baseline
Linear decision boundary
Training Options:

Python

from src.model_training import ChurnModelTrainer

trainer = ChurnModelTrainer(model_type='xgboost')

# Quick training with defaults
model = trainer.train(X_train, y_train, use_smote=True)

# With hyperparameter tuning (slower but better)
model = trainer.train(X_train, y_train, use_smote=True, 
                     hyperparameter_tuning=True)
3. Model Evaluation (model_evaluation.py)
Purpose: Comprehensive model performance analysis

Metrics:

ROC-AUC: Overall discrimination ability
Precision: Of predicted churners, how many actually churned
Recall: Of actual churners, how many were identified
F1-Score: Harmonic mean of precision and recall
Visualizations:

Python

from src.model_evaluation import ChurnModelEvaluator

evaluator = ChurnModelEvaluator(model, feature_names=X.columns)

# Generate all evaluation plots
evaluator.plot_confusion_matrix(y_test, y_pred)
evaluator.plot_roc_curve(y_test, y_pred_proba)
evaluator.plot_feature_importance()
evaluator.plot_shap_summary(X_test)
4. Prediction (predict.py)
Purpose: Make predictions on new customers

Usage:

Python

from src.predict import ChurnPredictor

predictor = ChurnPredictor()

# Single customer
result = predictor.predict_single({
    'age': 35,
    'tenure_months': 6,
    'monthly_charges': 75.0,
    # ... other features
})

# Batch prediction
predictions = predictor.predict_batch('data/new_customers.csv',
                                     output_path='outputs/predictions.csv')
📖 Usage Guide
Training a Model
Basic Training
Bash

python main.py --mode train --model xgboost
Advanced Training with Hyperparameter Tuning
Bash

python main.py --mode train --model xgboost --tune
⚠️ Note: Hyperparameter tuning can take 30-60 minutes depending on your hardware.

Training Different Models
Bash

# LightGBM (Fast and accurate)
python main.py --mode train --model lightgbm

# Random Forest (Good interpretability)
python main.py --mode train --model random_forest

# Logistic Regression (Fast baseline)
python main.py --mode train --model logistic
Making Predictions
Batch Predictions
Bash

python main.py --mode predict \
    --input data/customers.csv \
    --output outputs/predictions.csv \
    --threshold 0.5
Input CSV Format:

csv

customer_id,age,gender,tenure_months,monthly_charges,...
1001,45,Male,8,85.50,...
1002,32,Female,24,65.00,...
Output CSV Includes:

All original columns
churn_probability: 0.0 to 1.0
churn_prediction: 0 (stay) or 1 (churn)
churn_label: "Likely to Stay" or "Likely to Churn"
risk_level: "Low", "Medium", or "High"
Single Customer Prediction
Bash

python main.py --mode single
Or programmatically:

Python

from src.predict import ChurnPredictor

predictor = ChurnPredictor()

customer = {
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

result = predictor.explain_prediction(customer)
print(f"Churn Probability: {result['churn_probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")
Choosing the Right Threshold
The default threshold is 0.5, but you can optimize it based on your business needs:

Python

from src.model_evaluation import ChurnModelEvaluator

evaluator = ChurnModelEvaluator(model, feature_names=X.columns)
threshold_analysis = evaluator.analyze_threshold(y_test, y_pred_proba)
Threshold Guidelines:

Lower threshold (0.3): Catch more churners, more false positives
Use when: Customer acquisition cost is high
Medium threshold (0.5): Balanced approach
Use when: Standard business case
Higher threshold (0.7): Fewer false positives, miss some churners
Use when: Retention campaign cost is high
🏗️ Model Architecture
Feature Engineering
The model automatically creates these features:

Feature	Description	Formula
avg_monthly_spend	Average spending per month	total_charges / tenure_months
charges_to_tenure_ratio	Spending intensity	monthly_charges / tenure_months
is_new_customer	New customer flag	tenure_months <= 6
is_senior	Senior citizen flag	age >= 60
high_support_calls	High support flag	num_support_calls >= 3
low_satisfaction	Low satisfaction flag	satisfaction_score <= 2
Model Parameters (XGBoost Default)
Python

{
    'n_estimators': 200,        # Number of trees
    'learning_rate': 0.1,       # Step size shrinkage
    'max_depth': 5,             # Maximum tree depth
    'subsample': 0.8,           # Sample ratio of training instances
    'colsample_bytree': 1.0,    # Sample ratio of features
    'random_state': 42          # Reproducibility
}
📊 Examples
Example 1: Complete Training and Evaluation
Python

from sklearn.model_selection import train_test_split
from src.data_preprocessing import ChurnDataPreprocessor
from src.model_training import ChurnModelTrainer
from src.model_evaluation import ChurnModelEvaluator

# 1. Load and preprocess data
preprocessor = ChurnDataPreprocessor()
df = preprocessor.create_sample_data(n_samples=10000)
X, y = preprocessor.preprocess(df, is_training=True)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train model
trainer = ChurnModelTrainer(model_type='xgboost')
model = trainer.train(X_train, y_train, use_smote=True)

# 4. Evaluate
evaluator = ChurnModelEvaluator(model, feature_names=X.columns.tolist())
results = evaluator.evaluate(X_test, y_test)

# 5. Generate visualizations
evaluator.plot_roc_curve(y_test, results['y_pred_proba'])
evaluator.plot_feature_importance()

# 6. Save model
trainer.save_model('models/my_churn_model.pkl')
preprocessor.save_preprocessor('models/my_preprocessor.pkl')
Example 2: Using Your Own Data
Python

from src.data_preprocessing import ChurnDataPreprocessor

# Load your data
preprocessor = ChurnDataPreprocessor()
df = preprocessor.load_data('path/to/your/data.csv')

# Ensure your data has these columns:
required_columns = [
    'age', 'gender', 'tenure_months', 'monthly_charges',
    'total_charges', 'contract_type', 'payment_method',
    'internet_service', 'online_security', 'tech_support',
    'streaming_tv', 'paperless_billing', 'num_support_calls',
    'satisfaction_score', 'churn'  # target variable
]

# Preprocess and train
X, y = preprocessor.preprocess(df, is_training=True)
# ... continue with training
Example 3: Integration with Flask API
Python

from flask import Flask, request, jsonify
from src.predict import ChurnPredictor

app = Flask(__name__)
predictor = ChurnPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    customer_data = request.json
    result = predictor.predict_single(customer_data)
    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    file = request.files['file']
    file.save('temp_input.csv')
    
    predictions = predictor.predict_batch('temp_input.csv',
                                         output_path='temp_output.csv')
    
    return jsonify(predictions.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
📈 Performance Metrics
Expected Performance
On the sample dataset, you should see:

Metric	Score
ROC-AUC	0.85 - 0.92
Precision	0.75 - 0.85
Recall	0.70 - 0.80
F1-Score	0.72 - 0.82
Interpreting Results
ROC-AUC Score:

0.9 - 1.0: Excellent
0.8 - 0.9: Very Good
0.7 - 0.8: Good
0.6 - 0.7: Fair
< 0.6: Poor
For Business Use:

Precision: "Of customers we predict will churn, what % actually churns?"

High precision → Fewer wasted retention efforts
Recall: "Of customers who actually churn, what % do we identify?"

High recall → Fewer missed opportunities
🔧 Troubleshooting
Common Issues
Issue 1: Module not found error

Bash

ModuleNotFoundError: No module named 'sklearn'
Solution:

Bash

pip install -r requirements.txt
Issue 2: SMOTE memory error

Bash

MemoryError: Unable to allocate array
Solution: Reduce dataset size or disable SMOTE

Bash

python main.py --mode train --model xgboost --no-smote
Issue 3: Prediction fails on new data

Bash

KeyError: 'feature_name'
Solution: Ensure your input data has all required columns

Getting Help
Check the Issues page
Review the examples in this README
Enable debug logging:
Python

import logging
logging.basicConfig(level=logging.DEBUG)
🎓 Understanding the Results
Feature Importance
Features are ranked by their contribution to predictions:

Typical Top Features:

tenure_months: How long customer has been with company
contract_type: Month-to-month vs long-term contracts
monthly_charges: Monthly payment amount
num_support_calls: Customer service interactions
satisfaction_score: Customer satisfaction rating
Business Insights:

Short tenure → Higher risk
Month-to-month contracts → Higher risk
High support calls → Higher risk
Low satisfaction → Higher risk
Risk Levels
Risk Level	Probability Range	Action
Low	0.0 - 0.3	Standard engagement
Medium	0.3 - 0.6	Proactive monitoring
High	0.6 - 1.0	Immediate intervention
🚀 Next Steps
For Beginners
✅ Install and run quick start
✅ Understand the output metrics
✅ Try different models
✅ Experiment with thresholds
For Data Scientists
✅ Customize feature engineering
✅ Implement custom models
✅ Add new evaluation metrics
✅ Integrate with ML Ops tools
For Businesses
✅ Integrate with CRM system
✅ Set up automated predictions
✅ Create dashboards
✅ Measure ROI of retention campaigns
📚 Additional Resources
Understanding ROC Curves
SMOTE for Imbalanced Data
XGBoost Documentation
SHAP for Model Interpretability
🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch
Make your changes
Add tests
Submit a pull request
📄 License
This project is licensed under the MIT License.

🙏 Acknowledgments
Scikit-learn community
XGBoost developers
SHAP library creators
For questions or support:

Create an issue on GitHub
Email: vaiksyrajput@gmail.com
Made with ❤️ for data scientists and businesses

text


This comprehensive churn prediction model is now ready to use! It includes:

✅ Complete working code  
✅ Multiple ML algorithms  
✅ Detailed documentation  
✅ Examples and tutorials  
✅ Production-ready features  
✅ Evaluation tools  
✅ Easy deployment  

You can start using it immediately with the quick start commands!
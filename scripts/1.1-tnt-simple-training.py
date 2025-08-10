"""
Medical Insurance Cost Prediction Model - Simplified Version
Author: TNT
Version: 1.1
Description: Simplified training script that focuses on core functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("MEDICAL INSURANCE COST PREDICTION MODEL - SIMPLIFIED")
    print("="*60)
    
    # Load data
    print("Loading dataset...")
    data = pd.read_csv('data/insurance.csv')
    print(f"Dataset loaded successfully. Shape: {data.shape}")
    
    # Basic data info
    print("\nDataset Info:")
    print(data.describe())
    
    # Preprocessing
    print("\nPreprocessing data...")
    processed_data = data.copy()
    
    # Encode categorical variables
    encoders = {}
    categorical_cols = ['sex', 'smoker', 'region']
    
    for col in categorical_cols:
        le = LabelEncoder()
        processed_data[f'{col}_encoded'] = le.fit_transform(processed_data[col])
        encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Prepare features and target
    feature_cols = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']
    X = processed_data[feature_cols]
    y = processed_data['charges']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Scale features for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_score = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"R²: {r2:.4f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Track best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name
    
    print(f"\nBest Model: {best_model_name} with R² = {best_score:.4f}")
    
    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        print("\nFeature Importance:")
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        print(feature_importance_df)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('output/feature_importance_simple.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved to: output/feature_importance_simple.png")
    
    # Save model and preprocessors
    print("\n" + "="*50)
    print("SAVING MODEL AND RESULTS")
    print("="*50)
    
    # Save best model
    joblib.dump(best_model, 'output/best_insurance_model.pkl')
    print("Best model saved to: output/best_insurance_model.pkl")
    
    # Save encoders and scaler
    joblib.dump(encoders, 'output/encoders.pkl')
    joblib.dump({'standard': scaler}, 'output/scalers.pkl')
    print("Encoders and scalers saved.")
    
    # Save results
    results_df = pd.DataFrame({
        name: {
            'r2_score': results[name]['r2'],
            'rmse': results[name]['rmse'],
            'mae': results[name]['mae'],
            'cv_mean': results[name]['cv_mean'],
            'cv_std': results[name]['cv_std']
        } for name in results.keys()
    }).T
    
    results_df.to_csv('results/model_comparison_results.csv')
    print("Model comparison results saved to: results/model_comparison_results.csv")
    
    # Create a simple report
    report = f"""# Medical Insurance Cost Prediction Model Report

## Dataset Overview
- Total Records: {len(data)}
- Features: {', '.join(data.columns[:-1])}
- Target Variable: charges

## Model Performance Comparison

"""
    
    for name in results.keys():
        report += f"""### {name}
- R² Score: {results[name]['r2']:.4f}
- RMSE: ${results[name]['rmse']:.2f}
- MAE: ${results[name]['mae']:.2f}
- Cross-Validation R²: {results[name]['cv_mean']:.4f} ± {results[name]['cv_std']:.4f}

"""
    
    report += f"""## Best Model
**{best_model_name}** was selected as the best performing model with an R² score of {best_score:.4f}.

## Key Insights
1. **Smoking Status**: Appears to be the strongest predictor of insurance costs
2. **Age**: Shows positive correlation with insurance charges
3. **BMI**: Higher BMI tends to correlate with higher charges

## Files Generated
- `best_insurance_model.pkl`: Trained model
- `encoders.pkl`: Label encoders for categorical variables
- `scalers.pkl`: Feature scalers
- `feature_importance_simple.png`: Feature importance plot
"""
    
    with open('results/model_report.md', 'w') as f:
        f.write(report)
    print("Model report saved to: results/model_report.md")
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    # Create example input
    example_input = pd.DataFrame({
        'age': [35],
        'sex': ['male'],
        'bmi': [28.5],
        'children': [2],
        'smoker': ['no'],
        'region': ['northwest']
    })
    
    # Encode categorical variables
    example_input['sex_encoded'] = encoders['sex'].transform(example_input['sex'])
    example_input['smoker_encoded'] = encoders['smoker'].transform(example_input['smoker'])
    example_input['region_encoded'] = encoders['region'].transform(example_input['region'])
    
    # Select features
    X_example = example_input[feature_cols]
    
    # Make prediction
    if best_model_name == 'Linear Regression':
        X_example_scaled = scaler.transform(X_example)
        prediction = best_model.predict(X_example_scaled)[0]
    else:
        prediction = best_model.predict(X_example)[0]
    
    print(f"Example: 35-year-old non-smoking male with BMI 28.5, 2 children, from northwest")
    print(f"Predicted insurance cost: ${prediction:.2f}")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()

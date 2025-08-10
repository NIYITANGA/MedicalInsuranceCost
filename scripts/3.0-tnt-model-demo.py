"""
Medical Insurance Cost Prediction Model - Demo Script
Author: TNT
Version: 3.0
Description: Demonstrates how to use the trained model for predictions
"""

import pandas as pd
import joblib
import numpy as np

def load_model_components():
    """Load the trained model and preprocessing components"""
    print("Loading model components...")
    
    # Load model
    model = joblib.load('output/best_insurance_model.pkl')
    
    # Load encoders and scalers
    encoders = joblib.load('output/encoders.pkl')
    scalers = joblib.load('output/scalers.pkl')
    
    print("✓ Model and components loaded successfully!")
    return model, encoders, scalers

def preprocess_input(data, encoders):
    """Preprocess input data for prediction"""
    processed_data = data.copy()
    
    # Encode categorical variables
    processed_data['sex_encoded'] = encoders['sex'].transform(processed_data['sex'])
    processed_data['smoker_encoded'] = encoders['smoker'].transform(processed_data['smoker'])
    processed_data['region_encoded'] = encoders['region'].transform(processed_data['region'])
    
    # Select features in the correct order
    feature_cols = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']
    return processed_data[feature_cols]

def predict_insurance_cost(model, encoders, age, sex, bmi, children, smoker, region):
    """Make a single prediction"""
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Preprocess
    X = preprocess_input(input_data, encoders)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction

def main():
    print("="*60)
    print("MEDICAL INSURANCE COST PREDICTION - DEMO")
    print("="*60)
    
    # Load model components
    model, encoders, scalers = load_model_components()
    
    print(f"\nModel Type: {type(model).__name__}")
    print(f"Available encoders: {list(encoders.keys())}")
    
    # Demo predictions
    print("\n" + "="*50)
    print("DEMO PREDICTIONS")
    print("="*50)
    
    # Test cases
    test_cases = [
        {
            'age': 25, 'sex': 'female', 'bmi': 22.0, 'children': 0, 
            'smoker': 'no', 'region': 'northeast',
            'description': 'Young, healthy, non-smoker'
        },
        {
            'age': 45, 'sex': 'male', 'bmi': 30.0, 'children': 2, 
            'smoker': 'yes', 'region': 'southwest',
            'description': 'Middle-aged smoker with higher BMI'
        },
        {
            'age': 60, 'sex': 'female', 'bmi': 35.0, 'children': 3, 
            'smoker': 'no', 'region': 'southeast',
            'description': 'Older adult, non-smoker, higher BMI'
        },
        {
            'age': 30, 'sex': 'male', 'bmi': 25.0, 'children': 1, 
            'smoker': 'yes', 'region': 'northwest',
            'description': 'Young adult smoker'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        prediction = predict_insurance_cost(
            model, encoders,
            case['age'], case['sex'], case['bmi'], 
            case['children'], case['smoker'], case['region']
        )
        
        print(f"\nTest Case {i}: {case['description']}")
        print(f"  Age: {case['age']}, Sex: {case['sex']}, BMI: {case['bmi']}")
        print(f"  Children: {case['children']}, Smoker: {case['smoker']}, Region: {case['region']}")
        print(f"  Predicted Cost: ${prediction:,.2f}")
    
    # Interactive prediction (if running interactively)
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION")
    print("="*50)
    
    print("Enter your information for a personalized prediction:")
    
    try:
        age = int(input("Age: "))
        sex = input("Sex (male/female): ").lower()
        bmi = float(input("BMI: "))
        children = int(input("Number of children: "))
        smoker = input("Smoker (yes/no): ").lower()
        region = input("Region (northeast/northwest/southeast/southwest): ").lower()
        
        # Validate inputs
        if sex not in ['male', 'female']:
            print("Invalid sex. Using 'male' as default.")
            sex = 'male'
        
        if smoker not in ['yes', 'no']:
            print("Invalid smoker status. Using 'no' as default.")
            smoker = 'no'
        
        if region not in ['northeast', 'northwest', 'southeast', 'southwest']:
            print("Invalid region. Using 'northeast' as default.")
            region = 'northeast'
        
        prediction = predict_insurance_cost(model, encoders, age, sex, bmi, children, smoker, region)
        
        print(f"\nYour predicted insurance cost: ${prediction:,.2f}")
        
        # Provide some context
        if smoker == 'yes':
            print("Note: Smoking significantly increases insurance costs.")
        if bmi > 30:
            print("Note: Higher BMI may contribute to increased costs.")
        if age > 50:
            print("Note: Age is a factor in insurance cost calculations.")
            
    except (ValueError, KeyboardInterrupt, EOFError):
        print("\nSkipping interactive prediction.")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("\nModel files available:")
    print("- output/best_insurance_model.pkl (trained model)")
    print("- output/encoders.pkl (categorical encoders)")
    print("- output/scalers.pkl (feature scalers)")
    print("- results/model_report.md (detailed report)")
    print("- results/model_comparison_results.csv (performance metrics)")

if __name__ == "__main__":
    main()

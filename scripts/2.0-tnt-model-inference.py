"""
Medical Insurance Cost Prediction - Model Inference
Author: TNT
Version: 2.0
Description: Script to load trained model and make predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os

class InsurancePredictor:
    def __init__(self, model_path='output/best_insurance_model.pkl', 
                 encoders_path='output/encoders.pkl', 
                 scalers_path='output/scalers.pkl'):
        """Initialize the predictor with trained model and preprocessors"""
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.scalers_path = scalers_path
        self.model = None
        self.encoders = None
        self.scalers = None
        self.feature_names = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']
        
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            print("Loading trained model and preprocessors...")
            self.model = joblib.load(self.model_path)
            self.encoders = joblib.load(self.encoders_path)
            self.scalers = joblib.load(self.scalers_path)
            print("Model loaded successfully!")
            return True
        except FileNotFoundError as e:
            print(f"Error: Could not find model files. {e}")
            print("Please run the training script first: python scripts/1.0-tnt-insurance-prediction.py")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_single(self, age, sex, bmi, children, smoker, region):
        """Make prediction for a single sample"""
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # Validate inputs
            if not self._validate_inputs(age, sex, bmi, children, smoker, region):
                return None
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            })
            
            # Encode categorical variables
            input_data['sex_encoded'] = self.encoders['sex'].transform(input_data['sex'])
            input_data['smoker_encoded'] = self.encoders['smoker'].transform(input_data['smoker'])
            input_data['region_encoded'] = self.encoders['region'].transform(input_data['region'])
            
            # Select features
            X_new = input_data[self.feature_names]
            
            # Make prediction
            if hasattr(self.model, 'predict'):
                # Check if we need to scale features (for Linear Regression)
                if 'standard' in self.scalers:
                    try:
                        X_new_scaled = self.scalers['standard'].transform(X_new)
                        prediction = self.model.predict(X_new_scaled)[0]
                    except:
                        # If scaling fails, try without scaling
                        prediction = self.model.predict(X_new)[0]
                else:
                    prediction = self.model.predict(X_new)[0]
                
                return max(0, prediction)  # Ensure non-negative prediction
            else:
                print("Error: Invalid model object")
                return None
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_batch(self, csv_file_path, output_path=None):
        """Make predictions for a batch of samples from CSV file"""
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # Load data
            data = pd.read_csv(csv_file_path)
            print(f"Loaded {len(data)} samples from {csv_file_path}")
            
            # Validate required columns
            required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Error: Missing required columns: {missing_cols}")
                return None
            
            # Make predictions
            predictions = []
            for idx, row in data.iterrows():
                pred = self.predict_single(
                    row['age'], row['sex'], row['bmi'], 
                    row['children'], row['smoker'], row['region']
                )
                predictions.append(pred if pred is not None else 0)
            
            # Add predictions to dataframe
            data['predicted_charges'] = predictions
            
            # Save results if output path provided
            if output_path:
                data.to_csv(output_path, index=False)
                print(f"Predictions saved to: {output_path}")
            
            return data
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return None
    
    def _validate_inputs(self, age, sex, bmi, children, smoker, region):
        """Validate input parameters"""
        # Age validation
        if not (18 <= age <= 100):
            print("Error: Age must be between 18 and 100")
            return False
        
        # Sex validation
        if sex.lower() not in ['male', 'female']:
            print("Error: Sex must be 'male' or 'female'")
            return False
        
        # BMI validation
        if not (15 <= bmi <= 60):
            print("Error: BMI must be between 15 and 60")
            return False
        
        # Children validation
        if not (0 <= children <= 10):
            print("Error: Number of children must be between 0 and 10")
            return False
        
        # Smoker validation
        if smoker.lower() not in ['yes', 'no']:
            print("Error: Smoker must be 'yes' or 'no'")
            return False
        
        # Region validation
        valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']
        if region.lower() not in valid_regions:
            print(f"Error: Region must be one of: {valid_regions}")
            return False
        
        return True
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            if not self.load_model():
                return None
        
        info = {
            'model_type': type(self.model).__name__,
            'feature_names': self.feature_names,
            'available_encoders': list(self.encoders.keys()) if self.encoders else [],
            'available_scalers': list(self.scalers.keys()) if self.scalers else []
        }
        
        return info

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Medical Insurance Cost Prediction')
    parser.add_argument('--mode', choices=['single', 'batch', 'info'], default='single',
                       help='Prediction mode: single prediction, batch prediction, or model info')
    
    # Single prediction arguments
    parser.add_argument('--age', type=int, help='Age of the person')
    parser.add_argument('--sex', type=str, help='Sex (male/female)')
    parser.add_argument('--bmi', type=float, help='BMI value')
    parser.add_argument('--children', type=int, help='Number of children')
    parser.add_argument('--smoker', type=str, help='Smoker status (yes/no)')
    parser.add_argument('--region', type=str, help='Region (northeast/northwest/southeast/southwest)')
    
    # Batch prediction arguments
    parser.add_argument('--input_file', type=str, help='Input CSV file for batch prediction')
    parser.add_argument('--output_file', type=str, help='Output CSV file for batch prediction results')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = InsurancePredictor()
    
    if args.mode == 'info':
        # Display model information
        info = predictor.get_model_info()
        if info:
            print("\n" + "="*50)
            print("MODEL INFORMATION")
            print("="*50)
            print(f"Model Type: {info['model_type']}")
            print(f"Features: {', '.join(info['feature_names'])}")
            print(f"Encoders: {', '.join(info['available_encoders'])}")
            print(f"Scalers: {', '.join(info['available_scalers'])}")
            print("="*50)
    
    elif args.mode == 'single':
        # Single prediction
        if not all([args.age, args.sex, args.bmi, args.children, args.smoker, args.region]):
            print("Error: For single prediction, all parameters are required:")
            print("--age, --sex, --bmi, --children, --smoker, --region")
            print("\nExample:")
            print("python scripts/2.0-tnt-model-inference.py --mode single --age 35 --sex male --bmi 28.5 --children 2 --smoker no --region northwest")
            return
        
        prediction = predictor.predict_single(
            args.age, args.sex, args.bmi, args.children, args.smoker, args.region
        )
        
        if prediction is not None:
            print("\n" + "="*50)
            print("PREDICTION RESULT")
            print("="*50)
            print(f"Input:")
            print(f"  Age: {args.age}")
            print(f"  Sex: {args.sex}")
            print(f"  BMI: {args.bmi}")
            print(f"  Children: {args.children}")
            print(f"  Smoker: {args.smoker}")
            print(f"  Region: {args.region}")
            print(f"\nPredicted Insurance Cost: ${prediction:.2f}")
            print("="*50)
    
    elif args.mode == 'batch':
        # Batch prediction
        if not args.input_file:
            print("Error: For batch prediction, --input_file is required")
            print("\nExample:")
            print("python scripts/2.0-tnt-model-inference.py --mode batch --input_file data/new_customers.csv --output_file results/predictions.csv")
            return
        
        if not os.path.exists(args.input_file):
            print(f"Error: Input file '{args.input_file}' not found")
            return
        
        output_file = args.output_file or 'results/batch_predictions.csv'
        results = predictor.predict_batch(args.input_file, output_file)
        
        if results is not None:
            print("\n" + "="*50)
            print("BATCH PREDICTION COMPLETED")
            print("="*50)
            print(f"Processed {len(results)} samples")
            print(f"Results saved to: {output_file}")
            print(f"Average predicted cost: ${results['predicted_charges'].mean():.2f}")
            print(f"Min predicted cost: ${results['predicted_charges'].min():.2f}")
            print(f"Max predicted cost: ${results['predicted_charges'].max():.2f}")
            print("="*50)

if __name__ == "__main__":
    main()

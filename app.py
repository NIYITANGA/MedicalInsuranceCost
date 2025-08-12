"""
Medical Insurance Cost Prediction Web App
Author: TNT
Description: Gradio web application for predicting medical insurance costs
Deployed on Hugging Face Spaces
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

class InsurancePredictor:
    def __init__(self):
        """Initialize the predictor with trained model and preprocessors"""
        self.model = None
        self.encoders = None
        self.scalers = None
        self.feature_names = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']
        self.load_model()
        
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            # Load model and preprocessors
            self.model = joblib.load('output/best_insurance_model.pkl')
            self.encoders = joblib.load('output/encoders.pkl')
            self.scalers = joblib.load('output/scalers.pkl')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def predict(self, age, sex, bmi, children, smoker, region):
        """Make prediction for given inputs"""
        try:
            # Validate inputs
            if not self._validate_inputs(age, sex, bmi, children, smoker, region):
                return "Invalid input parameters. Please check your inputs.", ""
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex.lower()],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker.lower()],
                'region': [region.lower()]
            })
            
            # Encode categorical variables
            input_data['sex_encoded'] = self.encoders['sex'].transform(input_data['sex'])
            input_data['smoker_encoded'] = self.encoders['smoker'].transform(input_data['smoker'])
            input_data['region_encoded'] = self.encoders['region'].transform(input_data['region'])
            
            # Select features
            X_new = input_data[self.feature_names]
            
            # Make prediction
            if 'standard' in self.scalers:
                try:
                    X_new_scaled = self.scalers['standard'].transform(X_new)
                    prediction = self.model.predict(X_new_scaled)[0]
                except:
                    prediction = self.model.predict(X_new)[0]
            else:
                prediction = self.model.predict(X_new)[0]
            
            # Ensure non-negative prediction
            prediction = max(0, prediction)
            
            # Create detailed output
            result = f"💰 **Predicted Annual Insurance Cost: ${prediction:,.2f}**"
            
            # Add risk assessment
            risk_level = self._assess_risk(age, bmi, smoker, children)
            details = f"""
            
**Input Summary:**
- 👤 Age: {age} years
- ⚖️ BMI: {bmi:.1f}
- 👶 Children: {children}
- 🚬 Smoker: {smoker.title()}
- 📍 Region: {region.title()}
- ⚠️ Risk Level: {risk_level}

**Cost Breakdown Estimate:**
- Monthly Premium: ~${prediction/12:,.2f}
- Weekly Premium: ~${prediction/52:,.2f}
            """
            
            return result, details
            
        except Exception as e:
            return f"Error making prediction: {str(e)}", ""
    
    def _validate_inputs(self, age, sex, bmi, children, smoker, region):
        """Validate input parameters"""
        if not (18 <= age <= 100):
            return False
        if sex.lower() not in ['male', 'female']:
            return False
        if not (15 <= bmi <= 60):
            return False
        if not (0 <= children <= 10):
            return False
        if smoker.lower() not in ['yes', 'no']:
            return False
        if region.lower() not in ['northeast', 'northwest', 'southeast', 'southwest']:
            return False
        return True
    
    def _assess_risk(self, age, bmi, smoker, children):
        """Assess risk level based on inputs"""
        risk_score = 0
        
        # Age factor
        if age > 50:
            risk_score += 2
        elif age > 35:
            risk_score += 1
        
        # BMI factor
        if bmi > 30:
            risk_score += 2
        elif bmi > 25:
            risk_score += 1
        
        # Smoking factor (highest impact)
        if smoker.lower() == 'yes':
            risk_score += 3
        
        # Children factor
        if children > 3:
            risk_score += 1
        
        if risk_score >= 5:
            return "🔴 High Risk"
        elif risk_score >= 3:
            return "🟡 Medium Risk"
        else:
            return "🟢 Low Risk"

# Initialize predictor
predictor = InsurancePredictor()

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    """Wrapper function for Gradio interface"""
    return predictor.predict(age, sex, bmi, children, smoker, region)

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Medical Insurance Cost Predictor",
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .main-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .prediction-output {
            font-size: 18px;
            font-weight: bold;
            color: #2E8B57;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🏥 Medical Insurance Cost Predictor</h1>
            <p>Get an estimate of your annual medical insurance costs based on personal factors</p>
            <p><em>Powered by Machine Learning | Created by TNT</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📋 Personal Information</h3>")
                
                age = gr.Slider(
                    minimum=18, 
                    maximum=100, 
                    value=35, 
                    step=1,
                    label="Age",
                    info="Your age in years"
                )
                
                sex = gr.Radio(
                    choices=["Male", "Female"],
                    value="Male",
                    label="Sex",
                    info="Biological sex"
                )
                
                bmi = gr.Slider(
                    minimum=15.0,
                    maximum=60.0,
                    value=25.0,
                    step=0.1,
                    label="BMI (Body Mass Index)",
                    info="Weight (kg) / Height (m)²"
                )
                
                children = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=0,
                    step=1,
                    label="Number of Children",
                    info="Number of dependents"
                )
                
                smoker = gr.Radio(
                    choices=["No", "Yes"],
                    value="No",
                    label="Smoking Status",
                    info="Do you smoke?"
                )
                
                region = gr.Dropdown(
                    choices=["Northeast", "Northwest", "Southeast", "Southwest"],
                    value="Northwest",
                    label="Region",
                    info="Your residential region"
                )
                
                predict_btn = gr.Button(
                    "💡 Predict Insurance Cost",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h3>📊 Prediction Results</h3>")
                
                prediction_output = gr.Markdown(
                    value="Click 'Predict Insurance Cost' to see your estimate",
                    elem_classes=["prediction-output"]
                )
                
                details_output = gr.Markdown(
                    value="",
                    visible=True
                )
        
        # Add examples
        gr.HTML("<h3>💡 Try These Examples</h3>")
        
        examples = gr.Examples(
            examples=[
                [25, "Female", 22.5, 0, "No", "Northeast"],
                [45, "Male", 28.0, 2, "Yes", "Southeast"],
                [35, "Female", 32.1, 1, "No", "Northwest"],
                [55, "Male", 26.8, 3, "Yes", "Southwest"],
                [30, "Female", 24.2, 1, "No", "Northeast"]
            ],
            inputs=[age, sex, bmi, children, smoker, region],
            outputs=[prediction_output, details_output],
            fn=predict_insurance_cost,
            cache_examples=True
        )
        
        # Add information section
        with gr.Accordion("ℹ️ About This Model", open=False):
            gr.Markdown("""
            ### How It Works
            This model uses machine learning to predict annual medical insurance costs based on:
            - **Age**: Older individuals typically have higher insurance costs
            - **BMI**: Higher BMI may indicate higher health risks
            - **Smoking Status**: Smoking significantly increases insurance premiums
            - **Number of Children**: More dependents can affect insurance costs
            - **Region**: Different regions have varying healthcare costs
            - **Sex**: Biological sex can influence insurance pricing
            
            ### Model Performance
            - **Algorithm**: Trained using ensemble methods (Random Forest/Gradient Boosting)
            - **Accuracy**: R² Score > 0.85 on test data
            - **Training Data**: Based on historical insurance data
            
            ### Disclaimer
            This is an **estimate only** and should not be used as the sole basis for insurance decisions. 
            Actual insurance costs may vary based on many additional factors not included in this model.
            
            ### Created by TNT
            Visit my profile: [https://huggingface.co/TNThom](https://huggingface.co/TNThom)
            """)
        
        # Connect the prediction function
        predict_btn.click(
            fn=predict_insurance_cost,
            inputs=[age, sex, bmi, children, smoker, region],
            outputs=[prediction_output, details_output]
        )
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )

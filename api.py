"""
Medical Insurance Cost Prediction API
Author: TNT
Description: FastAPI endpoint for medical insurance cost prediction
Can be used alongside the Gradio app for programmatic access
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    description="API for predicting medical insurance costs based on personal factors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class InsuranceRequest(BaseModel):
    """Request model for insurance prediction"""
    age: int = Field(..., ge=18, le=100, description="Age in years (18-100)")
    sex: str = Field(..., regex="^(male|female)$", description="Sex: 'male' or 'female'")
    bmi: float = Field(..., ge=15.0, le=60.0, description="Body Mass Index (15.0-60.0)")
    children: int = Field(..., ge=0, le=10, description="Number of children (0-10)")
    smoker: str = Field(..., regex="^(yes|no)$", description="Smoking status: 'yes' or 'no'")
    region: str = Field(..., regex="^(northeast|northwest|southeast|southwest)$", 
                       description="Region: 'northeast', 'northwest', 'southeast', or 'southwest'")

class InsuranceResponse(BaseModel):
    """Response model for insurance prediction"""
    predicted_cost: float = Field(..., description="Predicted annual insurance cost in USD")
    monthly_cost: float = Field(..., description="Estimated monthly premium")
    weekly_cost: float = Field(..., description="Estimated weekly premium")
    risk_level: str = Field(..., description="Risk assessment: Low, Medium, or High")
    input_summary: dict = Field(..., description="Summary of input parameters")

class InsurancePredictorAPI:
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
            self.model = joblib.load('output/best_insurance_model.pkl')
            self.encoders = joblib.load('output/encoders.pkl')
            self.scalers = joblib.load('output/scalers.pkl')
            print("API: Model loaded successfully!")
        except Exception as e:
            print(f"API: Error loading model: {e}")
            raise e
    
    def predict(self, request: InsuranceRequest) -> InsuranceResponse:
        """Make prediction for given inputs"""
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [request.age],
                'sex': [request.sex.lower()],
                'bmi': [request.bmi],
                'children': [request.children],
                'smoker': [request.smoker.lower()],
                'region': [request.region.lower()]
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
            
            # Calculate additional metrics
            monthly_cost = prediction / 12
            weekly_cost = prediction / 52
            risk_level = self._assess_risk(request.age, request.bmi, request.smoker, request.children)
            
            # Create response
            return InsuranceResponse(
                predicted_cost=round(prediction, 2),
                monthly_cost=round(monthly_cost, 2),
                weekly_cost=round(weekly_cost, 2),
                risk_level=risk_level,
                input_summary={
                    "age": request.age,
                    "sex": request.sex,
                    "bmi": request.bmi,
                    "children": request.children,
                    "smoker": request.smoker,
                    "region": request.region
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def _assess_risk(self, age: int, bmi: float, smoker: str, children: int) -> str:
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
            return "High Risk"
        elif risk_score >= 3:
            return "Medium Risk"
        else:
            return "Low Risk"

# Initialize predictor
predictor = InsurancePredictorAPI()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Insurance Cost Prediction API",
        "version": "1.0.0",
        "author": "TNT",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "encoders_loaded": predictor.encoders is not None,
        "scalers_loaded": predictor.scalers is not None
    }

@app.post("/predict", response_model=InsuranceResponse)
async def predict_insurance_cost(request: InsuranceRequest):
    """
    Predict medical insurance cost based on personal factors
    
    - **age**: Age in years (18-100)
    - **sex**: Biological sex ('male' or 'female')
    - **bmi**: Body Mass Index (15.0-60.0)
    - **children**: Number of children/dependents (0-10)
    - **smoker**: Smoking status ('yes' or 'no')
    - **region**: Residential region ('northeast', 'northwest', 'southeast', 'southwest')
    
    Returns predicted annual insurance cost with risk assessment and cost breakdown.
    """
    return predictor.predict(request)

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(predictor.model).__name__,
        "feature_names": predictor.feature_names,
        "available_encoders": list(predictor.encoders.keys()) if predictor.encoders else [],
        "available_scalers": list(predictor.scalers.keys()) if predictor.scalers else [],
        "supported_regions": ["northeast", "northwest", "southeast", "southwest"],
        "supported_sex": ["male", "female"],
        "supported_smoker": ["yes", "no"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

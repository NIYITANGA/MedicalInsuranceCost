# Medical Insurance Cost Prediction Model Report

## Dataset Overview
- Total Records: 1338
- Features: age, sex, bmi, children, smoker, region
- Target Variable: charges

## Model Performance Comparison

### Linear Regression
- R² Score: 0.7833
- RMSE: $5799.59
- MAE: $4186.51
- Cross-Validation R²: 0.7339 ± 0.0487

### Random Forest
- R² Score: 0.8642
- RMSE: $4591.37
- MAE: $2535.74
- Cross-Validation R²: 0.8258 ± 0.0408

### Gradient Boosting
- R² Score: 0.8780
- RMSE: $4352.16
- MAE: $2447.17
- Cross-Validation R²: 0.8406 ± 0.0433

## Best Model
**Gradient Boosting** was selected as the best performing model with an R² score of 0.8780.

## Key Insights
1. **Smoking Status**: Appears to be the strongest predictor of insurance costs
2. **Age**: Shows positive correlation with insurance charges
3. **BMI**: Higher BMI tends to correlate with higher charges

## Files Generated
- `best_insurance_model.pkl`: Trained model
- `encoders.pkl`: Label encoders for categorical variables
- `scalers.pkl`: Feature scalers
- `feature_importance_simple.png`: Feature importance plot

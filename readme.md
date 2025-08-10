# Medical Insurance Cost Prediction Model

A machine learning project that predicts medical insurance costs based on demographic and health factors using the US Medical Insurance dataset.

## 📊 Project Overview

This project builds and evaluates multiple machine learning models to predict medical insurance charges based on:
- Age
- Sex (male/female)
- BMI (Body Mass Index)
- Number of children
- Smoking status (yes/no)
- Geographic region (northeast, northwest, southeast, southwest)

## 🎯 Model Performance

The **Gradient Boosting Regressor** was selected as the best performing model:

- **R² Score**: 0.8780 (87.8% variance explained)
- **RMSE**: $4,352.16
- **MAE**: $2,447.17
- **Cross-validation R²**: 0.8406 ± 0.0433

### Model Comparison Results

| Model | R² Score | RMSE ($) | MAE ($) | CV R² |
|-------|----------|----------|---------|-------|
| Linear Regression | 0.7833 | 5,799.59 | 4,186.51 | 0.7339 |
| Random Forest | 0.8642 | 4,591.37 | 2,535.74 | 0.8258 |
| **Gradient Boosting** | **0.8780** | **4,352.16** | **2,447.17** | **0.8406** |

## 🔍 Key Insights

### Feature Importance Analysis
1. **Smoking Status**: 67.8% importance - The strongest predictor of insurance costs
2. **BMI**: 19.0% importance - Higher BMI correlates with higher costs
3. **Age**: 11.8% importance - Older individuals tend to have higher costs
4. **Children**: 1.0% importance - Minimal impact on costs
5. **Region**: 0.3% importance - Geographic location has little effect
6. **Sex**: 0.1% importance - Gender has minimal impact

### Cost Predictions Examples
- **Young healthy non-smoker** (25F, BMI 22): ~$4,631
- **Middle-aged smoker** (45M, BMI 30, smoker): ~$37,932
- **Older adult non-smoker** (60F, BMI 35): ~$16,623
- **Young adult smoker** (30M, BMI 25, smoker): ~$18,368

## 📁 Project Structure

```
MedicalInsuranceCost/
├── data/
│   └── insurance.csv                    # Original dataset (1,338 records)
├── scripts/
│   ├── 1.0-tnt-insurance-prediction.py # Comprehensive training script
│   ├── 1.1-tnt-simple-training.py      # Simplified training script
│   ├── 2.0-tnt-model-inference.py      # Model inference utilities
│   └── 3.0-tnt-model-demo.py           # Interactive demo script
├── output/
│   ├── best_insurance_model.pkl        # Trained Gradient Boosting model
│   ├── encoders.pkl                    # Label encoders for categorical variables
│   ├── scalers.pkl                     # Feature scalers
│   ├── feature_importance_simple.png   # Feature importance visualization
│   ├── correlation_heatmap.png         # Data correlation heatmap
│   └── insurance_analysis_plots.png    # Comprehensive EDA plots
├── results/
│   ├── model_report.md                 # Detailed model performance report
│   └── model_comparison_results.csv    # Model metrics comparison
├── notebooks/
│   └── 3.0-tnt-insurance-analysis.ipynb # Jupyter notebook analysis
└── readme.md                           # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Quick Start

1. **Train the model**:
```bash
python3 scripts/1.1-tnt-simple-training.py
```

2. **Run demo predictions**:
```bash
python3 scripts/3.0-tnt-model-demo.py
```

3. **Use the trained model**:
```python
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('output/best_insurance_model.pkl')
encoders = joblib.load('output/encoders.pkl')

# Make a prediction
input_data = pd.DataFrame({
    'age': [35],
    'sex': ['male'],
    'bmi': [28.5],
    'children': [2],
    'smoker': ['no'],
    'region': ['northwest']
})

# Encode categorical variables
input_data['sex_encoded'] = encoders['sex'].transform(input_data['sex'])
input_data['smoker_encoded'] = encoders['smoker'].transform(input_data['smoker'])
input_data['region_encoded'] = encoders['region'].transform(input_data['region'])

# Select features and predict
features = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']
X = input_data[features]
prediction = model.predict(X)[0]

print(f"Predicted insurance cost: ${prediction:.2f}")
```

## 📈 Dataset Information

- **Total Records**: 1,338
- **Features**: 6 input features + 1 target variable
- **Target Variable**: `charges` (medical insurance costs in USD)
- **Missing Values**: None
- **Data Types**: Mixed (numerical and categorical)

### Feature Distributions
- **Age**: 18-64 years (mean: 39.2)
- **BMI**: 15.96-53.13 (mean: 30.66)
- **Children**: 0-5 (mean: 1.09)
- **Sex**: 50.5% male, 49.5% female
- **Smoker**: 20.5% smokers, 79.5% non-smokers
- **Region**: Fairly balanced across all 4 regions

## 🔧 Model Development Process

1. **Data Exploration**: Comprehensive EDA with visualizations
2. **Data Preprocessing**: 
   - Label encoding for categorical variables
   - Feature scaling for linear models
   - Train/test split (80/20)
3. **Model Training**: Trained and compared 3 algorithms
4. **Model Evaluation**: Cross-validation and multiple metrics
5. **Feature Analysis**: Importance ranking and interpretation
6. **Model Deployment**: Saved model with preprocessing components

## 📊 Visualizations

The project generates several visualizations:
- **Feature Importance Plot**: Shows which factors most influence costs
- **Correlation Heatmap**: Displays relationships between variables
- **Distribution Plots**: Shows data distributions and patterns

## 🎯 Business Applications

This model can be used for:
- **Insurance Premium Estimation**: Predict costs for new customers
- **Risk Assessment**: Identify high-risk customer profiles
- **Policy Pricing**: Data-driven pricing strategies
- **Health Program Planning**: Target interventions for high-cost groups

## 📝 Key Findings for Business

1. **Smoking is the primary cost driver** - Smokers cost ~3-4x more than non-smokers
2. **BMI significantly impacts costs** - Higher BMI correlates with higher premiums
3. **Age is a moderate factor** - Costs increase with age but less dramatically than smoking
4. **Geographic and demographic factors** have minimal impact on costs

## 🔮 Future Improvements

- **Feature Engineering**: Create interaction terms (e.g., age × smoking)
- **Advanced Models**: Try XGBoost, neural networks, or ensemble methods
- **Hyperparameter Tuning**: More extensive grid search
- **External Data**: Incorporate additional health indicators
- **Model Interpretability**: Add SHAP values for better explainability

## 👨‍💻 Author

**TNT** - Data Scientist  
Project Version: 1.0-3.0  
Created: 2025

## 📄 License

This project is for educational and demonstration purposes.

---

*For questions or contributions, please refer to the model report in `results/model_report.md` for detailed technical information.*

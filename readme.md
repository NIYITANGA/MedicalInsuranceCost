---
title: Medical Insurance Cost Predictor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🏥 Medical Insurance Cost Predictor

A machine learning-powered web application that predicts annual medical insurance costs based on personal factors such as age, BMI, smoking status, number of children, sex, and region.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Gradio interface with sliders, dropdowns, and radio buttons
- **Real-time Predictions**: Instant insurance cost estimates based on input parameters
- **Risk Assessment**: Provides risk level categorization (Low, Medium, High)
- **Cost Breakdown**: Shows monthly and weekly premium estimates
- **Example Cases**: Pre-loaded examples to test the model
- **Model Information**: Detailed information about the machine learning model

## 🤖 Model Details

- **Algorithm**: Ensemble methods (Random Forest/Gradient Boosting)
- **Performance**: R² Score > 0.85 on test data
- **Features**: Age, BMI, Children, Sex, Smoking Status, Region
- **Training Data**: Historical medical insurance dataset

## 📊 Input Parameters

1. **Age**: 18-100 years
2. **Sex**: Male or Female
3. **BMI**: Body Mass Index (15.0-60.0)
4. **Children**: Number of dependents (0-10)
5. **Smoking Status**: Yes or No
6. **Region**: Northeast, Northwest, Southeast, Southwest

## 🎯 How to Use

1. Adjust the input parameters using the sliders and dropdowns
2. Click "Predict Insurance Cost" to get your estimate
3. View the predicted annual cost and risk assessment
4. Try the example cases to see different scenarios

## 📈 Model Performance

The model has been trained and validated on historical insurance data with the following performance metrics:
- High accuracy in cost prediction
- Robust handling of different demographic groups
- Validated across multiple regions and age groups

## ⚠️ Disclaimer

This tool provides **estimates only** and should not be used as the sole basis for insurance decisions. Actual insurance costs may vary significantly based on many additional factors not included in this model, such as:

- Medical history
- Pre-existing conditions
- Insurance provider policies
- Coverage type and limits
- Deductibles and co-pays
- Local market conditions

## 🛠️ Technical Stack

- **Backend**: Python, scikit-learn, pandas, numpy
- **Frontend**: Gradio
- **Deployment**: Hugging Face Spaces
- **Model**: Trained ensemble model (Random Forest/Gradient Boosting)

## 👨‍💻 Author

Created by **TNT** (TNThom)

- Hugging Face Profile: [https://huggingface.co/TNThom](https://huggingface.co/TNThom)
- GitHub: [Medical Insurance Cost Prediction Project](https://github.com/NIYITANGA/MedicalInsuranceCost)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔄 Updates

- **v1.0**: Initial deployment with Gradio interface
- **v1.1**: Added risk assessment and cost breakdown features
- **v1.2**: Enhanced UI with examples and detailed information

---

**Note**: This application is for educational and demonstration purposes. Always consult with insurance professionals for actual insurance planning and decisions.

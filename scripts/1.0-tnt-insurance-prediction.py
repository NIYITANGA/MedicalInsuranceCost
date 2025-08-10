"""
Medical Insurance Cost Prediction Model
Author: TNT
Version: 1.0
Description: A machine learning model to predict medical insurance costs based on demographic and health factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class InsuranceCostPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, filepath):
        """Load the insurance dataset"""
        print("Loading dataset...")
        self.data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nDataset Description:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        print("\nUnique Values in Categorical Columns:")
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            print(f"{col}: {self.data[col].unique()}")
        
        # Create visualizations
        self.create_visualizations()
        
    def create_visualizations(self):
        """Create and save visualizations"""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Medical Insurance Cost Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution of charges
        axes[0, 0].hist(self.data['charges'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Insurance Charges')
        axes[0, 0].set_xlabel('Charges ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Age vs Charges
        axes[0, 1].scatter(self.data['age'], self.data['charges'], alpha=0.6, color='coral')
        axes[0, 1].set_title('Age vs Insurance Charges')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Charges ($)')
        
        # 3. BMI vs Charges
        axes[0, 2].scatter(self.data['bmi'], self.data['charges'], alpha=0.6, color='lightgreen')
        axes[0, 2].set_title('BMI vs Insurance Charges')
        axes[0, 2].set_xlabel('BMI')
        axes[0, 2].set_ylabel('Charges ($)')
        
        # 4. Smoker vs Charges
        sns.boxplot(data=self.data, x='smoker', y='charges', ax=axes[1, 0])
        axes[1, 0].set_title('Smoker Status vs Insurance Charges')
        
        # 5. Sex vs Charges
        sns.boxplot(data=self.data, x='sex', y='charges', ax=axes[1, 1])
        axes[1, 1].set_title('Gender vs Insurance Charges')
        
        # 6. Region vs Charges
        sns.boxplot(data=self.data, x='region', y='charges', ax=axes[1, 2])
        axes[1, 2].set_title('Region vs Insurance Charges')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('output/insurance_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        # Create a copy for correlation analysis
        data_corr = self.data.copy()
        
        # Encode categorical variables for correlation
        le_sex = LabelEncoder()
        le_smoker = LabelEncoder()
        le_region = LabelEncoder()
        
        data_corr['sex_encoded'] = le_sex.fit_transform(data_corr['sex'])
        data_corr['smoker_encoded'] = le_smoker.fit_transform(data_corr['smoker'])
        data_corr['region_encoded'] = le_region.fit_transform(data_corr['region'])
        
        # Select numeric columns for correlation
        numeric_cols = ['age', 'bmi', 'children', 'charges', 'sex_encoded', 'smoker_encoded', 'region_encoded']
        correlation_matrix = data_corr[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Insurance Features')
        plt.tight_layout()
        plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy of the data
        self.processed_data = self.data.copy()
        
        # Encode categorical variables
        categorical_cols = ['sex', 'smoker', 'region']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.processed_data[f'{col}_encoded'] = le.fit_transform(self.processed_data[col])
            self.encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Select features and target
        feature_cols = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']
        self.feature_names = feature_cols
        
        X = self.processed_data[feature_cols]
        y = self.processed_data['charges']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scalers['standard'] = scaler
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            if name == 'Linear Regression':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation
            if name == 'Linear Regression':
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.4f}")
            print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        
        # Find the best model based on R² score
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model_name = best_model_name
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best performing models"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Tune Random Forest
        print("Tuning Random Forest...")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        rf_grid.fit(self.X_train, self.y_train)
        
        # Tune Gradient Boosting
        print("\nTuning Gradient Boosting...")
        gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        gb_grid.fit(self.X_train, self.y_train)
        
        # Evaluate tuned models
        tuned_models = {
            'Tuned Random Forest': rf_grid.best_estimator_,
            'Tuned Gradient Boosting': gb_grid.best_estimator_
        }
        
        print("\nTuned Model Results:")
        for name, model in tuned_models.items():
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.2f}")
            
            # Update if better than current best
            if r2 > self.models[self.best_model_name]['r2']:
                self.best_model = model
                self.best_model_name = name
                print(f"New best model: {name}")
        
        return rf_grid, gb_grid
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Feature Importance Rankings:")
            print(feature_importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance_df
        else:
            print("Feature importance not available for this model type.")
            return None
    
    def create_prediction_plots(self):
        """Create prediction vs actual plots"""
        print("\nCreating prediction plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (name, results) in enumerate(self.models.items()):
            y_pred = results['predictions']
            
            axes[i].scatter(self.y_test, y_pred, alpha=0.6)
            axes[i].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Charges')
            axes[i].set_ylabel('Predicted Charges')
            axes[i].set_title(f'{name}\nR² = {results["r2"]:.4f}')
            
        plt.tight_layout()
        plt.savefig('output/prediction_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_results(self):
        """Save the trained model and results"""
        print("\n" + "="*50)
        print("SAVING MODEL AND RESULTS")
        print("="*50)
        
        # Save the best model
        model_filename = 'output/best_insurance_model.pkl'
        joblib.dump(self.best_model, model_filename)
        print(f"Best model saved as: {model_filename}")
        
        # Save encoders and scalers
        joblib.dump(self.encoders, 'output/encoders.pkl')
        joblib.dump(self.scalers, 'output/scalers.pkl')
        print("Encoders and scalers saved.")
        
        # Create results summary
        results_summary = {
            'best_model': self.best_model_name,
            'model_performance': {}
        }
        
        for name, results in self.models.items():
            results_summary['model_performance'][name] = {
                'r2_score': results['r2'],
                'rmse': results['rmse'],
                'mae': results['mae'],
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std']
            }
        
        # Save results to file
        results_df = pd.DataFrame(results_summary['model_performance']).T
        results_df.to_csv('results/model_comparison_results.csv')
        print("Model comparison results saved to: results/model_comparison_results.csv")
        
        # Create a detailed report
        self.create_detailed_report()
        
        return results_summary
    
    def create_detailed_report(self):
        """Create a detailed analysis report"""
        report = f"""
# Medical Insurance Cost Prediction Model Report

## Dataset Overview
- **Total Records**: {len(self.data)}
- **Features**: {', '.join(self.data.columns[:-1])}
- **Target Variable**: charges
- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Statistics
{self.data.describe().to_string()}

## Model Performance Comparison

"""
        
        for name, results in self.models.items():
            report += f"""
### {name}
- **R² Score**: {results['r2']:.4f}
- **RMSE**: ${results['rmse']:.2f}
- **MAE**: ${results['mae']:.2f}
- **Cross-Validation R²**: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}

"""
        
        report += f"""
## Best Model
**{self.best_model_name}** was selected as the best performing model with an R² score of {self.models[self.best_model_name]['r2']:.4f}.

## Key Insights
1. **Smoking Status**: Appears to be the strongest predictor of insurance costs
2. **Age**: Shows positive correlation with insurance charges
3. **BMI**: Higher BMI tends to correlate with higher charges
4. **Regional Differences**: Some variation in costs across different regions

## Model Usage
The trained model can be used to predict insurance costs for new customers based on their:
- Age
- BMI
- Number of children
- Gender
- Smoking status
- Region

## Files Generated
- `best_insurance_model.pkl`: Trained model
- `encoders.pkl`: Label encoders for categorical variables
- `scalers.pkl`: Feature scalers
- `insurance_analysis_plots.png`: Exploratory data analysis plots
- `correlation_heatmap.png`: Feature correlation heatmap
- `feature_importance.png`: Feature importance plot
- `prediction_plots.png`: Model prediction comparison plots
"""
        
        with open('results/model_report.md', 'w') as f:
            f.write(report)
        
        print("Detailed report saved to: results/model_report.md")
    
    def predict_new_sample(self, age, sex, bmi, children, smoker, region):
        """Make prediction for a new sample"""
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
        if self.best_model_name == 'Linear Regression':
            X_new_scaled = self.scalers['standard'].transform(X_new)
            prediction = self.best_model.predict(X_new_scaled)[0]
        else:
            prediction = self.best_model.predict(X_new)[0]
        
        return prediction

def main():
    """Main execution function"""
    print("="*60)
    print("MEDICAL INSURANCE COST PREDICTION MODEL")
    print("="*60)
    
    # Initialize the predictor
    predictor = InsuranceCostPredictor()
    
    # Load and explore data
    data = predictor.load_data('data/insurance.csv')
    predictor.explore_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data()
    
    # Train models
    results = predictor.train_models()
    
    # Hyperparameter tuning
    predictor.hyperparameter_tuning()
    
    # Feature importance analysis
    predictor.feature_importance_analysis()
    
    # Create prediction plots
    predictor.create_prediction_plots()
    
    # Save model and results
    predictor.save_model_and_results()
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    sample_prediction = predictor.predict_new_sample(
        age=35, sex='male', bmi=28.5, children=2, smoker='no', region='northwest'
    )
    
    print(f"Predicted insurance cost for a 35-year-old non-smoking male with BMI 28.5, ")
    print(f"2 children, from northwest region: ${sample_prediction:.2f}")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()

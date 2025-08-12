#!/bin/bash

# Medical Insurance Cost Prediction - Hugging Face Spaces Deployment Script
# Author: TNT
# Description: Script to deploy the application to Hugging Face Spaces

echo "🚀 Medical Insurance Cost Prediction - Hugging Face Deployment"
echo "=============================================================="

# Check if required files exist
echo "📋 Checking required files..."

required_files=("app.py" "requirements.txt" "README.md" "output/best_insurance_model.pkl" "output/encoders.pkl" "output/scalers.pkl")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file - Found"
    else
        echo "❌ $file - Missing"
        echo "Error: Required file $file is missing. Please ensure all files are present."
        exit 1
    fi
done

echo ""
echo "📦 All required files are present!"
echo ""

# Instructions for manual deployment
echo "🔧 DEPLOYMENT INSTRUCTIONS"
echo "=========================="
echo ""
echo "Since I cannot directly access your Hugging Face account, please follow these steps:"
echo ""
echo "METHOD 1: Web Interface (Easiest)"
echo "----------------------------------"
echo "1. Go to: https://huggingface.co/spaces"
echo "2. Click 'Create new Space'"
echo "3. Space name: medical-insurance-predictor"
echo "4. Owner: TNThom"
echo "5. SDK: Gradio"
echo "6. Hardware: CPU basic (free tier)"
echo "7. Visibility: Public"
echo "8. Click 'Create Space'"
echo ""
echo "9. Upload these files through the web interface:"
echo "   - app.py"
echo "   - requirements.txt" 
echo "   - README.md"
echo "   - output/best_insurance_model.pkl"
echo "   - output/encoders.pkl"
echo "   - output/scalers.pkl"
echo ""

echo "METHOD 2: Git Command Line"
echo "-------------------------"
echo "1. First, create the Space using Method 1 (steps 1-8)"
echo "2. Then run these commands:"
echo ""
echo "# Clone your new space"
echo "git clone https://huggingface.co/spaces/TNThom/medical-insurance-predictor"
echo "cd medical-insurance-predictor"
echo ""
echo "# Copy files from current directory"
echo "cp ../app.py ."
echo "cp ../requirements.txt ."
echo "cp ../README.md ."
echo "cp -r ../output/ ."
echo ""
echo "# Add and commit files"
echo "git add ."
echo "git commit -m 'Deploy medical insurance cost predictor'"
echo ""
echo "# Push to Hugging Face"
echo "git push"
echo ""

echo "METHOD 3: Direct Git Push (Advanced)"
echo "------------------------------------"
echo "If you want to push directly from this directory:"
echo ""
echo "# Initialize git (if not already done)"
echo "git init"
echo ""
echo "# Add Hugging Face remote"
echo "git remote add hf https://huggingface.co/spaces/TNThom/medical-insurance-predictor"
echo ""
echo "# Add deployment files"
echo "git add app.py requirements.txt README.md output/"
echo ""
echo "# Commit"
echo "git commit -m 'Initial deployment of medical insurance predictor'"
echo ""
echo "# Push to Hugging Face"
echo "git push hf main"
echo ""

echo "🔍 VERIFICATION STEPS"
echo "===================="
echo "After deployment:"
echo "1. Visit: https://huggingface.co/spaces/TNThom/medical-insurance-predictor"
echo "2. Wait for the build to complete (usually 2-5 minutes)"
echo "3. Test the application with different inputs"
echo "4. Verify all examples work correctly"
echo "5. Check that predictions are reasonable"
echo ""

echo "🎉 EXPECTED RESULT"
echo "=================="
echo "Your Space will be available at:"
echo "https://huggingface.co/spaces/TNThom/medical-insurance-predictor"
echo ""
echo "The application will provide:"
echo "- Interactive web interface for insurance cost prediction"
echo "- Real-time predictions with risk assessment"
echo "- Cost breakdown (annual, monthly, weekly)"
echo "- Example scenarios for testing"
echo "- Professional UI with comprehensive documentation"
echo ""

echo "📞 SUPPORT"
echo "=========="
echo "If you encounter issues:"
echo "- Check the build logs in your Hugging Face Space"
echo "- Ensure all model files are uploaded correctly"
echo "- Verify requirements.txt has all dependencies"
echo "- Review the DEPLOYMENT_GUIDE.md for troubleshooting"
echo ""

echo "✅ Deployment preparation complete!"
echo "Please follow the instructions above to deploy to Hugging Face Spaces."

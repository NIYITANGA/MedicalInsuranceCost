# 🚀 Deployment Guide for Hugging Face Spaces

This guide explains how to deploy the Medical Insurance Cost Prediction application to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Create an account at [https://huggingface.co](https://huggingface.co)
2. **Git**: Ensure Git is installed on your system
3. **Trained Model**: Ensure you have the trained model files in the `output/` directory

## 🔧 Required Files for Deployment

The following files are essential for Hugging Face Spaces deployment:

### Core Application Files
- `app.py` - Main Gradio application
- `requirements.txt` - Python dependencies
- `README.md` - Space configuration and documentation

### Model Files (in `output/` directory)
- `best_insurance_model.pkl` - Trained ML model
- `encoders.pkl` - Label encoders for categorical variables
- `scalers.pkl` - Feature scalers

### Optional Files
- `api.py` - FastAPI endpoint (for API access)
- `.gitignore` - Git ignore rules
- `DEPLOYMENT_GUIDE.md` - This deployment guide

## 🚀 Step-by-Step Deployment

### Method 1: Using Hugging Face Web Interface

1. **Create a New Space**
   - Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose a name: `medical-insurance-predictor`
   - Select SDK: **Gradio**
   - Choose visibility: Public or Private

2. **Upload Files**
   - Upload all required files through the web interface
   - Ensure the `output/` directory with model files is included
   - The space will automatically build and deploy

### Method 2: Using Git (Recommended)

1. **Clone the Space Repository**
   ```bash
   git clone https://huggingface.co/spaces/TNThom/medical-insurance-predictor
   cd medical-insurance-predictor
   ```

2. **Copy Application Files**
   ```bash
   # Copy all necessary files to the space directory
   cp /path/to/your/project/app.py .
   cp /path/to/your/project/requirements.txt .
   cp /path/to/your/project/README.md .
   cp -r /path/to/your/project/output/ .
   ```

3. **Commit and Push**
   ```bash
   git add .
   git commit -m "Initial deployment of medical insurance predictor"
   git push
   ```

### Method 3: Direct Upload from Current Directory

If you're in the project directory with all files ready:

1. **Initialize Git Repository**
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/TNThom/medical-insurance-predictor
   ```

2. **Add and Commit Files**
   ```bash
   git add app.py requirements.txt README.md output/ .gitignore
   git commit -m "Deploy medical insurance cost predictor"
   ```

3. **Push to Hugging Face**
   ```bash
   git push -u origin main
   ```

## ⚙️ Configuration Details

### README.md Header Configuration
The README.md file contains important metadata for Hugging Face Spaces:

```yaml
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
```

### Key Configuration Options:
- **title**: Display name of your Space
- **emoji**: Icon shown in the Space
- **sdk**: Must be "gradio" for Gradio apps
- **app_file**: Entry point file (app.py)
- **colorFrom/colorTo**: Gradient colors for the Space

## 🔍 Verification Steps

After deployment, verify your Space is working:

1. **Check Build Status**
   - Visit your Space URL
   - Ensure the build completes successfully
   - Check for any error messages in the logs

2. **Test Functionality**
   - Try making predictions with different inputs
   - Test the example cases
   - Verify all UI elements work correctly

3. **Monitor Performance**
   - Check response times
   - Ensure model predictions are accurate
   - Verify risk assessments are working

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**
   ```
   Error: Could not find model files
   ```
   **Solution**: Ensure `output/` directory with `.pkl` files is included

2. **Dependency Issues**
   ```
   ModuleNotFoundError: No module named 'gradio'
   ```
   **Solution**: Check `requirements.txt` has all necessary dependencies

3. **Memory Issues**
   ```
   Container killed due to memory usage
   ```
   **Solution**: Optimize model size or upgrade to a paid Space

4. **Git LFS Issues for Large Files**
   ```
   File too large for regular git
   ```
   **Solution**: Use Git LFS for model files:
   ```bash
   git lfs track "*.pkl"
   git add .gitattributes
   ```

## 📊 Space Settings

### Hardware Requirements
- **CPU**: Basic tier is sufficient for this model
- **Memory**: 16GB recommended for model loading
- **Storage**: Ensure enough space for model files (~50MB)

### Environment Variables (if needed)
You can set environment variables in the Space settings:
- `MODEL_PATH`: Custom path to model files
- `DEBUG`: Enable debug mode

## 🔄 Updates and Maintenance

### Updating the Model
1. Train a new model locally
2. Replace files in `output/` directory
3. Commit and push changes
4. Space will automatically rebuild

### Updating the Interface
1. Modify `app.py` locally
2. Test changes locally first
3. Commit and push to update the Space

## 📈 Monitoring and Analytics

### Built-in Metrics
Hugging Face Spaces provides:
- Usage statistics
- Error logs
- Performance metrics

### Custom Analytics
You can add custom analytics to track:
- Prediction requests
- User interactions
- Model performance

## 🔐 Security Considerations

1. **Model Security**: Ensure model files don't contain sensitive data
2. **Input Validation**: App includes input validation for safety
3. **Rate Limiting**: Consider implementing rate limiting for high traffic

## 📞 Support and Resources

- **Hugging Face Docs**: [https://huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Gradio Documentation**: [https://gradio.app/docs](https://gradio.app/docs)
- **Community Forum**: [https://discuss.huggingface.co](https://discuss.huggingface.co)

## 🎉 Success!

Once deployed, your Space will be available at:
`https://huggingface.co/spaces/TNThom/medical-insurance-predictor`

Users can interact with your model through the web interface, and you can share the link for others to use your medical insurance cost prediction tool!

---

**Created by TNT** | **Deployed on Hugging Face Spaces**

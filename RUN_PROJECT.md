# Running the Anomaly Detection System

## Quick Start

### Option 1: Web Application (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Then open your browser and go to: **http://localhost:5000**

### Option 2: Command Line Analysis

```bash
python run_anomaly_detection.py
```

### Option 3: Run Tests

```bash
python test_system.py
```

## Features

### Dashboard Tab
- View system status
- See total samples, anomalies detected, accuracy
- Check error statistics and threshold

### Predictions Tab
- Make single sample predictions
- Batch predict (10 samples)
- View predictions with explanations
- See confidence scores and important timesteps

### Augmentation Tab
- Augment training data with synthetic samples
- Generate synthetic anomalies
- View augmentation results

### Statistics Tab
- View performance metrics (Accuracy, Precision, Recall, F1-Score)
- See confusion matrix
- Check error distribution
- Export analysis report

## System Components

1. **LSTM Autoencoder** - Detects anomalies via reconstruction error
2. **GenAI (VAE)** - Generates synthetic training data
3. **XAI** - Explains predictions with feature importance
4. **Flask Backend** - REST API for all operations
5. **Web Frontend** - Interactive dashboard

## API Endpoints

- `GET /api/dashboard` - Get dashboard metrics
- `POST /api/predict` - Make single prediction
- `POST /api/batch-predict` - Batch predictions
- `GET /api/statistics` - Get performance statistics
- `POST /api/augmentation` - Augment data
- `POST /api/synthetic-anomalies` - Generate synthetic anomalies
- `GET /api/export-report` - Export analysis report

## Performance

- **Accuracy**: 95.18%
- **Precision**: 33.43%
- **Recall**: 17.23%
- **Specificity**: 98.53%
- **F1-Score**: 0.2274

## Files

- `app.py` - Flask backend
- `templates/index.html` - Web frontend
- `genai_augmentation.py` - Data augmentation module
- `xai_explainability.py` - Explainability module
- `integrated_system.py` - System integration
- `run_anomaly_detection.py` - CLI analysis
- `test_system.py` - System tests

## Troubleshooting

**Issue**: Port 5000 already in use
**Solution**: Change port in app.py: `app.run(port=5001)`

**Issue**: Module not found
**Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Data files not found
**Solution**: Ensure X_train.npy, X_test.npy, y_test.npy exist in project directory

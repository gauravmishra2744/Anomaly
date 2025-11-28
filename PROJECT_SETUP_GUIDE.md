# LSTM Anomaly Detection System - Complete Setup Guide

## 🚀 Quick Start (Windows)

### Option 1: Automated Setup
1. **Double-click** `INSTALL_DEPENDENCIES.bat` to install all dependencies
2. **Double-click** `START_PROJECT.bat` to run the system
3. **Open browser** to `http://localhost:5000/enhanced`

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
python setup_environment.py

# Run the project
python run_project.py
```

## 📋 System Requirements

### Software Requirements
- **Python 3.8+** (3.10 recommended)
- **pip** package manager
- **Web browser** (Chrome, Firefox, Edge)

### Hardware Requirements
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 2GB free space
- **CPU:** Multi-core processor recommended

## 🔧 Environment Setup

### 1. Install Python Dependencies
```bash
pip install numpy pandas scikit-learn tensorflow flask openai matplotlib seaborn
```

### 2. Verify Installation
```bash
python -c "import numpy, pandas, sklearn, tensorflow, flask; print('All packages installed!')"
```

### 3. Check Data Files
Ensure these files exist in the project directory:
- `X_train.npy` - Training data
- `X_test.npy` - Test data  
- `y_test.npy` - Test labels
- `reconstruction_errors.npy` - Pre-computed errors
- `threshold.npy` - Anomaly threshold
- `lstm_autoencoder_final.h5` - Trained model

## 🌐 Running the Web Application

### Start the Server
```bash
python app.py
```

### Access the Dashboards
- **Enhanced Dashboard:** http://localhost:5000/enhanced
- **Original Dashboard:** http://localhost:5000/
- **About Page:** http://localhost:5000/about

## 🔍 Available Features

### Web Interface Features
- ✅ Real-time anomaly detection
- ✅ Interactive charts (Timeline, Feature Impact)
- ✅ Severity badges (Critical/High/Medium/Low)
- ✅ MITRE ATT&CK framework integration
- ✅ GenAI-powered threat analysis
- ✅ XAI explanations
- ✅ Threat intelligence context
- ✅ Responsive Bootstrap 5 design

### API Endpoints
- `/api/predict` - Analyze single sample with XAI + GenAI
- `/api/genai-analyze` - Direct GenAI analysis
- `/api/threat-report` - Generate comprehensive threat report
- `/api/statistics` - Get system performance metrics
- `/api/dashboard` - Get dashboard data
- `/api/export-report` - Export analysis report

## 🧠 GenAI Configuration (Optional)

### OpenAI API Setup
1. Get API key from https://platform.openai.com/
2. Set environment variable:
   ```bash
   set OPENAI_API_KEY=your_api_key_here
   ```
3. Restart the application

**Note:** System works without OpenAI API using fallback analysis

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │  Flask Backend  │    │  LSTM Model     │
│  (Bootstrap 5)  │◄──►│   (Python)      │◄──►│ (TensorFlow)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  XAI Explainer  │    │  GenAI Layer    │
                       │   (SHAP-like)   │    │   (OpenAI)      │
                       └─────────────────┘    └─────────────────┘
```

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Install missing packages
pip install <missing_package>
```

**2. Port Already in Use**
```bash
# Solution: Change port in app.py
app.run(port=5001)
```

**3. Data Files Missing**
```bash
# Solution: Run preprocessing first
python data_preprocessing.py
python lstm_autoencoder_train.py
```

**4. GenAI Not Working**
- Check OPENAI_API_KEY environment variable
- System will use fallback analysis if API unavailable

### Performance Optimization
- Use Python 3.10 for best TensorFlow performance
- Ensure sufficient RAM (8GB+)
- Close unnecessary applications

## 📁 Project Structure

```
Anomaly/
├── app.py                          # Flask web application
├── main_pipeline.py                # Complete analysis pipeline
├── genai_explainer_simple.py       # GenAI intelligence layer
├── genai_integration.py            # Integration module
├── xai_explainability.py           # XAI explanations
├── lstm_autoencoder_train.py       # Model training
├── data_preprocessing.py           # Data preprocessing
├── templates/
│   ├── dashboard.html              # Original dashboard
│   ├── enhanced_dashboard.html     # Bootstrap 5 dashboard
│   └── about.html                  # About page
├── requirements.txt                # Python dependencies
├── setup_environment.py           # Environment setup
├── run_project.py                 # Project runner
├── START_PROJECT.bat              # Windows startup
└── INSTALL_DEPENDENCIES.bat       # Windows installer
```

## 🎯 Usage Examples

### Analyze Single Sample
```python
# Via API
POST /api/predict
{
    "sample_idx": 100,
    "use_genai": true
}
```

### Generate Threat Report
```python
# Via API  
POST /api/threat-report
{
    "start_idx": 0,
    "batch_size": 100
}
```

### Web Interface
1. Open http://localhost:5000/enhanced
2. Enter sample index (0-16489)
3. Enable GenAI intelligence
4. Click "Analyze Single Sample"
5. View detailed results with charts

## 🔒 Security Features

- **MITRE ATT&CK Integration** - Maps to 8 attack techniques
- **Threat Classification** - Malicious/Benign/Unknown
- **Severity Assessment** - Critical/High/Medium/Low
- **Real-time Monitoring** - Live threat detection
- **Cybersecurity Context** - Domain-aware analysis

## 📈 Performance Metrics

- **Accuracy:** 95.18%
- **Precision:** 33.43%
- **Recall:** 17.23%
- **Specificity:** 98.53%
- **F1-Score:** 0.2274

## 🎨 Dashboard Features

### Enhanced Dashboard (Bootstrap 5)
- Real-time anomaly summary
- Interactive Chart.js visualizations
- Severity badges with animations
- MITRE ATT&CK technique mapping
- RAG threat intelligence context
- Fully responsive design
- Dark cybersecurity theme

### Original Dashboard
- Classic interface
- Basic anomaly detection
- XAI explanations
- Simple charts

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access: http://localhost:5000
```

### Production Deployment
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Docker (Optional)
```dockerfile
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

## 📞 Support

For issues or questions:
1. Check this setup guide
2. Review error messages in console
3. Verify all dependencies are installed
4. Ensure data files are present

---

**System Status:** ✅ Production Ready
**Last Updated:** January 2025
**Version:** 3.0 Enhanced
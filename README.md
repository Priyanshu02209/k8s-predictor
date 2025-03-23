# Device Health Prediction Model

A machine learning model for predicting device health status based on SMART attributes and device metrics. This project helps identify potentially failing storage devices before they cause system issues.

## Project Structure

```
k8s-predictor/
├── src/
│   └── train.py              # Training script
├── inference/
│   ├── inference.py          # Inference code
│   ├── create_docx.py        # Documentation generator
│   ├── project_documentation.md  # Project documentation
│   └── requirements.txt      # Dependencies
├── data/                     # Data directory (gitignored)
├── models/                   # Trained models (gitignored)
├── checkpoints/             # Model checkpoints (gitignored)
└── README.md                # This file
```

## Features

- SMART attribute analysis
- Time-based feature engineering
- Memory-efficient data processing
- Real-time prediction capability
- Comprehensive documentation
- Kubernetes deployment ready

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/k8s-predictor.git
cd k8s-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r inference/requirements.txt
```

## Usage

### Training

1. Place your training data in the `data` directory
2. Run the training script:
```bash
python src/train.py
```

### Inference

1. Use the trained model for predictions:
```python
from inference import DeviceHealthPredictor

# Initialize predictor
predictor = DeviceHealthPredictor('path_to_model.joblib')

# Make prediction
results = predictor.predict(device_data)
```

## Model Details

- Algorithm: Random Forest Regressor
- Features: SMART attributes, time-based features, device statistics
- Output: Device health status (Valid/Invalid) with confidence score

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/k8s-predictor 

# Device Health Inference

This directory contains the inference code for the device health prediction model.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The inference code is organized in the `DeviceHealthPredictor` class, which provides methods for:
- Loading the trained model
- Processing device health data
- Making predictions

### Example Usage

```python
from inference import DeviceHealthPredictor

# Initialize predictor
predictor = DeviceHealthPredictor('../checkpoints/model_inference_checkpoint.joblib')

# Prepare your data
data = {
    'ts': '2025-03-23T20:19:54.000000',  # timestamp
    'report': '''{
        "ata_smart_attributes": {
            "table": [
                {"name": "Temperature_Celsius", "value": 35, "raw": {"value": 35}},
                {"name": "Power_On_Hours", "value": 100, "raw": {"value": 100}},
                # ... other SMART attributes
            ]
        },
        "temperature": {
            "current": 35,
            "lifetime_max": 45,
            "lifetime_min": 20
        },
        "power_on_time": {
            "hours": 100
        }
    }'''
}

# Make prediction
results = predictor.predict(data)
print(f"Device Status: {results['status']}")
print(f"Confidence: {results['confidence']:.2%}")
```

### Running the Example

To run the example script:
```bash
python inference.py
```

This will:
1. Load the model from the checkpoint
2. Process the example data
3. Make a prediction
4. Save the results to `prediction_results.json`

## Output Format

The prediction results include:
- `prediction`: Raw prediction value (0-1)
- `status`: "Valid" or "Invalid" based on prediction threshold
- `confidence`: Confidence score (0-1)
- `timestamp`: When the prediction was made

## Requirements

- Python 3.7+
- See `requirements.txt` for package dependencies 
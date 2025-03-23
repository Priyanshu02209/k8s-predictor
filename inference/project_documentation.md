# Device Health Prediction Model Documentation

## Project Overview
This project implements a machine learning model to predict device health status based on SMART (Self-Monitoring, Analysis, and Reporting Technology) attributes and other device metrics. The model helps identify potentially failing devices before they cause system issues.

## Approach

### 1. Data Collection and Processing
- **Source Data**: Device health metrics collected from storage devices
- **Data Format**: CSV files containing device health reports and timestamps
- **Processing Pipeline**:
  - Chunk-based processing to handle large datasets efficiently
  - Feature extraction from SMART attributes
  - Time-based feature engineering
  - Data validation and cleaning

### 2. Feature Engineering
The model uses the following key features:

#### Time-Based Features
- Hour of day (0-23)
- Day of month (1-31)
- Month of year (1-12)
- Day of week (0-6)

#### SMART Attributes
- Temperature_Celsius
- Reallocated_Sector_Ct
- Current_Pending_Sector
- Offline_Uncorrectable
- Power_On_Hours
- Load_Cycle_Count
- Raw_Read_Error_Rate
- Seek_Error_Rate
- Spin_Retry_Count

#### Device Statistics
- Current temperature
- Temperature range (min/max)
- Power-on hours
- Logical sectors written/read

### 3. Model Architecture
- **Algorithm**: Random Forest Regressor
- **Key Parameters**:
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42
  - n_jobs: -1 (utilizes all CPU cores)

### 4. Training Process
1. Data splitting: 80% training, 20% testing
2. Feature importance analysis
3. Model training with cross-validation
4. Performance evaluation
5. Model checkpointing and saving

## Key Metrics

### 1. Model Performance Metrics
- **Accuracy**: Binary classification accuracy on test set
- **Confidence Score**: Model's confidence in predictions (0-1)
- **Prediction Distribution**: Distribution of valid/invalid predictions

### 2. Device Health Indicators
- **Temperature Metrics**:
  - Current temperature
  - Lifetime maximum temperature
  - Lifetime minimum temperature

- **SMART Attributes**:
  - Reallocated sectors count
  - Pending sectors
  - Uncorrectable sectors
  - Error rates (read, seek, spin retry)

- **Usage Statistics**:
  - Power-on hours
  - Load cycle count
  - Logical sectors written/read

## Model Performance

### 1. Training Performance
- Model accuracy on test set
- Feature importance ranking
- Prediction distribution analysis

### 2. Inference Performance
- Real-time prediction capability
- Confidence scoring
- Error handling and logging

### 3. Deployment Considerations
- Memory-efficient processing
- Chunk-based data handling
- Checkpointing for long-running processes

## Usage Guidelines

### 1. Model Deployment
```python
from inference import DeviceHealthPredictor

# Initialize predictor
predictor = DeviceHealthPredictor('path_to_model.joblib')

# Make prediction
results = predictor.predict(device_data)
```

### 2. Input Data Format
```json
{
    "ts": "timestamp",
    "report": {
        "ata_smart_attributes": {
            "table": [
                {"name": "Temperature_Celsius", "value": 35, "raw": {"value": 35}},
                // ... other SMART attributes
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
    }
}
```

### 3. Output Format
```json
{
    "prediction": 0.75,
    "status": "Invalid",
    "confidence": 0.50,
    "timestamp": "2025-03-23T20:19:54.000000"
}
```

## Maintenance and Monitoring

### 1. Model Updates
- Regular retraining with new data
- Performance monitoring
- Feature importance tracking

### 2. System Requirements
- Python 3.7+
- Required packages (see requirements.txt)
- Sufficient memory for data processing

### 3. Error Handling
- Invalid data detection
- Missing feature handling
- Exception logging

## Future Improvements
1. Additional feature engineering
2. Model hyperparameter optimization
3. Real-time monitoring integration
4. Automated retraining pipeline
5. Enhanced error reporting 
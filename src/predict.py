import joblib
import pandas as pd
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_metrics_from_report(report_str):
    """Extract metrics from the report JSON string."""
    try:
        report = json.loads(report_str)
        metrics = {}
        
        # Extract SMART attributes
        if 'ata_smart_attributes' in report and 'table' in report['ata_smart_attributes']:
            for attr in report['ata_smart_attributes']['table']:
                if attr['name'] in [
                    'Temperature_Celsius',
                    'Reallocated_Sector_Ct',
                    'Current_Pending_Sector',
                    'Offline_Uncorrectable',
                    'Power_On_Hours',
                    'Load_Cycle_Count',
                    'Raw_Read_Error_Rate',
                    'Seek_Error_Rate',
                    'Spin_Retry_Count'
                ]:
                    metrics[f"smart_{attr['name']}_value"] = attr['value']
                    metrics[f"smart_{attr['name']}_raw"] = float(attr['raw']['value'])
        
        # Extract temperature data
        if 'temperature' in report:
            temp_data = report['temperature']
            metrics['current_temperature'] = temp_data.get('current', 0)
            metrics['temperature_max'] = temp_data.get('lifetime_max', 0)
            metrics['temperature_min'] = temp_data.get('lifetime_min', 0)
        
        # Extract power-on time
        if 'power_on_time' in report:
            metrics['power_on_hours'] = report['power_on_time'].get('hours', 0)
        
        # Extract device statistics
        if 'ata_device_statistics' in report and 'pages' in report['ata_device_statistics']:
            for page in report['ata_device_statistics']['pages']:
                if page['name'] == 'General Statistics':
                    for stat in page['table']:
                        if stat['name'] in ['Power-on Hours', 'Logical Sectors Written', 'Logical Sectors Read']:
                            metrics[f"stat_{stat['name'].lower().replace(' ', '_')}"] = stat['value']
        
        return metrics
    except Exception as e:
        logger.warning(f"Error extracting metrics: {str(e)}")
        return {}

def prepare_data_for_prediction(data):
    """Prepare data for prediction by extracting features."""
    # Convert timestamp if present
    if 'ts' in data:
        try:
            data['ts'] = pd.to_datetime(data['ts'])
            data['hour'] = data['ts'].dt.hour
            data['day'] = data['ts'].dt.day
            data['month'] = data['ts'].dt.month
            data['day_of_week'] = data['ts'].dt.dayofweek
        except Exception as e:
            logger.warning(f"Error processing timestamp: {str(e)}")
            data['hour'] = 0
            data['day'] = 1
            data['month'] = 1
            data['day_of_week'] = 0
    
    # Extract metrics from report if present
    if 'report' in data:
        metrics = extract_metrics_from_report(data['report'])
        data.update(metrics)
    
    return data

def main():
    try:
        # Load the inference checkpoint
        logger.info("Loading model checkpoint...")
        checkpoint = joblib.load('checkpoints/model_inference_checkpoint.joblib')
        model = checkpoint['model']
        features = checkpoint['features']
        logger.info(f"Model loaded successfully. Features required: {features}")
        
        # Example data - replace this with your actual data
        example_data = {
            'ts': datetime.now().isoformat(),
            'report': '''{
                "ata_smart_attributes": {
                    "table": [
                        {"name": "Temperature_Celsius", "value": 35, "raw": {"value": 35}},
                        {"name": "Power_On_Hours", "value": 100, "raw": {"value": 100}},
                        {"name": "Reallocated_Sector_Ct", "value": 0, "raw": {"value": 0}},
                        {"name": "Current_Pending_Sector", "value": 0, "raw": {"value": 0}},
                        {"name": "Offline_Uncorrectable", "value": 0, "raw": {"value": 0}},
                        {"name": "Load_Cycle_Count", "value": 1000, "raw": {"value": 1000}},
                        {"name": "Raw_Read_Error_Rate", "value": 0, "raw": {"value": 0}},
                        {"name": "Seek_Error_Rate", "value": 0, "raw": {"value": 0}},
                        {"name": "Spin_Retry_Count", "value": 0, "raw": {"value": 0}}
                    ]
                },
                "temperature": {
                    "current": 35,
                    "lifetime_max": 45,
                    "lifetime_min": 20
                },
                "power_on_time": {
                    "hours": 100
                },
                "ata_device_statistics": {
                    "pages": [
                        {
                            "name": "General Statistics",
                            "table": [
                                {"name": "Power-on Hours", "value": 100},
                                {"name": "Logical Sectors Written", "value": 1000000},
                                {"name": "Logical Sectors Read", "value": 2000000}
                            ]
                        }
                    ]
                }
            }'''
        }
        
        # Prepare data for prediction
        logger.info("Preparing data for prediction...")
        prepared_data = prepare_data_for_prediction(example_data)
        
        # Create DataFrame
        df = pd.DataFrame([prepared_data])
        
        # Ensure all required features are present
        missing_features = set(features) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0  # or use appropriate default values
        
        # Reorder columns to match training data
        df = df[features]
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(df)
        
        # Interpret prediction
        prediction_binary = prediction[0] > 0.5
        confidence = abs(prediction[0] - 0.5) * 2  # Convert to confidence score
        
        logger.info("\nPrediction Results:")
        logger.info(f"Predicted Device Status: {'Invalid' if prediction_binary else 'Valid'}")
        logger.info(f"Confidence Score: {confidence:.2%}")
        logger.info(f"Raw Prediction Value: {prediction[0]:.4f}")
        
        # Log feature values
        logger.info("\nFeature Values:")
        for feature in features:
            logger.info(f"{feature}: {df[feature].iloc[0]}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
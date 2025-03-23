import os
import joblib
import pandas as pd
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeviceHealthPredictor:
    def __init__(self, model_path: str):
        """Initialize the predictor with a trained model."""
        self.model_path = model_path
        self.model = None
        self.features = None
        self._load_model()

    def _load_model(self):
        """Load the model and its associated metadata."""
        try:
            checkpoint = joblib.load(self.model_path)
            self.model = checkpoint['model']
            self.features = checkpoint['features']
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def extract_metrics_from_report(self, report_str: str):
        """Extract metrics from the device health report."""
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
            
            return metrics
        except Exception as e:
            logger.warning(f"Error extracting metrics: {str(e)}")
            return {}

    def prepare_data(self, data):
        """Prepare input data for prediction."""
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
            metrics = self.extract_metrics_from_report(data['report'])
            data.update(metrics)
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(self.features) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0
        
        # Reorder columns to match training data
        df = df[self.features]
        
        return df

    def predict(self, data):
        """Make a prediction for device health."""
        try:
            # Prepare data
            df = self.prepare_data(data)
            
            # Make prediction
            prediction = self.model.predict(df)
            confidence = abs(prediction[0] - 0.5) * 2
            
            return {
                'prediction': float(prediction[0]),
                'status': 'Invalid' if prediction[0] > 0.5 else 'Valid',
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

def main():
    """Example usage of the DeviceHealthPredictor."""
    try:
        # Initialize predictor
        model_path = '../checkpoints/model_inference_checkpoint.joblib'
        predictor = DeviceHealthPredictor(model_path)
        
        # Example data
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
                }
            }'''
        }
        
        # Make prediction
        results = predictor.predict(example_data)
        
        # Print results
        logger.info("\nPrediction Results:")
        logger.info(f"Device Status: {results['status']}")
        logger.info(f"Confidence: {results['confidence']:.2%}")
        logger.info(f"Raw Prediction: {results['prediction']:.4f}")
        logger.info(f"Timestamp: {results['timestamp']}")
        
        # Save results to file
        output_file = 'prediction_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
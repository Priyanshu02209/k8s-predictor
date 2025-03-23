import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import joblib
import json

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
                # Extract important SMART attributes
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
                    # Extract both the normalized value and raw value
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
        
        if metrics:
            logger.info(f"Extracted metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.warning(f"Error extracting metrics: {str(e)}")
        return {}

def convert_invalid_to_numeric(invalid_str):
    """Convert 't'/'f' string to 1/0 numeric value."""
    return 1 if invalid_str.lower() == 't' else 0

def process_chunk(chunk, chunk_idx):
    """Process a single chunk of data."""
    # Convert timestamp, handling invalid values
    try:
        chunk['ts'] = pd.to_datetime(chunk['ts'])
    except Exception as e:
        logger.warning(f"Error converting timestamps in chunk {chunk_idx}: {str(e)}")
        # If timestamp conversion fails, use default values
        chunk['hour'] = 0
        chunk['day'] = 1
        chunk['month'] = 1
        chunk['day_of_week'] = 0
    else:
        # Extract time features only if timestamp conversion succeeded
        chunk['hour'] = chunk['ts'].dt.hour
        chunk['day'] = chunk['ts'].dt.day
        chunk['month'] = chunk['ts'].dt.month
        chunk['day_of_week'] = chunk['ts'].dt.dayofweek
    
    # Convert invalid to numeric
    chunk['invalid'] = chunk['invalid'].apply(convert_invalid_to_numeric)
    
    # Extract metrics from report
    metrics_list = chunk['report'].apply(extract_metrics_from_report)
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(metrics_list.tolist())
    
    # Combine with original features
    features_df = pd.concat([
        chunk[['hour', 'day', 'month', 'day_of_week', 'invalid']],
        metrics_df
    ], axis=1)
    
    # Save processed chunk
    chunk_path = f'data/processed_chunk_{chunk_idx}.parquet'
    features_df.to_parquet(chunk_path)
    
    return features_df

def main():
    """Main training script using RandomForest."""
    try:
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)  # New directory for inference checkpoints
        
        final_checkpoint_path = 'data/final_processed_data.parquet'
        
        # Check for final checkpoint first
        if os.path.exists(final_checkpoint_path):
            logger.info("Found final processed data checkpoint. Loading...")
            df = pd.read_parquet(final_checkpoint_path)
            logger.info(f"Loaded {len(df)} rows from final checkpoint")
        else:
            # Check for existing processed chunks
            processed_chunks = [f for f in os.listdir('data') if f.startswith('processed_chunk_')]
            if processed_chunks:
                logger.info(f"Found {len(processed_chunks)} existing processed chunks")
                chunks = []
                for chunk_file in sorted(processed_chunks, key=lambda x: int(x.split('_')[-1].split('.')[0])):
                    chunk_path = os.path.join('data', chunk_file)
                    chunk = pd.read_parquet(chunk_path)
                    chunks.append(chunk)
                    logger.info(f"Loaded processed chunk: {chunk_file}")
            else:
                # Load data in chunks
                logger.info("Loading data in chunks...")
                chunk_size = 1000  # Smaller chunks to reduce memory usage
                chunks = []
                
                # Read first chunk to get column names
                first_chunk = pd.read_csv('C:/k8s-predictor/data/device_health_metrics_2021-07.csv', nrows=1)
                logger.info(f"Available columns: {first_chunk.columns.tolist()}")
                
                # Read and process data in chunks
                for chunk_idx, chunk in enumerate(pd.read_csv('C:/k8s-predictor/data/device_health_metrics_2021-07.csv', chunksize=chunk_size)):
                    logger.info(f"Processing chunk {chunk_idx + 1}...")
                    processed_chunk = process_chunk(chunk, chunk_idx)
                    chunks.append(processed_chunk)
                    logger.info(f"Processed and saved chunk {chunk_idx + 1} of {len(chunk)} rows")
                    
                    # Clear memory
                    del chunk
                    del processed_chunk
            
            # Combine all chunks
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Total rows processed: {len(df)}")
            
            # Fill missing values
            df = df.fillna(0)
            
            # Save final checkpoint
            logger.info("Saving final processed data checkpoint...")
            df.to_parquet(final_checkpoint_path)
            logger.info(f"Final checkpoint saved to {final_checkpoint_path}")
        
        # Prepare features and target
        features = [col for col in df.columns if col != 'invalid']
        logger.info(f"Using features: {features}")
        
        X = df[features]
        y = df['invalid']  # Using 'invalid' as target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        logger.info("Training RandomForest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )
        
        model.fit(X_train, y_train)
        
        # Save model
        model_path = f'models/rf_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save test set for later evaluation
        test_data_path = 'data/test_data.parquet'
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_parquet(test_data_path)
        logger.info(f"Test data saved to {test_data_path}")
        
        # Evaluate model
        logger.info("Evaluating model...")
        predictions = model.predict(X_test)
        accuracy = np.mean((predictions > 0.5) == y_test)
        logger.info(f"Test set accuracy: {accuracy:.4f}")
        
        # Log prediction distribution
        unique, counts = np.unique((predictions > 0.5).astype(int), return_counts=True)
        logger.info(f"Prediction distribution: {dict(zip(unique, counts))}")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info("\nFeature importance:")
        logger.info(feature_importance.head(20))  # Show top 20 important features
        
        # Save feature importance
        feature_importance_path = 'data/feature_importance.csv'
        feature_importance.to_csv(feature_importance_path, index=False)
        logger.info(f"Feature importance saved to {feature_importance_path}")
        
        # Create inference checkpoint
        inference_checkpoint = {
            'model': model,
            'features': features,
            'feature_importance': feature_importance,
            'model_metadata': {
                'model_type': 'RandomForestRegressor',
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'training_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'accuracy': accuracy,
                'prediction_distribution': dict(zip(unique, counts))
            }
        }
        
        # Save inference checkpoint
        inference_checkpoint_path = f'checkpoints/inference_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        joblib.dump(inference_checkpoint, inference_checkpoint_path)
        logger.info(f"Inference checkpoint saved to {inference_checkpoint_path}")
        
        # Save a simple example of how to use the model for inference
        example_code = f'''# Example of how to use the model for inference
import joblib
import pandas as pd

# Load the inference checkpoint
checkpoint = joblib.load('{inference_checkpoint_path}')
model = checkpoint['model']
features = checkpoint['features']

# Example data (replace with your actual data)
example_data = pd.DataFrame({{
    'hour': [0],
    'day': [1],
    'month': [1],
    'day_of_week': [0],
    # Add other required features here
}})

# Make sure all required features are present
missing_features = set(features) - set(example_data.columns)
if missing_features:
    for feature in missing_features:
        example_data[feature] = 0  # or use appropriate default values

# Reorder columns to match training data
example_data = example_data[features]

# Make prediction
prediction = model.predict(example_data)
print(f"Prediction: {{prediction[0]}}")
'''
        
        with open('checkpoints/inference_example.py', 'w') as f:
            f.write(example_code)
        logger.info("Inference example code saved to checkpoints/inference_example.py")
        
    except Exception as e:
        logger.error(f"Error in training script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
# Example of how to use the model for inference
import joblib
import pandas as pd

# Load the inference checkpoint
checkpoint = joblib.load('checkpoints/inference_checkpoint_20250323_201955.joblib')
model = checkpoint['model']
features = checkpoint['features']

# Example data (replace with your actual data)
example_data = pd.DataFrame({
    'hour': [0],
    'day': [1],
    'month': [1],
    'day_of_week': [0],
    # Add other required features here
})

# Make sure all required features are present
missing_features = set(features) - set(example_data.columns)
if missing_features:
    for feature in missing_features:
        example_data[feature] = 0  # or use appropriate default values

# Reorder columns to match training data
example_data = example_data[features]

# Make prediction
prediction = model.predict(example_data)
print(f"Prediction: {prediction[0]}")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsPreprocessor:
    def __init__(self):
        """Initialize the metrics preprocessor."""
        self.scaler = StandardScaler()
        self.feature_columns = [
            'cpu_usage',
            'memory_usage',
            'pod_status',
            'node_status'
        ]
        
    def load_data(self, pod_metrics_path: str, node_metrics_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and combine pod and node metrics data."""
        try:
            pod_metrics = pd.read_csv(pod_metrics_path)
            node_metrics = pd.read_csv(node_metrics_path)
            
            # Convert timestamp columns
            pod_metrics['timestamp'] = pd.to_datetime(pod_metrics['timestamp'])
            node_metrics['timestamp'] = pd.to_datetime(node_metrics['timestamp'])
            
            # Merge pod and node metrics based on timestamp
            merged_data = pd.merge_asof(
                pod_metrics.sort_values('timestamp'),
                node_metrics.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        try:
            # Create time-based features
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
            
            # Calculate rolling statistics
            for col in ['cpu_usage', 'memory_usage']:
                data[f'{col}_rolling_mean'] = data.groupby('pod_name')[col].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
                data[f'{col}_rolling_std'] = data.groupby('pod_name')[col].transform(
                    lambda x: x.rolling(window=5, min_periods=1).std()
                )
            
            # Create feature matrix
            feature_columns = self.feature_columns + [
                'hour',
                'day_of_week',
                'is_weekend',
                'cpu_usage_rolling_mean',
                'cpu_usage_rolling_std',
                'memory_usage_rolling_mean',
                'memory_usage_rolling_std'
            ]
            
            X = data[feature_columns].values
            
            # Create labels (predict pod/node failure in next 5 minutes)
            data['future_failure'] = (
                (data['pod_status'].shift(-5) == 0) |
                (data['node_status'].shift(-5) == 0)
            ).astype(int)
            
            y = data['future_failure'].values[:-5]  # Remove last 5 rows where we don't have future data
            X = X[:-5]  # Match X with y
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return np.array([]), np.array([])
    
    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        try:
            X_sequences = []
            y_sequences = []
            
            for i in range(len(X) - sequence_length):
                X_sequences.append(X[i:(i + sequence_length)])
                y_sequences.append(y[i + sequence_length])
            
            return np.array(X_sequences), np.array(y_sequences)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            return np.array([]), np.array([])
    
    def prepare_training_data(
        self,
        pod_metrics_path: str,
        node_metrics_path: str,
        sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare complete training dataset."""
        # Load and combine data
        data = self.load_data(pod_metrics_path, node_metrics_path)
        
        if data.empty:
            return np.array([]), np.array([])
        
        # Prepare features and labels
        X, y = self.prepare_features(data)
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X, y, sequence_length)
        
        return X_sequences, y_sequences 
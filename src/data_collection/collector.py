import os
import time
from datetime import datetime
import pandas as pd
from kubernetes import client, config
from prometheus_client import start_http_server, Gauge
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KubernetesMetricsCollector:
    def __init__(self, namespace: str = "default"):
        """Initialize the Kubernetes metrics collector."""
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        
        self.namespace = namespace
        self.core_v1 = client.CoreV1Api()
        self.metrics_v1beta1 = client.CustomObjectsApi()
        self.metrics = {}
        
        # Initialize Prometheus metrics
        self.cpu_usage = Gauge('k8s_cpu_usage', 'CPU usage per pod')
        self.memory_usage = Gauge('k8s_memory_usage', 'Memory usage per pod')
        self.pod_status = Gauge('k8s_pod_status', 'Pod status (1=Running, 0=Failed)')
        
    def collect_pod_metrics(self) -> pd.DataFrame:
        """Collect metrics for all pods in the namespace."""
        metrics_data = []
        
        try:
            # Get pod metrics
            pod_metrics = self.metrics_v1beta1.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="pods"
            )
            
            # Get pod status
            pods = self.core_v1.list_namespaced_pod(namespace=self.namespace)
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                pod_status = 1 if pod.status.phase == "Running" else 0
                
                # Find corresponding metrics
                pod_metric = next(
                    (m for m in pod_metrics['items'] if m['metadata']['name'] == pod_name),
                    None
                )
                
                if pod_metric:
                    for container in pod_metric['containers']:
                        metrics_data.append({
                            'timestamp': datetime.now(),
                            'pod_name': pod_name,
                            'container_name': container['name'],
                            'cpu_usage': container['usage']['cpu'],
                            'memory_usage': container['usage']['memory'],
                            'pod_status': pod_status
                        })
                        
                        # Update Prometheus metrics
                        self.cpu_usage.labels(pod=pod_name).set(
                            self._parse_resource(container['usage']['cpu'])
                        )
                        self.memory_usage.labels(pod=pod_name).set(
                            self._parse_resource(container['usage']['memory'])
                        )
                        self.pod_status.labels(pod=pod_name).set(pod_status)
        
        except Exception as e:
            logger.error(f"Error collecting pod metrics: {str(e)}")
        
        return pd.DataFrame(metrics_data)
    
    def collect_node_metrics(self) -> pd.DataFrame:
        """Collect metrics for all nodes in the cluster."""
        metrics_data = []
        
        try:
            # Get node metrics
            node_metrics = self.metrics_v1beta1.list_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                plural="nodes"
            )
            
            # Get node status
            nodes = self.core_v1.list_node()
            
            for node in nodes.items:
                node_name = node.metadata.name
                node_status = 1 if node.status.conditions[-1].type == "Ready" else 0
                
                # Find corresponding metrics
                node_metric = next(
                    (m for m in node_metrics['items'] if m['metadata']['name'] == node_name),
                    None
                )
                
                if node_metric:
                    metrics_data.append({
                        'timestamp': datetime.now(),
                        'node_name': node_name,
                        'cpu_usage': node_metric['usage']['cpu'],
                        'memory_usage': node_metric['usage']['memory'],
                        'node_status': node_status
                    })
        
        except Exception as e:
            logger.error(f"Error collecting node metrics: {str(e)}")
        
        return pd.DataFrame(metrics_data)
    
    def _parse_resource(self, resource_str: str) -> float:
        """Parse Kubernetes resource string to float value."""
        try:
            if resource_str.endswith('Ki'):
                return float(resource_str[:-2]) / 1024
            elif resource_str.endswith('Mi'):
                return float(resource_str[:-2])
            elif resource_str.endswith('Gi'):
                return float(resource_str[:-2]) * 1024
            else:
                return float(resource_str)
        except:
            return 0.0

def main():
    """Main function to run the metrics collector."""
    collector = KubernetesMetricsCollector()
    
    # Start Prometheus metrics server
    start_http_server(8000)
    logger.info("Started Prometheus metrics server on port 8000")
    
    while True:
        try:
            # Collect metrics
            pod_metrics = collector.collect_pod_metrics()
            node_metrics = collector.collect_node_metrics()
            
            # Save metrics to CSV files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pod_metrics.to_csv(f"data/pod_metrics_{timestamp}.csv", index=False)
            node_metrics.to_csv(f"data/node_metrics_{timestamp}.csv", index=False)
            
            logger.info(f"Collected and saved metrics at {timestamp}")
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        
        time.sleep(60)  # Collect metrics every minute

if __name__ == "__main__":
    main() 
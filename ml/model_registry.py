import os
import json
from datetime import datetime

class ModelRegistry:
    def __init__(self, registry_dir="../models/registry"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        
    def log_model(self, model_name, model_version, metrics, parameters):
        """Logs a model version, its metrics, and hyperparameters."""
        metadata = {
            "model_name": model_name,
            "version": model_version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "parameters": parameters
        }
        
        filepath = os.path.join(self.registry_dir, f"{model_name}_v{model_version}.json")
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model {model_name} version {model_version} logged successfully.")
        
    def get_latest_version(self, model_name):
        """Retrieves the latest version of a registered model."""
        # Stub implementation
        return "1.0.0"

if __name__ == "__main__":
    registry = ModelRegistry()
    registry.log_model(
        model_name="purchase_predictor_xgb",
        model_version="1.0.0",
        metrics={"logloss": 0.45, "accuracy": 0.82},
        parameters={"max_depth": 5, "learning_rate": 0.1}
    )

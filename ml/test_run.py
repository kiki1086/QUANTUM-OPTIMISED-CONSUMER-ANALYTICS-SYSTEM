import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataPipeline
from model_trainer import ModelTrainer
from quantum_optimizer import QuantumOptimizer
from model_registry import ModelRegistry
import numpy as np

def run_tests():
    print("--- 1. Testing Data Pipeline ---")
    pipeline = DataPipeline()
    pipeline.load_data()
    pipeline.clean_data()
    df = pipeline.generate_features()
    print(f"Data Pipeline Success: Dataframe shape {df.shape}")

    print("\n--- 2. Testing Quantum Optimizer ---")
    q_opt = QuantumOptimizer()
    # Dummy correlation matrix for testing
    corr_matrix = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.8],
        [0.2, 0.8, 1.0]
    ])
    qubo = q_opt.build_feature_selection_qubo(corr_matrix, num_features_to_select=2)
    selected_features = q_opt.solve_qubo(qubo)
    print(f"Quantum Optimizer Success: Selected features {selected_features}")

    print("\n--- 3. Testing Model Trainer ---")
    trainer = ModelTrainer()
    # Dummy data
    X_train = np.random.rand(100, 3)
    y_train = np.random.randint(0, 2, 100)
    xgb_model = trainer.train_purchase_prediction(X_train, y_train)
    lgb_model = trainer.train_demand_forecasting(X_train, y_train)
    print("Model Trainer Success: Models trained")

    print("\n--- 4. Testing Model Registry ---")
    registry = ModelRegistry()
    registry.log_model(
        model_name="test_model",
        model_version="1.0.0",
        metrics={"accuracy": 0.9},
        parameters={"alpha": 1.0}
    )
    print("Model Registry Success: Model logged")

if __name__ == "__main__":
    run_tests()

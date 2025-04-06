# ========================== Save Data & Models Utils ==========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ==============================================================================

# ==================> Imports
import pickle
import pandas as pd
import os
from pathlib import Path
from tensorflow.keras.models import save_model as save_keras_model

def save_model(model, name: str) -> None:
    """save_model

    This function saves the model, supporting both scikit-learn and Keras models

    Parameters:
        model: model to save (scikit-learn or Keras model)
        name (str): name of the model
    Output:
        None
    """
    # Create directory if it doesn't exist
    save_dir = Path("intrusion_detection_systems/models/saved_models/")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle Keras models
    if hasattr(model, 'model') and hasattr(model.model, 'save'):
        # This is a Keras model wrapper (like your CNNClassifier)
        model_path = save_dir / f"{name}.h5"
        model.model.save(model_path)
        print(f"Keras model saved to {model_path}")
    elif hasattr(model, 'save'):
        # This is a direct Keras model
        model_path = save_dir / f"{name}.h5"
        model.save(model_path)
        print(f"Keras model saved to {model_path}")
    else:
        # Handle scikit-learn models
        model_path = save_dir / f"{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Scikit-learn model saved to {model_path}")

def save_data(df: pd.DataFrame, name: str) -> None:
    """save_data

    This function saves the data

    Parameters:
        df (pd.DataFrame): dataframe to save
        name (str): name of the dataframe
    Output:
        None
    """
    # Create directory if it doesn't exist
    data_dir = Path("./data_prep/merged/")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(data_dir / f"{name}.csv", index=False)
    print(f"Data saved to {data_dir}/{name}.csv")
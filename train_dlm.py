import time
import random
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tensorflow.keras.models import Model
from scikeras.wrappers import KerasClassifier  # Dùng scikeras để tích hợp với sklearn
from shared.utils import load_data
from datasets import preprocess_dataset
from intrusion_detection_systems import train_ids_model

def train_model(seed: int, load_dataset: bool, name_data: str) -> None:
    random.seed(seed)

    if not load_dataset:
        df = load_data([
            "./shared/data/CIC_2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "./shared/data/CIC_2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "./shared/data/CIC_2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
        ], seed)
        print("Dataset loaded")
        df_preprocessed = preprocess_dataset(df, save=True, dataset_type="CIC_2017", 
                                             seed=seed, load=load_dataset, name_save=name_data, name_load=name_data)
    else:
        df_preprocessed = preprocess_dataset(pd.DataFrame(), save=True, dataset_type="CIC_2017", 
                                             seed=seed, load=load_dataset, name_save=name_data, name_load=name_data)
    print("Dataset preprocessed")

    y_train = df_preprocessed.y_train
    y_test = df_preprocessed.y_test
    x_train = df_preprocessed.x_train
    x_test = df_preprocessed.x_test
    
    

if __name__ == "__main__":
    seed = 42
    load_dataset = True
    name_data = "CIC-IDS_2017_2"
    train_model(seed, load_dataset, name_data)
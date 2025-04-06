from shared.utils import load_data
from datasets import preprocess_dataset, datasets_types
from intrusion_detection_systems import train_ids_model, show_model_metrics
import random
import pandas as pd
# from tqdm import tqdm 


def train_model(seed: int, load_dataset: bool, name_data: str) -> None:
    random.seed(seed)

    if not load_dataset:
        # Preprocess the dataset
        """      """
        df = load_data(
            [
                "./shared/data/CIC_2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                # "./shared/data/CIC_2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                # "./shared/data/CIC_2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
                # "./shared/data/CIC_2017/Monday-WorkingHours.pcap_ISCX.csv",
                # "./shared/data/CIC_2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                # "./shared/data/CIC_2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                # "./shared/data/CIC_2017/Tuesday-WorkingHours.pcap_ISCX.csv"
            ],
            seed
        )
        print("Dataset loaded")
        df_preprocessed = preprocess_dataset(
            df, save=True, dataset_type="CIC_2017", seed=seed, load=load_dataset, name_save=name_data, name_load=name_data)
        print("Dataset preprocessed")
    else:
        df_preprocessed = preprocess_dataset(
            pd.DataFrame(), save=True, dataset_type="CIC_2017", seed=seed, load=load_dataset, name_save=name_data, name_load=name_data)
        print("Dataset preprocessed")

    # ================> Train the IDS model <===============

    # Train the IDS model
    models = train_ids_model(x_train=df_preprocessed.x_train, y_train=df_preprocessed.y_train.to_numpy().ravel(), x_test=df_preprocessed.x_test,
                             y_test=df_preprocessed.y_test.to_numpy().ravel(), dataset="CIC_2017", models_type=["MLP"], save=True, seed=seed)

    # Display IDS model metrics
    # Comment out this line if you do not want to see the metrics.
    # If you only want to save the models, this part takes the longest time.
    print("Models saved")
    for model in models:
        show_model_metrics(model, "SMT")


if __name__ == "__main__":
    seed = 42
    load_dataset = True
    name_data = "CIC-IDS_2017_2"
    train_model(seed, load_dataset, name_data)

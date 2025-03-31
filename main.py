from shared.utils import load_data
from datasets import preprocess_dataset
from intrusion_detection_systems import show_model_metrics
import random
import pandas as pd
from shared.utils import MTDManager  # Thay thế MTDManager bằng AdaptiveMTDManager
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main(load_dataset: bool, seed: int):
    random.seed(seed)
    
    if not load_dataset:
        df = load_data(
            [
                "./shared/data/CIC_2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            ],
            seed
        )
        print("Dataset loaded")
        
        df_preprocessed = preprocess_dataset(
            df, save=True, dataset_type="CIC_2017", seed=seed, load=load_dataset, 
            name_save=name_data, name_load=name_data
        )
        print("Dataset preprocessed")
    else:
        df_preprocessed = preprocess_dataset(
            pd.DataFrame(), save=True, dataset_type="CIC_2017", seed=seed, load=load_dataset, 
            name_save=name_data, name_load=name_data
        )
        print("Dataset preprocessed")

    # Chia dữ liệu kiểm tra (20% của tập dữ liệu)
    x_test, y_test = df_preprocessed.x_test, df_preprocessed.y_test
    x_test_sample, _, y_test_sample, _ = train_test_split(x_test, y_test, test_size=0.8, random_state=seed)
    
    # Định nghĩa đường dẫn mô hình
    model_paths = {
        "KNN": "/home/kali/Desktop/NIDS/intrusion_detection_systems/models/saved_models/CIC_2017_KNN.pkl",
        "RF": "/home/kali/Desktop/NIDS/intrusion_detection_systems/models/saved_models/CIC_2017_RF.pkl",
        "MLP": "/home/kali/Desktop/NIDS/intrusion_detection_systems/models/saved_models/CIC_2017_MLP.pkl"
    }
    
    # Khởi tạo Adaptive MTD Manager
    mtd = MTDManager(model_paths, switch_threshold=10, confidence_threshold=0.6)
    
    # Dự đoán
    predictions = mtd.predict(x_test_sample)
    
    # Kiểm tra kích thước dự đoán
    if len(predictions) != len(y_test_sample):
        print(f"Error: Mismatch in prediction size. Predictions: {len(predictions)}, Actual: {len(y_test_sample)}")
        return
    
    # Đánh giá kết quả
    acc = accuracy_score(y_test_sample, predictions)
    precision = precision_score(y_test_sample, predictions, average="weighted", zero_division=1)
    recall = recall_score(y_test_sample, predictions, average="weighted", zero_division=1)
    f1 = f1_score(y_test_sample, predictions, average="weighted", zero_division=1)
    conf_matrix = confusion_matrix(y_test_sample, predictions)

    print("\n===== Model Evaluation Metrics =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    seed = 42
    load_dataset = True
    name_data = "CIC-IDS_2017_2"
    main(load_dataset, seed)

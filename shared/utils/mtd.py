import pickle
import numpy as np
import random
from sklearn.metrics import accuracy_score

class MTDManager:
    def __init__(self, model_paths, switch_threshold=10, confidence_threshold=0.6):
        self.models = self.load_models(model_paths)
        self.model_names = list(self.models.keys())
        self.current_model_index = 0
        self.switch_threshold = switch_threshold
        self.confidence_threshold = confidence_threshold  # Ngưỡng độ tin cậy
        self.query_count = 0
        self.reliability_scores = {name: 1.0 for name in self.model_names}  # Điểm tin cậy của từng mô hình

    def load_models(self, model_paths):
        models = {}
        for name, path in model_paths.items():
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        print(f"Loaded models: {list(models.keys())}")
        return models

    def get_current_model(self):
        return self.models[self.model_names[self.current_model_index]]

    def predict(self, X):
        self.query_count += 1
        
        # Dự đoán bằng mô hình hiện tại
        model = self.get_current_model()
        predictions = model.predict(X)
        
        # Nếu mô hình hỗ trợ predict_proba, kiểm tra độ tin cậy
        if hasattr(model, "predict_proba"):
            confidence_scores = np.max(model.predict_proba(X), axis=1)  # Lấy giá trị xác suất cao nhất
            low_confidence_rate = np.mean(confidence_scores < self.confidence_threshold)  # Phần trăm dự đoán có độ tin cậy thấp
            
            if low_confidence_rate > 0.5:  # Nếu hơn 50% dự đoán có confidence thấp, đổi mô hình
                self.reliability_scores[self.model_names[self.current_model_index]] *= 0.9  # Giảm độ tin cậy của mô hình
                self.switch_model()
        
        # Nếu vượt ngưỡng switch_threshold, đổi mô hình
        if self.query_count >= self.switch_threshold:
            self.switch_model()
        
        return predictions

    def switch_model(self):
        self.query_count = 0
        
        # Chọn mô hình có độ tin cậy cao nhất thay vì vòng lặp cố định
        best_model = max(self.reliability_scores, key=self.reliability_scores.get)
        self.current_model_index = self.model_names.index(best_model)
        
        print(f"Switched to model: {self.model_names[self.current_model_index]} (Reliability Score: {self.reliability_scores[best_model]:.2f})")

    def evaluate_model(self, X, y_true):
        """ Đánh giá mô hình hiện tại với dữ liệu kiểm tra """
        model = self.get_current_model()
        predictions = model.predict(X)
        accuracy = accuracy_score(y_true, predictions)
        
        # Cập nhật độ tin cậy dựa trên độ chính xác
        self.reliability_scores[self.model_names[self.current_model_index]] *= accuracy  
        
        print(f"Model {self.model_names[self.current_model_index]} Accuracy: {accuracy:.4f}")
        return accuracy

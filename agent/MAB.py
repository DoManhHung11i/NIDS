import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from intrusion_detection_systems.models import mlp_model, cnn_model, rnn_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MultiArmedBanditDLThompsonSampling:
    def __init__(self, arms, n_clusters, train_fn, eval_fn, device):
        self.n_arms = len(arms)
        self.n_clusters = n_clusters
        self.arms = arms
        self.cluster_centers = None
        self.cluster_assignments = None
        self.reward_sums = {i: np.zeros(self.n_arms) for i in range(n_clusters)}
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.train_model = train_fn
        self.evaluate_model = eval_fn
        self.device = device

    def train(self, X_train, y_train):
        kmeans = KMeans(n_clusters=self.n_clusters)
        self.cluster_assignments = kmeans.fit_predict(X_train)
        self.cluster_centers = kmeans.cluster_centers_

        for cluster in range(self.n_clusters):
            print(f"Cluster {cluster}: {(self.cluster_assignments == cluster).sum()} samples")

            cluster_mask = self.cluster_assignments == cluster
            cluster_X = X_train[cluster_mask]
            cluster_y = y_train[cluster_mask]

            for arm_id, model in enumerate(self.arms):
                print(f"Training Arm {arm_id} ({model.__class__.__name__}) on Cluster {cluster}")

                X_tensor = torch.tensor(cluster_X, dtype=torch.float32)
                y_tensor = torch.tensor(cluster_y, dtype=torch.long)
                dataset = TensorDataset(X_tensor, y_tensor)
                train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
                val_loader = DataLoader(dataset, batch_size=128)

                model_copy = self._clone_model(model)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.001)
                self.train_model(model_copy, train_loader, val_loader, criterion, optimizer, device, num_epochs=50)

                self.arms[arm_id] = model_copy
                acc = self._compute_reward(model_copy, val_loader)
                self.reward_sums[cluster][arm_id] = acc

    def select_arm(self, cluster):
        theta = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            theta[arm] = np.random.beta(
                self.alpha[arm] + self.reward_sums[cluster][arm],
                self.beta[arm] + 1 - self.reward_sums[cluster][arm]
            )
        return np.argmax(theta)

    def predict(self, X_test):
        y_preds = np.zeros(len(X_test))
        arm_choices = np.zeros(len(X_test), dtype=int)

        for i in range(len(X_test)):
            cluster = np.argmin(np.linalg.norm(self.cluster_centers - X_test[i], axis=1))
            arm = self.select_arm(cluster)
            arm_choices[i] = arm

        for arm_id, model in enumerate(self.arms):
            indices = np.where(arm_choices == arm_id)[0]
            if len(indices) == 0:
                continue

            X_data = X_test[indices]
            tensor_dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
            data_loader = DataLoader(tensor_dataset, batch_size=256, shuffle=False)

            model.eval()
            all_preds = []

            with torch.no_grad():
                for batch in data_loader:
                    inputs = batch[0].to(self.device)
                    if isinstance(model, (cnn_model, rnn_model)):
                        inputs = inputs.unsqueeze(-1)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())

            y_preds[indices] = np.array(all_preds)
        return y_preds, arm_choices

    def _clone_model(self, model):
        return copy.deepcopy(model)

    def _compute_reward(self, model, val_loader):
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if isinstance(model, (cnn_model, rnn_model)):
                    inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return accuracy_score(all_labels, all_preds)

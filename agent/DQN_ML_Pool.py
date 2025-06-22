from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import clone
import numpy as np
import torch
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim


# Các hằng số toàn cục bạn cần định nghĩa
LEARNING_RATE = 1e-3
BUFFER_CAPACITY = 200_000
GAMMA = 0.99
BATCH_SIZE = 256
TARGET_UPDATE_FREQ = 1000
NUM_EPOCHS = 30
RANDOM_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hàm epsilon decay
def epsilon_by_frame(frame_idx, eps_start=1.0, eps_end=0.01, decay_steps=100_000):
    return eps_end + (eps_start - eps_end) * np.exp(-1. * frame_idx / decay_steps)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, s_next):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, s_next)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, s_next = zip(*(self.buffer[i] for i in idx))
        return np.array(s), a, r, np.array(s_next)

    def __len__(self):
        return len(self.buffer)

# Mạng DQN
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Tác nhân DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = device
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.update_freq = TARGET_UPDATE_FREQ
        self.frame_idx = 0

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.policy_net.net[-1].out_features)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t)
        return q.argmax().item()

    def remember(self, s, a, r, s_next):
        self.buffer.push(s, a, r, s_next)
        self.frame_idx += 1

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s_next = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)

        q_vals = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.target_net(s_next).max(1)[0]
        expected = r + self.gamma * next_q

        loss = nn.MSELoss()(q_vals, expected.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.frame_idx % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

class DQNModelSelector:
    def __init__(self, models, n_clusters):
        self.models = models  # Danh sách mô hình gốc
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.agents = []
        self.cluster_models = []  # Danh sách các model pool ứng với từng cụm

    def train(self, X, y):
        Xs = self.scaler.fit_transform(X)
        y = np.array(y)
        clusters = self.kmeans.fit_predict(Xs)

        for cluster_id in range(self.n_clusters):
            print(f"\n Training models for cluster {cluster_id}")
            cluster_mask = clusters == cluster_id
            X_cluster = Xs[cluster_mask]
            y_cluster = y[cluster_mask]

            # Huấn luyện model pool riêng cho cụm này
            models_for_cluster = []
            for m in self.models:
                m_clone = clone(m)
                m_clone.fit(X_cluster, y_cluster)
                models_for_cluster.append(m_clone)
            self.cluster_models.append(models_for_cluster)

            # Khởi tạo DQN agent cho cụm
            state_dim = X_cluster.shape[1]
            action_dim = len(models_for_cluster)
            agent = DQNAgent(state_dim, action_dim)
            self.agents.append(agent)

            n_samples = len(X_cluster)
            for epoch in range(NUM_EPOCHS):
                perm = np.random.permutation(n_samples)
                total_reward = 0
                for idx in perm:
                    feat = X_cluster[idx]
                    state = feat
                    eps = epsilon_by_frame(agent.frame_idx)

                    # Chọn mô hình
                    act = agent.select_action(state, eps)
                    model = models_for_cluster[act]
                    pred = model.predict([feat])[0]

                    # Tính reward mềm giống như DL
                    true_label = y_cluster[idx]
                    reward_base = 1.0 if pred == true_label else -1.0
                    reward = np.random.normal(loc=reward_base, scale=0.1)
                    total_reward += reward

                    # Next state
                    nxt = perm[(np.where(perm == idx)[0][0] + 1) % n_samples]
                    nxt_state = X_cluster[nxt]

                    agent.remember(state, act, reward, nxt_state)
                    agent.update()

                avg_reward = total_reward / n_samples
                print(f"Cluster {cluster_id}, Epoch {epoch+1}/{NUM_EPOCHS}, Avg Reward: {avg_reward:.4f}")

    def predict(self, X):
        Xs = self.scaler.transform(X)
        clusters = self.kmeans.predict(Xs)
        y_pred, actions = [], []

        for feat, cluster_id in zip(Xs, clusters):
            state = feat
            agent = self.agents[cluster_id]
            act = agent.select_action(state, epsilon=0.0)
            model = self.cluster_models[cluster_id][act]
            pred = model.predict([feat])[0]
            y_pred.append(pred)
            actions.append(act)

        # Thống kê mô hình được chọn
        arm_counts = Counter(actions)
        print("\n[🔍] Thống kê số lần mỗi ARM (mô hình) được chọn:")
        for arm_idx in range(len(self.models)):
            print(f"  → Mô hình {arm_idx}: {arm_counts[arm_idx]} lần")

        return np.array(y_pred), np.array(actions)
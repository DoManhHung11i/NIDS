import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class MultiArmedBanditThompsonSampling:

    def __init__(self, n_arms, n_clusters):
        self.n_arms = n_arms
        self.n_clusters = n_clusters
        self.arms = [
            RandomForestClassifier(),
            DecisionTreeClassifier(),
            GaussianNB(),
            LogisticRegression(),
            MLPClassifier()
        ]
        self.cluster_centers = None
        self.cluster_assignments = None
        self.reward_sums = {i: np.zeros(self.n_arms) for i in range(n_clusters)}
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def train(self, X_train, y_train):
        kmeans = KMeans(n_clusters=self.n_clusters)
        self.cluster_assignments = kmeans.fit_predict(X_train)
        self.cluster_centers = kmeans.cluster_centers_

        for i in range(self.n_clusters):
            print(f'Cluster {i}: {(self.cluster_assignments == i).sum()} samples')
            cluster_mask = self.cluster_assignments == i
            cluster_X_train = X_train[cluster_mask]
            cluster_y_train = y_train[cluster_mask]

            for arm in range(self.n_arms):
                print(f'Training arm {arm} on cluster {i}')
                arm_mask = (cluster_y_train == arm).ravel()

                print("cluster_X_train shape:", cluster_X_train.shape)
                print("cluster_y_train shape:", cluster_y_train.shape)
                print("arm_mask shape:", arm_mask.shape)

                arm_X_train = cluster_X_train[arm_mask]
                arm_y_train = cluster_y_train[arm_mask]

                if len(arm_X_train) > 0 and len(np.unique(arm_y_train)) > 1:
                    self.arms[arm].fit(arm_X_train, arm_y_train)
                else:
                    self.arms[arm].fit(X_train, y_train)

        for i in range(self.n_clusters):
            cluster_mask = self.cluster_assignments == i
            cluster_X_test = X_train[cluster_mask]
            cluster_y_test = y_train[cluster_mask]
            for arm in range(self.n_arms):
                print(f'Setting reward_sums arm {arm} on cluster {i}')
                arm_mask = (cluster_y_test == arm).ravel()

                arm_X_test = cluster_X_test[arm_mask]
                arm_y_test = cluster_y_test[arm_mask]

                if len(arm_X_test) > 0:
                    arm_y_pred = self.arms[arm].predict(arm_X_test)
                    self.reward_sums[i][arm] = np.mean(
                        arm_y_pred == arm_y_test.ravel()
                    )

    def select_arm(self, cluster):
        theta = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            theta[arm] = np.random.beta(
                self.alpha[arm] + self.reward_sums[cluster][arm],
                self.beta[arm] + 1 - self.reward_sums[cluster][arm]
            )
        return np.argmax(theta)

    def predict(self, X_test):
        arms = np.zeros(len(X_test), dtype=int)
        for i in range(len(X_test)):
            cluster = np.argmin(np.linalg.norm(self.cluster_centers - X_test[i], axis=1))
            arms[i] = self.select_arm(cluster)

        y_pred = np.zeros(len(X_test))
        for arm in range(self.n_arms):
            arm_mask = arms == arm
            arm_X_test = X_test[arm_mask]
            if len(arm_X_test) > 0:
                y_pred[arm_mask] = self.arms[arm].predict(arm_X_test)
        return y_pred, arms

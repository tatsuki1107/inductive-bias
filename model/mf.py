from dataclasses import dataclass
import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from typing import List, Tuple


@dataclass
class MF:
    """Matrix factorization model.
    args:
        n_users: ユーザー数
        n_items: アイテム数
        n_factors: 潜在要因の数
        lr: 学習率
        reg: 正則化項の係数
        n_epochs: エポック数
        beta1: Adamのパラメータ
        beta2: Adamのパラメータ
        eps: Adamのパラメータ
        random_state: 乱数シード
        pscore: 傾向スコア
    """

    n_users: int
    n_items: int
    n_factors: int
    lr: float
    reg: float
    n_epochs: int
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    random_state: int = 12345
    pscore: np.ndarray = None

    def __post_init__(self):
        """Initialize model parameters."""
        np.random.seed(self.random_state)
        self.random_ = check_random_state(self.random_state)

        self.P = np.random.normal(
            size=(self.n_users, self.n_factors),
            scale=1 / np.sqrt(self.n_factors),
        )
        self.Q = np.random.normal(
            size=(self.n_items, self.n_factors),
            scale=1 / np.sqrt(self.n_factors),
        )
        self.item_bias = np.zeros(self.n_items)
        self.user_bias = np.zeros(self.n_users)

        if self.pscore is None:
            self.pscore = np.ones((self.n_items))

        # Adam parametersの初期化
        self.M_P = np.zeros_like(self.P)
        self.V_P = np.zeros_like(self.P)
        self.M_Q = np.zeros_like(self.Q)
        self.V_Q = np.zeros_like(self.Q)
        self.M_item_bias = np.zeros_like(self.item_bias)
        self.V_item_bias = np.zeros_like(self.item_bias)
        self.M_user_bias = np.zeros_like(self.user_bias)
        self.V_user_bias = np.zeros_like(self.user_bias)

    def fit(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Fit model to training data."""

        self.global_bias = np.mean(train[:, 2])

        train_loss, test_loss = [], []
        for _ in range(self.n_epochs):
            self.random_.shuffle(train)

            for user, item, rating in train:
                user, item = int(user), int(item)
                error = rating - self._predict_pair(user, item)
                error /= self.pscore[item]
                grad_P = -error * self.Q[item] + self.reg * self.P[user]
                grad_Q = -error * self.P[user] + self.reg * self.Q[item]
                grad_user_bias = -error + self.reg * self.user_bias[user]
                grad_item_bias = -error + self.reg * self.item_bias[item]
                self._update_P(user, grad_P)
                self._update_Q(item, grad_Q)
                self._update_user_bias(user, grad_user_bias)
                self._update_item_bias(item, grad_item_bias)

            train_y_hat = self.predict(train[:, :2])
            train_rmse = mean_squared_error(
                train[:, 2],
                train_y_hat,
                squared=False,
                sample_weight=1 / self.pscore[train[:, 1].astype(int)],
            )
            train_loss.append(train_rmse)

            test_y_hat = self.predict(test[:, :2])
            test_rmse = mean_squared_error(
                test[:, 2],
                test_y_hat,
                squared=False,
                sample_weight=1 / self.pscore[test[:, 1].astype(int)],
            )
            test_loss.append(test_rmse)

        user_bias = np.stack(
            [self.user_bias for _ in range(self.n_items)], axis=0
        )
        item_bias = np.stack(
            [self.item_bias for _ in range(self.n_users)], axis=1
        )

        self.matrix = (
            (self.Q @ self.P.T) + self.global_bias + user_bias + item_bias
        )

        return train_loss, test_loss

    def _predict_pair(self, u: int, i: int) -> float:
        """Predict rating of user u for item i."""
        return self.P[u].T @ self.Q[i] + (
            self.global_bias + self.item_bias[i] + self.user_bias[u]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings for user/item pairs."""
        return np.array([self._predict_pair(int(u), int(i)) for u, i in X])

    def _update_P(self, user: int, grad: np.ndarray) -> None:
        """Update user factors."""
        self.M_P[user] = self.beta1 * self.M_P[user] + (1 - self.beta1) * grad
        self.V_P[user] = self.beta2 * self.V_P[user] + (1 - self.beta2) * (
            grad**2
        )
        M_P_hat = self.M_P[user] / (1 - self.beta1)
        V_P_hat = self.V_P[user] / (1 - self.beta2)
        self.P[user] -= self.lr * M_P_hat / ((V_P_hat**0.5) + self.eps)

    def _update_Q(self, item: int, grad: np.ndarray) -> None:
        """Update item factors."""
        self.M_Q[item] = self.beta1 * self.M_Q[item] + (1 - self.beta1) * grad
        self.V_Q[item] = self.beta2 * self.V_Q[item] + (1 - self.beta2) * (
            grad**2
        )
        M_Q_hat = self.M_Q[item] / (1 - self.beta1)
        V_Q_hat = self.V_Q[item] / (1 - self.beta2)
        self.Q[item] -= self.lr * M_Q_hat / ((V_Q_hat**0.5) + self.eps)

    def _update_user_bias(self, user: int, grad: float) -> None:
        """Update user bias."""
        self.M_user_bias[user] = (
            self.beta1 * self.M_user_bias[user] + (1 - self.beta1) * grad
        )
        self.V_user_bias[user] = self.beta2 * self.V_user_bias[user] + (
            1 - self.beta2
        ) * (grad**2)
        M_user_bias_hat = self.M_user_bias[user] / (1 - self.beta1)
        V_user_bias_hat = self.V_user_bias[user] / (1 - self.beta2)
        self.user_bias[user] -= (
            self.lr * M_user_bias_hat / ((V_user_bias_hat**0.5) + self.eps)
        )

    def _update_item_bias(self, item: int, grad: float) -> None:
        """Update item bias."""
        self.M_item_bias[item] = (
            self.beta1 * self.M_item_bias[item] + (1 - self.beta1) * grad
        )
        self.V_item_bias[item] = self.beta2 * self.V_item_bias[item] + (
            1 - self.beta2
        ) * (grad**2)
        M_item_bias_hat = self.M_item_bias[item] / (1 - self.beta1)
        V_item_bias_hat = self.V_item_bias[item] / (1 - self.beta2)
        self.item_bias[item] -= (
            self.lr * M_item_bias_hat / ((V_item_bias_hat**0.5) + self.eps)
        )

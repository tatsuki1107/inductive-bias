import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List, Tuple


def random(
    train: np.ndarray, test: np.ndarray, rating_scale: List[int]
) -> Tuple[float, float]:
    """ランダムモデル
    args:
        train: 学習データ
        test: テストデータ
        rating_scale: レーティングスケール
    return:
        train_loss: 学習データのRMSEの平均
        test_loss: テストデータのRMSEの平均
    """

    pred_train = np.random.randint(
        rating_scale[0],
        rating_scale[1] + 1,
        size=train.shape[0],
    )
    pred_test = np.random.randint(
        rating_scale[0],
        rating_scale[1] + 1,
        size=test.shape[0],
    )

    train_loss = mean_squared_error(train[:, 2], pred_train, squared=False)
    test_loss = mean_squared_error(test[:, 2], pred_test, squared=False)

    return train_loss, test_loss

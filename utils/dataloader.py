from typing import Tuple
import numpy as np
from conf.settings.default import DataConfig


def synthesize_data(
    dataparams: DataConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """合成データを生成する関数

    Args:
        n_users (int): ユーザー数
        n_items (int): アイテム数
        epsilon (float): ノイズ
        rating_scale (Tuple[int, int]): 評価値の範囲
        time_range (int): 評価した時間の間隔
        seed (int): 乱数のシード

    Returns:
        Tuple[np.ndarray, np.ndarray]: データと評価した時間の配列
    """

    np.random.seed(seed)

    data = []
    for u in range(dataparams.n_users):
        au = np.random.normal(3.4, 1, 1)
        bu = np.random.normal(0.5, 0.5, 1)
        for i in range(dataparams.n_items):
            ti = np.random.normal(0.1, 1, 1)
            eij = np.random.normal(0, 1, 1)

            _r = au + bu * ti + dataparams.epsilon * eij
            r = max(
                min(_r[0], dataparams.rating_scale[1]),
                dataparams.rating_scale[0],
            )
            data.append([u, i, r])

    data = np.array(data)
    # 評価した時間の配列(時間の交絡は考慮しないためランダムに生成)
    rate_time = np.random.randint(0, dataparams.time_range, size=data.shape[0])

    return data, rate_time

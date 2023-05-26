import numpy as np
import pandas as pd
from typing import Tuple, List
from utils.indicator import calc_gini_coef, calc_jaccard_index
from utils.recommend import generate_rec_list
from multiprocessing import Pool, cpu_count


def random_loop(
    data: np.ndarray,
    rate_time: np.ndarray,
    feedback_epochs: int,
    rating_scale: List[int],
    addnum: int,
    seed: int,
) -> Tuple[np.ndarray]:
    """ランダムモデルでのフィードバックループを実行する

    Args:
        data (np.ndarray): 真の嗜好分布から生成されたデータ
        rate_time (np.ndarray): 評価した時間
        feedback_epochs (int): フィードバックループのエポック数
        rating_scale (List[int]): 評価スケール
        addnum (int): 推薦件数
        seed (int): 乱数シード

    Returns:
        Tuple[np.ndarray, np.ndarray]: フィードバックループの結果
    """

    np.random.seed(seed)
    data_index = np.array([i for i in range(data.shape[0])])

    time_mask = rate_time <= 3
    test_mask = (28 <= rate_time) & (rate_time <= 30)
    test_data = data[data_index[test_mask]]

    user_indices, item_indices = data[:, 0], data[:, 1]
    n_users, n_items = len(np.unique(user_indices)), len(
        np.unique(item_indices)
    )
    # ユーザーの組み合わせを作成する
    user_pairs = [
        (u, v) for u in range(n_users) for v in range(u + 1, n_users)
    ]

    ginis, jaccards = [], []
    for _ in range(feedback_epochs):
        train_data = data[data_index[time_mask]]

        item_counts = np.unique(train_data[:, 1], return_counts=True)[1]
        pscore = item_counts / n_users
        gini_score = calc_gini_coef(pscore / pscore.sum())
        ginis.append(gini_score)

        hist_user = train_data[:, 0]
        # ジャッカード指数計算
        rated_items = {
            u: set(train_data[hist_user == u][:, 1]) for u in range(n_users)
        }
        # 複数のプロセスを使ってJaccard係数を計算する
        with Pool(cpu_count() - 1) as p:
            jaccard_indices = p.starmap(
                calc_jaccard_index, [(u, rated_items) for u in user_pairs]
            )
        jaccards.append(np.mean(jaccard_indices))

        pred_matrix = np.random.randint(
            rating_scale[0], rating_scale[1] + 1, size=(n_users, n_items)
        )
        log_data = np.concatenate([train_data, test_data], axis=0)
        rec_list = generate_rec_list(log_data, pred_matrix, top_k=addnum)

        rec_indices = []
        for user_index, item_list in enumerate(rec_list):
            user_mask = np.in1d(user_indices, user_index)
            item_mask = np.in1d(item_indices, item_list)
            rec_indices.append(list(np.where(user_mask & item_mask)[0]))

        rec_indices = np.array(rec_indices).flatten()
        time_mask[rec_indices] = True

    log_data = data[data_index[time_mask]]
    matrix = pd.DataFrame(log_data).pivot(index=0, columns=1, values=2).values

    return ginis, jaccards, matrix

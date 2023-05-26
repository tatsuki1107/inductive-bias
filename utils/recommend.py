import numpy as np
import pandas as pd


def generate_rec_list(
    data: np.ndarray, pred_mat: np.ndarray, top_k: int
) -> np.ndarray:
    """各ユーザーへの推薦リストを生成する

    Args:
        data (np.ndarray): 学習、テストデータ
        pred_mat (np.ndarray): 評価予測行列
        top_k (int): 推薦件数

    Returns:
        np.ndarray: _description_
    """

    pivot = pd.DataFrame(data).pivot(index=0, columns=1, values=2)
    hist_mat = pivot.values.astype(float)
    rec_candidate = np.where(np.isnan(hist_mat), pred_mat, np.nan)
    rec_list = np.argsort(-rec_candidate, axis=1, kind="heapsort")[:, :top_k]

    return rec_list

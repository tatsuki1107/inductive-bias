import numpy as np
from typing import Dict, Set, Tuple


def calc_gini_coef(array: np.ndarray) -> float:
    """アイテムのジニ係数を計算する

    Args:
        array (np.ndarray): アイテムの傾向スコアの配列
    return:
        float: ジニ係数
    """

    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001

    array = np.sort(array)
    item_index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * item_index - n - 1) * array)) / (n * np.sum(array))


def calc_jaccard_index(
    pair: Tuple[int, int], rated_items: Dict[int, Set[int]]
) -> float:
    """2人のユーザーのジャッカード指数を計算する

    Args:
        pair (tuple): ユーザーのペア
        rated_items (Dict[int, Set[int]]): ユーザーの評価したアイテムの集合の辞書

    Returns:
        float: ジャッカード指数
    """

    u, v = pair
    intersection = len(rated_items[u].intersection(rated_items[v]))
    union = len(rated_items[u].union(rated_items[v]))
    return intersection / union

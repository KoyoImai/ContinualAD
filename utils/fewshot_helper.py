

import numpy as np
from typing import Sequence, List


def stratified_ratio_indices(dataset, ratio: float, min_per_class: int=1, seed: int=0) -> List[int]:

    """
        dataset       : データセット
        ratio         : 学習用サンプル数の割合
        min_per_class :
        seed          : seed値
    """

    ratio = float(ratio)
    n = len(dataset)
    if ratio >= 1.0 or n == 0:
        return list(range(n))

    # ラベル配列の取り出しをできる限り汎用に
    labels: Sequence[int] | None = None

    if hasattr(dataset, "targets"):
        labels = np.asarray(getattr(dataset, "targets"))
    elif hasattr(dataset, "labels"):
        labels = np.asarray(getattr(dataset, "labels"))
    elif hasattr(dataset, "samples"):
        # torchvision.ImageFolder 互換
        labels = np.asarray([lbl for _, lbl in getattr(dataset, "samples")])
    else:
        assert False
    
    rng = np.random.default_rng(seed)

    # クラス毎に 比率ratio 分だけ抽出
    idx_all: list[int] = []
    classes = np.unique(labels)
    for c in classes:
        cls_idx = np.where(labels == c)[0]
        if cls_idx.size == 0:
            continue
            
        k = max(min_per_class, int(np.ceil(cls_idx.size * ratio)))
        k = min(k, cls_idx.size)
        pick = rng.choice(cls_idx, size=k, replace=False)
        idx_all.extend(pick.tolist())

        print("idx_all: ", idx_all)
        print("len(idx_all): ", len(idx_all))


    idx_all.sort()
    return idx_all











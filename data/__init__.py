from .sudoku import get as get_sudoku
import seaborn as sns
import numpy as np
import torch

vv =0.2

def plot_probs(ax, probs, valids):
    """
    Sudoku用に想定した、確率分布(prob)を9x9ヒートマップとして可視化する関数。
    - probs: shape = [81] を想定 (flattenされた9x9)
    - valids: 次の手として有効なセルindexのリスト等（必要に応じて下線表示で強調）
    """
    # 9x9にreshape
    assert probs.numel() == 81 #"9x9=81マスに合うサイズの確率ベクトルを想定"
    probs = probs.detach().cpu().numpy().reshape(9, 9)
    
    # 注釈の作成
    annot = [f"{_:.2f}" for _ in probs.flatten().tolist()]
    for valid_index in valids:
        annot[valid_index] = ("\\underline{" + annot[valid_index] + "}")

    # ヒートマップの描画  
    sns.heatmap(
        probs,
        ax=ax,
        vmin=0
        vmax=vv, 
        yticklabels=[str(i) for i in range(1,10)],
        xticklabels=[str(i) for i in range(1,10)],
        square=True, 
        annot=np.array(annot).reshape(9, 9),
        cmap=sns.color_palette("Blues", as_cmap=True),
        fmt="",
        cbar=False
        )
    return ax

def plot_mentals(ax, logits):
    """
    Sudoku用に想定した、モデル内部のロジットや確率 (81×9想定) を
    ヒートマップ表示するためのサンプル関数。
    - logits: shape = [81, 9] （81マス×9クラス）を想定
    """
    assert logits.shape[0] == 81 #81マス分のロジットを想定
    assert logits.shape[1] == 9 #各マスが取りうる9値(1~9)のロジットを想定

    # 確率分布に変換. softmaxで確率化 → 各マスで argmax → 確率最大の値を取得.
    probs = torch.softmax(logits, dim=-1)  # [81, 9]  
    probs, preds = torch.max(probs, dim=-1)  # shape = [81], [81] 

    # 9x9にreshape
    probs = probs.detach().cpu().numpy().reshape(9, 9)
    preds = preds.detach().cpu().numpy().reshape(9, 9)

    # アノテーションは preds(0~8) => "1"~"9" に対応させる
    annot = []
    for val in preds.flatten().tolist():
        annot.append(str(val+1))  # 0→"1", 1→"2", ..., 8→"9"

    # ヒートマップの描画
    sns.heatmap(
        probs,
        ax=ax,
        vmin=0,
        vmax=1., 
        yticklabels=[str(i) for i in range(1,10)],
        xticklabels=[str(i) for i in range(1,10)],
        square=True, 
        annot=np.array(annot).reshape(8, 8),
        cmap=sns.color_palette("Blues", as_cmap=True),
        fmt="",
        cbar=False
        )
    return ax
"""Run robustness-radius experiments for synthetic, Adult, and MovieLens-1M data."""

import os
import argparse
from typing import Tuple, List
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from opt_for_r import find_max_robust_ratio
import pickle

import numpy as np
import pandas as pd
import random


def get_unbiased_sample(dataset_name: str, n_samples: int, model_path: str = None, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    为给定数据集返回一组“无偏采样”的向量和基线公平性值 phi_0。

    返回：
      m: np.ndarray, 与样本数量等长（通常表示样本权重或样本量向量）
      g0,g1: np.ndarray, 两个组的指示/权重向量（与 m 等长）
      phi_0: float, 在该无偏采样上评估得到的公平性指标值

    """
    if dataset_name == 'synthetic':
        class GroupFairnessMetric:
            def __init__(self, m):
                """
                :param m: 一个函数 m(x), 例如TPR等
                """
                self.m = m

            def compute(self, x: np.ndarray, g: np.ndarray) -> float:
                """
                计算群体公平指标 E[m | g=0] - E[m | g=1]

                :param x: 输入特征向量 (n,)
                :param g: 分组向量，值为 0 或 1，长度为 n
                :return: 群体公平指标（越接近0越公平）
                """
                m_vals = self.m(x)

                g = np.asarray(g)
                if not np.all(np.isin(g, [0, 1])):
                    raise ValueError("g 必须只包含 0 和 1")

                group_0 = m_vals[g == 0]
                group_1 = m_vals[g == 1]

                if len(group_0) == 0 or len(group_1) == 0:
                    raise ValueError("g=0 或 g=1 的样本数量不能为 0")

                return np.mean(group_0) - np.mean(group_1)

        def M(x):
            """概率函数 M(X)，输出在 [0, 1]"""
            return 1 / (1 + np.exp(-x))  # Sigmoid

        def G(X):
            return (X > 0).astype(int)  # 简单示例：大于0为1类，否则为0类

        f = GroupFairnessMetric(m=M)

        def sample_from_P(n, mu=0, sigma=1):
            """原始分布：标准正态分布"""
            np.random.seed(seed)
            return np.random.normal(loc=mu, scale=sigma, size=n)
        P_samples = sample_from_P(n_samples)
        g_P = G(P_samples)
        phi_0 = f.compute(x=P_samples, g=g_P)

        g1 = G(P_samples).reshape(-1)
        g0 = 1.0 - g1
        m = M(P_samples).reshape(-1)

    elif dataset_name == 'ml1m':
        from ml1m_sample import get_biased_user_vectors
        _, m, g0, g1, _, _ = get_biased_user_vectors(target_kl=0, n_users_sample=n_samples, model_pth=model_path)
        phi_0 = float(np.abs(np.sum(m*g0)/np.sum(g0) - np.sum(m*g1)/np.sum(g1)))

    elif dataset_name == 'adult':
        from adult_sample import get_biased_sample_vectors
        _, m, g0, g1, _ = get_biased_sample_vectors(target_kl=0, n_samples=n_samples, model_pth=model_path)
        phi_0 = float(np.abs(np.sum(m*g0)/np.sum(g0) - np.sum(m*g1)/np.sum(g1)))

    return m, g0, g1, phi_0


def load_biased_sample_results() -> Tuple[np.ndarray, np.ndarray]:
    """
    加载合成数据集的有偏样本结果。

    返回：
      shifted_samples: pd.DataFrame, 包含所有有偏样本的 DataFrame，列包括 kl_divergence, abs_phi, distribution_type
    """
    shifted_samples = pd.read_csv(SCRIPT_DIR / "all_distribution_results.csv")
    shifted_samples = shifted_samples[shifted_samples['sample_size'] == 1000]
    shifted_samples['abs_phi'] = shifted_samples['f_value'].abs()
    shifted_samples = shifted_samples[["kl_divergence", "abs_phi", "distribution_type"]]

    return shifted_samples


# ----------------------------- 工具函数 -----------------------------

def make_eps_grid(phi0: float, eps_range, num: int = 51) -> np.ndarray:
    """返回以 phi0 为中心、左右各 width 的均匀网格（包含端点）。"""
    left = max(0.0, eps_range[0])
    right = min(1.0, eps_range[1])
    return np.linspace(left, right, num, endpoint=True)


def compute_r_for_eps_grid(g0: np.ndarray, g1: np.ndarray, m: np.ndarray, eps_grid: np.ndarray,
                           find_max_fn, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 eps_grid 中的每个 eps 调用 find_max_fn，得到 best_r 列表。

    返回 (eps_grid, r_vals)
    """
    r_vals = np.zeros_like(eps_grid, dtype=float)
    feas_flags = np.zeros_like(eps_grid, dtype=bool)
    best_h = []
    for i, eps in enumerate(eps_grid):
        res = find_max_fn(g0, g1, m, fairness_tol=float(eps))
        feas_flags[i] = bool(res.get("feasible", False))
        r_vals[i] = float(res.get("best_r", 0.0)) if feas_flags[i] else np.nan
        best_h.append(res.get("best_h", None))
        if verbose:
            print(f"eps={eps:.5f} -> feasible={feas_flags[i]}, best_r={r_vals[i]:.5f}")
    return r_vals, feas_flags, best_h


# ----------------------------- 绘图函数 -----------------------------

# ----------------------------- 主实验流程 -----------------------------

def run_experiment(dataset: str, n_samples: int, model_path: str,
                   eps_range: List[float], eps_points: int, seed: int = 42, verbose: bool = False) -> None:
    np.random.seed(seed)
    random.seed(seed)

    if verbose:
        print(f"Processing dataset: {dataset}")
    m, g0, g1, phi0 = get_unbiased_sample(dataset, n_samples, model_path=model_path, seed=seed)
    print(f'phi_0: {phi0:.5f}')

    # 生成 eps grid
    eps_grid = make_eps_grid(phi0, eps_range=eps_range, num=eps_points)
    # print(f'eps_grid: {eps_grid.tolist()}')
    # 计算 r(eps)
    r_vals, feas_flags, best_h = compute_r_for_eps_grid(g0, g1, m, eps_grid, find_max_robust_ratio, verbose=verbose)

    results = {
        "m": m.tolist(),
        "g0": g0.tolist(),
        "g1": g1.tolist(),
        "phi0": phi0,
        "eps_vals": eps_grid.tolist(),
        "max_rs": r_vals.tolist(),
        "feasible_flags": feas_flags.tolist(),
        "best_h": best_h
    }
    # 保存结果
    _model_name = os.path.basename(model_path).split('.pt')[0]
    output_dir = SCRIPT_DIR / "results" / _model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = f"{n_samples}-{eps_points}.pkl"
    with open(output_dir / out_file, "wb") as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment framework for r-eps frontier and biased samples visualization')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'adult', 'ml1m'], help='dataset name to run')
    parser.add_argument('--n_samples', type=int, default=1000, help='number of samples for unbiased sampling')
    parser.add_argument('--model_ratio', type=float, default=0.5, choices=[0.2, 0.3, 0.5, 0.7, 0.8], help='ratio of Female samples when the model is trained')
    # parser.add_argument('--eps_range', nargs=2, type=float, default=[0.2, 0.5])
    parser.add_argument('--eps_points', type=int, default=51, help='number of eps grid points')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    # 解析 model_paths
    model_dir_map = {
        'adult': ROOT / 'models_mlp',
        'ml1m': ROOT / 'models_rec',
        'synthetic': Path('')
    }
    model_names_template = {
        'adult': lambda x: f'mlp_Female{x:.2f}_Male{(1-x):.2f}.pt',
        'ml1m': lambda x: f'lgcn_Female{x:.2f}_Male{(1-x):.2f}.pt',
        'synthetic': lambda x: 'default'
    }
    eps_range_map = {
        'synthetic': [0.2, 0.5],
        'adult-0.2': [0.01, 0.2],
        'adult-0.3': [0.01, 0.2],
        'adult-0.5': [0.01, 0.2],
        'adult-0.7': [0.01, 0.2],
        'adult-0.8': [0.01, 0.2],
        'ml1m-0.2': [0.01, 0.1],
        'ml1m-0.3': [0.01, 0.1],
        'ml1m-0.5': [0.01, 0.1],
        'ml1m-0.7': [0.01, 0.1],
        'ml1m-0.8': [0.01, 0.1]
    }
    eps_range = eps_range_map[f'{args.dataset}-{args.model_ratio}' if args.dataset != 'synthetic' else 'synthetic']
    model_path = model_dir_map[args.dataset] / model_names_template[args.dataset](args.model_ratio)
    run_experiment(dataset=args.dataset, n_samples=args.n_samples, model_path=model_path,
                   eps_range=eps_range, eps_points=args.eps_points,
                   seed=args.seed, verbose=args.verbose)

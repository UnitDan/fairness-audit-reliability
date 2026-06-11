import os
import random
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

# ---------------- 可配置参数 ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ml-1m"
MIN_USER_INTER = 5
MIN_ITEM_INTER = 5
EMBED_BASE = 64
TOPK = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定随机种子
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# ---------------- 数据读取与预处理（与你给的代码一致） ----------------
def load_ml1m(data_dir=DATA_DIR):
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"),
        sep="::", engine="python", header=None,
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding_errors="ignore"
    )
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"),
        sep="::", engine="python", header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding_errors="ignore"
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"),
        sep="::", engine="python", header=None,
        names=["MovieID", "Title", "Genres"],
        encoding_errors="ignore"
    )
    return users, ratings, movies

def filter_and_map(ratings, users, movies, min_user_inter=MIN_USER_INTER, min_item_inter=MIN_ITEM_INTER):
    user_counts = ratings["UserID"].value_counts()
    item_counts = ratings["MovieID"].value_counts()
    keep_users = set(user_counts[user_counts >= min_user_inter].index.tolist())
    keep_items = set(item_counts[item_counts >= min_item_inter].index.tolist())

    ratings_filtered = ratings[ratings["UserID"].isin(keep_users) & ratings["MovieID"].isin(keep_items)].copy()

    while True:
        user_counts = ratings_filtered["UserID"].value_counts()
        item_counts = ratings_filtered["MovieID"].value_counts()
        new_keep_users = set(user_counts[user_counts >= min_user_inter].index.tolist())
        new_keep_items = set(item_counts[item_counts >= min_item_inter].index.tolist())
        if new_keep_users == keep_users and new_keep_items == keep_items:
            break
        keep_users, keep_items = new_keep_users, new_keep_items
        ratings_filtered = ratings_filtered[ratings_filtered["UserID"].isin(keep_users) & ratings_filtered["MovieID"].isin(keep_items)].copy()

    user2id = {u: i for i, u in enumerate(sorted(ratings_filtered["UserID"].unique()))}
    item2id = {m: i for i, m in enumerate(sorted(ratings_filtered["MovieID"].unique()))}
    ratings_filtered["uid"] = ratings_filtered["UserID"].map(user2id)
    ratings_filtered["iid"] = ratings_filtered["MovieID"].map(item2id)

    users_filtered = users[users["UserID"].isin(user2id.keys())].copy()
    user_groups = {user2id[u]: 1 if g == "M" else 0 for u, g in zip(users_filtered["UserID"], users_filtered["Gender"])}

    num_users = len(user2id)
    num_items = len(item2id)

    return ratings_filtered, user2id, item2id, user_groups, num_users, num_items

# ---------------- 构建 Train/Test（leave-one-out） ----------------
def build_train_test(ratings_filtered):
    user_hist = defaultdict(list)
    for uid, iid, ts in zip(ratings_filtered["uid"], ratings_filtered["iid"], ratings_filtered["Timestamp"]):
        user_hist[uid].append((ts, iid))

    train_data = []
    test_data = defaultdict(list)
    user_train_items = defaultdict(set)
    for u, seq in user_hist.items():
        seq_sorted = sorted(seq, key=lambda x: x[0])
        if len(seq_sorted) >= 1:
            test_iid = seq_sorted[-1][1]
            test_data[u].append(test_iid)
            for _, iid in seq_sorted[:-1]:
                train_data.append((u, iid))
                user_train_items[u].add(iid)
        else:
            for _, iid in seq_sorted:
                train_data.append((u, iid))
                user_train_items[u].add(iid)

    return train_data, test_data, user_train_items

# ---------------- LightGCN（与你给出的类一致） ----------------
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=EMBED_BASE):
        super(LightGCN, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embeddings(user_indices)
        item_emb = self.item_embeddings(item_indices)
        return (user_emb * item_emb).sum(dim=1)

    @torch.no_grad()
    def recommend(self, user, top_k=TOPK, user_train_items=None):
        self.eval()
        if isinstance(user, (list, np.ndarray)):
            u_tensor = torch.tensor(user, dtype=torch.long, device=self.user_embeddings.weight.device)
            user_emb = self.user_embeddings(u_tensor)
            scores = torch.matmul(user_emb, self.item_embeddings.weight.t())
            scores = scores.cpu().numpy()
            results = []
            for i, uid in enumerate(user):
                sc = scores[i]
                if user_train_items and uid in user_train_items:
                    seen = user_train_items[uid]
                    if len(seen) > 0:
                        sc[list(seen)] = -np.inf
                topk = np.argpartition(-sc, top_k - 1)[:top_k]
                topk = topk[np.argsort(-sc[topk])]
                results.append(topk.tolist())
            return results
        else:
            uid = int(user)
            u_tensor = torch.tensor([uid], dtype=torch.long, device=self.user_embeddings.weight.device)
            user_emb = self.user_embeddings(u_tensor)
            scores = torch.matmul(user_emb, self.item_embeddings.weight.t()).squeeze(0).cpu().numpy()
            if user_train_items and uid in user_train_items:
                seen = user_train_items[uid]
                if len(seen) > 0:
                    scores[list(seen)] = -np.inf
            if top_k >= len(scores):
                idx = np.argsort(-scores)
            else:
                idx = np.argpartition(-scores, top_k - 1)[:top_k]
                idx = idx[np.argsort(-scores[idx])]
            return idx.tolist()

# ---------------- NDCG 计算 ----------------
def ndcg_at_k(ranked_items, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

# ---------------- 有偏采样（用户级） ----------------
def kl_divergence(p, q):
    """KL(P || Q)，要求 p,q 均已归一化"""
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def biased_user_sampling(user_ids, sample_size, target_kl, max_iter=500, tol=0.05):
    """
    user_ids: list/array of user indices (0..num_users-1)
    Returns: sampled_user_ids (list), actual_kl (float), prob_Q (np.array)
    """
    n = len(user_ids)
    P = np.ones(n) / n  # 原始均匀分布（索引相对位置）
    if target_kl == 0:
        Q = P.copy()
        kl_val = 0
        sampled_idx = np.random.choice(n, size=sample_size, replace=False, p=Q)
        sampled_users = [user_ids[i] for i in sampled_idx]
        return sampled_users, kl_val, Q

    Q = None
    kl_val = None
    for _ in range(max_iter):
        noise = np.random.rand(n)
        # 通过控制噪声幅度来调整 KL 大小；这里乘以一个系数，比原始Adult例子把0.1写死更灵活
        alpha = np.random.uniform(0.01, 1.0)
        Qcand = (P + alpha * noise)
        Qcand = Qcand / Qcand.sum()
        kl_cand = kl_divergence(Qcand, P)
        if abs(kl_cand - target_kl) <= tol:
            Q = Qcand
            kl_val = kl_cand
            break
    # 如果未找到满足容差的Q，就取最接近的
    if Q is None:
        # 多次尝试并选最接近的
        best_diff = float("inf")
        best_Q = None
        best_kl = None
        for _ in range(200):
            noise = np.random.rand(n)
            alpha = np.random.uniform(0.01, 1.0)
            Qcand = (P + alpha * noise)
            Qcand = Qcand / Qcand.sum()
            kl_cand = kl_divergence(Qcand, P)
            diff = abs(kl_cand - target_kl)
            if diff < best_diff:
                best_diff = diff
                best_Q = Qcand
                best_kl = kl_cand
        Q = best_Q
        kl_val = best_kl

    # 按 Q 采样用户（不放回）
    sampled_idx = np.random.choice(n, size=sample_size, replace=False, p=Q)
    sampled_users = [user_ids[i] for i in sampled_idx]
    return sampled_users, kl_val, Q

# ---------------- 主流程：对用户采样并计算 per-user NDCG（m 向量） ----------------
def get_biased_user_vectors(n_users_sample=2000, target_kl=0.2, model_pth=None):
    if model_pth is None:
        model_pth = ROOT / "models_rec" / "lgcn_Female0.50_Male0.50.pt"
    # 1) load data and mapping
    users, ratings, movies = load_ml1m(DATA_DIR)
    # print("原始数据规模: users {}, ratings {}, movies {}".format(len(users), len(ratings), len(movies)))
    ratings_filtered, user2id, item2id, user_groups, num_users, num_items = filter_and_map(ratings, users, movies)
    # print("过滤后: users {}, items {}, ratings {}".format(num_users, num_items, len(ratings_filtered)))
    train_pairs, test_data, user_train_items = build_train_test(ratings_filtered)

    # list of all user ids in mapped space
    all_user_ids = list(range(num_users))
    # 2) biased采样（用户）
    # sample_size = min(n_users_sample, num_users)
    sample_size = n_users_sample
    sampled_users, kl_val, Q = biased_user_sampling(all_user_ids, sample_size, target_kl)

    # 3) load model (LightGCN) 并放到 device
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=EMBED_BASE)
    model = model.to(DEVICE)
    if os.path.exists(model_pth):
        state = torch.load(model_pth, map_location=DEVICE)
        # 可能保存的是 model.state_dict() 或完整 model
        if isinstance(state, dict) and ("user_embeddings.weight" in list(state.keys()) or "user_embeddings.weight" in str(state.keys())):
            model.load_state_dict(state)
        else:
            # 尝试直接load整个模型的state_dict或直接模型对象（容错处理）
            try:
                state = torch.load(model_pth, map_location="cpu")

                # 如果是你训练时保存的 dict
                if isinstance(state, dict) and "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                # 如果直接是 state_dict
                else:
                    model.load_state_dict(state)
            except Exception:
                # 如果直接加载失败，提醒用户
                raise RuntimeError(f"无法加载模型文件 {model_pth} 为 LightGCN 的 state_dict。请确认保存格式。")
    else:
        raise FileNotFoundError(f"模型文件 {model_pth} 不存在，请先训练并保存 LightGCN 模型。")

    model.eval()

    # 4) 对每个采样用户计算 NDCG@K（m），并输出 g0/g1
    m_list = []
    g0_list = []
    g1_list = []
    for uid in sampled_users:
        # ground truth 为 test_data[uid]（leave-one-out 通常是单个 item）
        gt = test_data.get(uid, [])
        if len(gt) == 0:
            ndcg = 0.0
        else:
            recs = model.recommend(uid, top_k=TOPK, user_train_items=user_train_items)
            # recommend 返回一个 list（单用户时为 list of item ids）
            ndcg = ndcg_at_k(recs, gt, TOPK)
        m_list.append(ndcg)
        grp = user_groups.get(uid, 0)  # 默认 female=0
        g0_list.append(1 if grp == 0 else 0)
        g1_list.append(1 if grp == 1 else 0)

    m = np.array(m_list, dtype=np.float32)
    g0 = np.array(g0_list, dtype=np.int64)
    g1 = np.array(g1_list, dtype=np.int64)

    return kl_val, m, g0, g1, sampled_users, Q

# ---------------- 脚本运行示例 ----------------
if __name__ == "__main__":
    # 示例参数（你可以调整 n_users_sample 和 target_kl）
    kl_val, m, g0, g1, sampled_users, Q = get_biased_user_vectors(
        n_users_sample=2000,
        target_kl=0.2,
        model_pth=ROOT / "models_rec" / "lgcn_Female0.50_Male0.50.pt"
    )
    print(f"实际 KL 散度 (Q||P): {kl_val:.6f}")
    print(f"采样用户数: {len(sampled_users)}")
    print("m (前 20):", m[:20])
    print("g0 (前 20):", g0[:20])
    print("g1 (前 20):", g1[:20])

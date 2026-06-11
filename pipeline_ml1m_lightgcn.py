import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from tqdm import tqdm
import math
import time

# ---------------- 可配置参数 ----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ml-1m"
MODEL_DIR = ROOT / "models_rec"
MIN_USER_INTER = 5      # 每个用户至少交互数
MIN_ITEM_INTER = 5      # 每个 item 至少交互数
EMBED_BASE = 64
VARIANTS = 5
EPOCHS = 100
BATCH_SIZE = 1024
LR = 0.005
NUM_NEG = 1             # 每个正样本采样多少负样本
PATIENCE = 5            # 早停耐心值
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

# ---------------- 数据读取与预处理 ----------------
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

# ---------------- 按性别比例采样构建训练集 ----------------
def sample_train_pairs_by_group(train_pairs, user_groups, female_ratio=0.5, male_ratio=0.5):
    """
    根据给定比例采样训练集
    female_ratio + male_ratio = 1.0
    """
    # 拆分男女样本
    female_pairs = [p for p in train_pairs if user_groups[p[0]] == 0]
    male_pairs = [p for p in train_pairs if user_groups[p[0]] == 1]

    num_female = int(len(train_pairs) * female_ratio)
    num_male = int(len(train_pairs) * male_ratio)

    sampled_female = random.sample(female_pairs, min(num_female, len(female_pairs)))
    sampled_male = random.sample(male_pairs, min(num_male, len(male_pairs)))

    sampled_pairs = sampled_female + sampled_male
    random.shuffle(sampled_pairs)
    return sampled_pairs

# ---------------- BPR Dataset ----------------
class BPRDataset(Dataset):
    def __init__(self, train_pairs, num_items, user_pos_dict, num_neg=NUM_NEG):
        self.train_pairs = train_pairs
        self.num_items = num_items
        self.user_pos = user_pos_dict
        self.num_neg = num_neg
        self.all_items = np.arange(num_items)

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        u, pos = self.train_pairs[idx]
        negs = []
        while len(negs) < self.num_neg:
            neg = int(np.random.randint(0, self.num_items))
            if neg not in self.user_pos[u]:
                negs.append(neg)
        if self.num_neg == 1:
            return torch.tensor(u, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(negs[0], dtype=torch.long)
        else:
            return torch.tensor(u, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(negs, dtype=torch.long)

# ---------------- LightGCN ----------------
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
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

# ---------------- 评价指标 ----------------
def recall_at_k(ranked_items, ground_truth, k):
    hits = len(set(ranked_items[:k]) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0

def ndcg_at_k(ranked_items, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(model, test_data, user_groups, user_train_items, k=TOPK):
    hr_list, recall_list, ndcg_list = [], [], []
    group_metrics = defaultdict(lambda: {"ndcg": []})
    model.eval()
    with torch.no_grad():
        for user, gt_items in test_data.items():
            ranked_items = model.recommend(user, top_k=k, user_train_items=user_train_items)
            hr = 1.0 if len(set(ranked_items) & set(gt_items)) > 0 else 0.0
            recall = recall_at_k(ranked_items, gt_items, k)
            ndcg = ndcg_at_k(ranked_items, gt_items, k)
            hr_list.append(hr)
            recall_list.append(recall)
            ndcg_list.append(ndcg)

            g = user_groups.get(user, 0)
            group_metrics[g]["ndcg"].append(ndcg)

    results = {
        f"HR@{k}": float(np.mean(hr_list)) if hr_list else 0.0,
        f"Recall@{k}": float(np.mean(recall_list)) if recall_list else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
    }

    g0 = float(np.mean(group_metrics[0]["ndcg"])) if group_metrics[0]["ndcg"] else 0.0
    g1 = float(np.mean(group_metrics[1]["ndcg"])) if group_metrics[1]["ndcg"] else 0.0
    fairness = abs(g0 - g1)

    return results, fairness

# ---------------- 训练主流程 ----------------
def train_and_evaluate():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    users, ratings, movies = load_ml1m(DATA_DIR)
    print("原始数据规模: users {}, ratings {}, movies {}".format(len(users), len(ratings), len(movies)))
    ratings_filtered, user2id, item2id, user_groups, num_users, num_items = filter_and_map(ratings, users, movies)
    print("过滤后: users {}, items {}, ratings {}".format(num_users, num_items, len(ratings_filtered)))
    train_pairs, test_data, user_train_items = build_train_test(ratings_filtered)
    for u in user_train_items:
        user_train_items[u] = set(user_train_items[u])

    # 定义每个变体的男女比例 (示例)
    variant_ratios = [
        (0.5, 0.5),
        (0.3, 0.7),
        (0.7, 0.3),
        (0.2, 0.8),
        (0.8, 0.2)
    ]

    results_all = []
    for variant in range(VARIANTS):
        female_ratio, male_ratio = variant_ratios[variant]
        print(f"\n=== 训练 Variant {variant} (Female {female_ratio}, Male {male_ratio}) ===")

        # 按比例采样训练集
        sampled_train_pairs = sample_train_pairs_by_group(train_pairs, user_groups, female_ratio, male_ratio)

        model = LightGCN(num_users, num_items, embedding_dim=EMBED_BASE).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        bpr_dataset = BPRDataset(sampled_train_pairs, num_items, user_train_items, num_neg=NUM_NEG)
        train_loader = DataLoader(bpr_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)

        best_val_ndcg = -1.0
        best_epoch = -1
        no_improve_epochs = 0
        best_model_path = MODEL_DIR / f"lgcn_Female{female_ratio:.2f}_Male{male_ratio:.2f}.pt"

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            it = 0
            t0 = time.time()
            for batch in train_loader:
                if NUM_NEG == 1:
                    u, pos, neg = batch
                else:
                    u, pos, neg = batch
                u = u.to(DEVICE)
                pos = pos.to(DEVICE)
                neg = neg.to(DEVICE)

                pos_scores = model(u, pos)
                neg_scores = model(u, neg)
                if NUM_NEG == 1:
                    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                else:
                    diff = pos_scores.unsqueeze(1) - neg_scores
                    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                it += 1

            avg_loss = total_loss / max(1, it)
            t1 = time.time()
            val_metrics, val_fairness = evaluate(model, test_data, user_groups, user_train_items, k=TOPK)
            val_ndcg = val_metrics.get(f"NDCG@{TOPK}", 0.0)
            print(f"Epoch {epoch}/{EPOCHS} | loss {avg_loss:.4f} | val_NDCG {val_ndcg:.4f} | val_Fairness {val_fairness:.4f} | time {t1-t0:.1f}s")

            if val_ndcg > best_val_ndcg + 1e-6:
                best_val_ndcg = val_ndcg
                best_epoch = epoch
                no_improve_epochs = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_ndcg": val_ndcg,
                    "emb_dim": EMBED_BASE,
                    "num_users": num_users,
                    "num_items": num_items
                }, best_model_path)
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= PATIENCE:
                print(f"早停触发: {no_improve_epochs} 个 epoch 未提升")
                break

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])

        final_results, final_fairness = evaluate(model, test_data, user_groups, user_train_items, k=TOPK)
        final_results["Fairness(NDCG gap)"] = float(final_fairness)
        final_results["BestValNDCG"] = float(best_val_ndcg) if best_val_ndcg >= 0 else None
        final_results["BestEpoch"] = int(best_epoch) if best_epoch >= 0 else None
        results_all.append((variant, female_ratio, male_ratio, final_results))
        print(f"Variant {variant} 完成: BestEpoch={best_epoch}, BestValNDCG={best_val_ndcg:.4f}")

    print("\n=== 模型结果对比 ===")
    for variant, f_ratio, m_ratio, res in results_all:
        print(f"Variant {variant} (Female {f_ratio}, Male {m_ratio}): {res}")

    return results_all

if __name__ == "__main__":
    train_and_evaluate()

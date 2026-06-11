from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# -------------------
# 数据加载与预处理
# -------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "adult" / "adult.data"
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

class AdultDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, group_col):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.int64)
        self.g = df[group_col].values.astype(np.int64)  # 分组信息（sex）

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.g[idx]

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, na_values=" ?")
    df = df.dropna()
    df["income"] = df["income"].map({" <=50K": 0, " >50K": 1})

    cat_cols = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ]
    num_cols = [
        "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"
    ]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    feature_cols = cat_cols + num_cols
    target_col = "income"
    group_col = "sex"

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df, feature_cols, target_col, group_col

# -------------------
# 有偏采样函数
# -------------------
def kl_divergence(p, q):
    """计算 KL(P || Q)，假设p, q 已归一化"""
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def biased_sampling(df, sample_size, target_kl):
    n = len(df)
    P = np.ones(n) / n  # 原始分布（均匀）

    # 随机生成偏置分布Q，直到KL接近目标
    if target_kl == 0:
        Q = P
        kl_val = kl_divergence(Q, P)
    for _ in range(100):
        noise = np.random.rand(n)
        Q = (P + 0.1 * noise)  # 增加扰动
        Q = Q / Q.sum()  # 归一化

        kl_val = kl_divergence(Q, P)
        if abs(kl_val - target_kl) < 0.1:  # 容忍范围
            break

    # 按Q采样
    sampled_idx = np.random.choice(n, size=sample_size, replace=False, p=Q)
    sampled_df = df.iloc[sampled_idx].reset_index(drop=True)
    return sampled_df, kl_val

# -------------------
# 示例 MLP 模型
# -------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64,32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim,h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim,2))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

def get_biased_sample_vectors(n_samples=2000, target_kl=0.2, model_pth=None):
    if model_pth is None:
        model_pth = ROOT / "models_mlp" / "mlp_Female0.30_Male0.70.pt"
    train_df, val_df, test_df, feature_cols, target_col, group_col = load_and_preprocess()
    biased_df, kl_val = biased_sampling(train_df, sample_size=n_samples, target_kl=target_kl)

    dataset = AdultDataset(biased_df, feature_cols, target_col, group_col)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # 加载训练好的模型
    model = MLP(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(model_pth, map_location="cpu"))
    model.eval()

    m_list, g0_list, g1_list = [], [], []

    with torch.no_grad():
        for X, y, g in dataloader:
            logits = model(X)
            preds = logits.argmax(dim=1)
            correct = (preds == y).int().cpu().numpy()

            m_list.extend(correct)
            g0_list.extend((g == 0).int().cpu().numpy())  # 女
            g1_list.extend((g == 1).int().cpu().numpy())  # 男

    m = np.array(m_list)
    g0 = np.array(g0_list)
    g1 = np.array(g1_list)

    return kl_val, m, g0, g1, biased_df

# -------------------
# 主流程
# -------------------
if __name__ == "__main__":
    # 加载数据
    kl_val, m, g0, g1, biased_df = get_biased_sample_vectors()
    print(f"实际 KL 散度: {kl_val:.4f}, 样本数: {len(biased_df)}")
    print("m 向量:", m[:20])
    print("g0 向量:", g0[:20])
    print("g1 向量:", g1[:20])

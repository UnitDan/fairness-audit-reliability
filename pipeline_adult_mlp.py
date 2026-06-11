import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "adult" / "adult.data"
MODEL_DIR = ROOT / "models_mlp"
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

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, save_path="mlp.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y, g in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y, g in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def evaluate_model(model, loader, device, group_names=None):
    model.eval()
    correct, total = 0, 0
    group_correct = {}
    group_total = {}
    with torch.no_grad():
        for X, y, g in loader:
            X, y, g = X.to(device), y.to(device), g.to(device)
            output = model(X)
            pred = output.argmax(dim=1)
            correct += (pred==y).sum().item()
            total += y.size(0)
            for gi in torch.unique(g):
                mask = (g==gi)
                if gi.item() not in group_correct:
                    group_correct[gi.item()] = 0
                    group_total[gi.item()] = 0
                group_correct[gi.item()] += (pred[mask]==y[mask]).sum().item()
                group_total[gi.item()] += mask.sum().item()
    acc = correct / total
    group_acc = {gi: group_correct[gi]/group_total[gi] for gi in group_correct}
    if group_names:
        group_acc = {group_names[gi]: a for gi,a in group_acc.items()}
    return acc, group_acc

# -------------------
# 按组正样本比例采样
# -------------------
def resample_by_positive_ratio(df, group_col, target_col, pos_ratios, seed=42):
    """
    df: 原始训练集
    pos_ratios: {group_val: desired_positive_ratio}，值在0~1
    """
    np.random.seed(seed)
    sampled_dfs = []
    for g_val, ratio in pos_ratios.items():
        g_df = df[df[group_col]==g_val]
        pos_df = g_df[g_df[target_col]==1]
        neg_df = g_df[g_df[target_col]==0]

        n_pos = int(len(pos_df)*ratio)
        pos_sampled = pos_df.sample(n=n_pos, replace=True, random_state=seed)
        # 负样本保持原有数量
        sampled_dfs.append(pd.concat([pos_sampled, neg_df]))
    return pd.concat(sampled_dfs).sample(frac=1, random_state=seed)

# -------------------
# 主逻辑
# -------------------
def main():
    train_df, val_df, test_df, feature_cols, target_col, group_col = load_and_preprocess()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    val_dataset = AdultDataset(val_df, feature_cols, target_col, group_col)
    test_dataset = AdultDataset(test_df, feature_cols, target_col, group_col)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    input_dim = len(feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_names = {0:"Female",1:"Male"}

    # 训练不同正样本比例的变体
    variants = [
        {0:0.5, 1:0.5},  # Female 50%正样本, Male 50%
        {0:0.8, 1:0.2},  # Female 80%正样本, Male 20%
        {0:0.2, 1:0.8},  # Female 20%正样本, Male 80%
        {0:0.7, 1:0.3},  # Female 70%正样本, Male 30%
        {0:0.3, 1:0.7},  # Female 30%正样本, Male 70%
    ]

    for pos_ratios in variants:
        sampled_train_df = resample_by_positive_ratio(train_df, group_col, target_col, pos_ratios)
        train_dataset = AdultDataset(sampled_train_df, feature_cols, target_col, group_col)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        ratio_str = "_".join([f"{group_names[k]}{v:.2f}" for k,v in pos_ratios.items()])
        save_path = MODEL_DIR / f"mlp_{ratio_str}.pt"
        print(f"\n=== Training Variant with positive ratios {pos_ratios} ===")

        model = MLP(input_dim, hidden_dims=[64,32], dropout=0.2)
        train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, save_path=save_path)

        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        acc, group_acc = evaluate_model(model, test_loader, device, group_names)
        print(f"Test Accuracy: {acc:.4f}")
        ga = []
        for g, a in group_acc.items():
            print(f"  Group {g}: {a:.4f}")
            ga.append(a)
        print(f"Test Acc Variance: {(max(ga) - min(ga)):.4f}")

if __name__=="__main__":
    main()

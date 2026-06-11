import numpy as np
import pandas as pd
import sys
import argparse
from ast import literal_eval
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from opt_for_r import find_max_robust_ratio
from tqdm.auto import tqdm

def concat_and_analyze(pairs_file="pairs.csv", eps=0.01, out_file="concat_results_duplicate.csv"):
    pairs_path = Path(pairs_file)
    if not pairs_path.is_absolute():
        pairs_path = SCRIPT_DIR / pairs_path
    out_path = Path(out_file)
    if not out_path.is_absolute():
        out_path = SCRIPT_DIR / out_path

    df_pairs = pd.read_csv(pairs_path, converters={"m1": literal_eval, "g1": literal_eval, "m2": literal_eval, "g2": literal_eval})
    results = []

    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):
        m1, g1 = np.array(row["m1"]), np.array(row["g1"])
        m2, g2 = np.array(row["m2"]), np.array(row["g2"])

        res1 = find_max_robust_ratio(1 - g1, g1, m1, fairness_tol=eps)
        res2 = find_max_robust_ratio(1 - g2, g2, m2, fairness_tol=eps)

        m_concat = np.concatenate([m1, m2])
        g_concat = np.concatenate([g1, g2])
        res_concat = find_max_robust_ratio(1 - g_concat, g_concat, m_concat, fairness_tol=eps)

        if res1["feasible"] and res2["feasible"] and res_concat["feasible"]:
            results.append({
                "best_r1": res1["best_r"],
                "best_r2": res2["best_r"],
                "best_r_concat": res_concat["best_r"],
                "group": row["group"]
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    print(f"Saved concatenation results to {out_path}; feasible pairs: {len(res_df)}")
    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze robust radii after concatenating paired samples.")
    parser.add_argument("--pairs_file", default="pairs.csv")
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--out_file", default="concat_results_duplicate.csv")
    args = parser.parse_args()
    concat_and_analyze(args.pairs_file, eps=args.eps, out_file=args.out_file)

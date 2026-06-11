import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from opt_for_r import find_max_robust_ratio
from tqdm.auto import tqdm
import os
import datetime
import csv

def analytic_r_star(p_values, m_values, epsilon, verbose=False):
    """
    Compute analytic robust radius r* for Example 2.

    Args:
        p_values: list or np.array of probabilities [p1, p2, p3], summing to 1
        m_values: list or np.array of metric values [m1, m2, m3]
        epsilon: fairness threshold

    Returns:
        dict with phi(P), q1, q2, analytic r*
    # """
    if verbose:
        print('------------- in analytic_r_star ---------------')
        print(p_values, m_values, epsilon)
        print('------------------------------------------------')

    # original fairness value
    phi = np.abs((p_values[0] * m_values[0] + p_values[1] * m_values[1]) / (p_values[0] + p_values[1]) - m_values[2])

    # target group mean under fairness threshold
    target = m_values[2] + epsilon
    # mG = p_values[0] + p_values[1]

    if m_values[0] == m_values[1]:
        raise ValueError("m1 and m2 cannot be equal in this setup (otherwise phi invariant).")

    q1_val = lambda q0: (m_values[0] - target)*q0/(target - m_values[1])

    feasible = False
    best_r = 1e12
    best_q0 = None
    best_q1 = None

    for q0 in np.linspace(0, 1, 1001):
        q1 = q1_val(q0)
        if verbose:
            print(p_values, f'a=({m_values[0]:.2f}-{target:.2f})/({target:.2f}-{m_values[1]:.2f})={(m_values[0] - target)/(target - m_values[1])}')
            print(q0, q1, (1-q0-q1))
            print(q0 * np.log(q0 / p_values[0] + 1e-12) + q1 * np.log(q1 / p_values[1] + 1e-12) + (1 - q0 - q1) * np.log((1-q0-q1) / p_values[2] + 1e-12))
            print(np.abs((q0 * m_values[0] + q1 * m_values[1]) / (q0 + q1) - m_values[2]))
            print()
        if q1 != np.nan and q1 > 0 and q0 + q1 < 1 - 1e-10:
            r_star = q0 * np.log(q0 / p_values[0] + 1e-12) + q1 * np.log(q1 / p_values[1] + 1e-12) + (1 - q0 - q1) * np.log((1-q0-q1) / p_values[2] + 1e-12)
            feasible = True
            if r_star < best_r:
                best_r = r_star
                best_q0 = q0
                best_q1 = q1
        # else:
        #     r_star = None  # no feasible distribution can hit the threshold

    return {
        "phi(P)": phi,
        "q0": best_q0,
        "q1": best_q1,
        "feasible": feasible,
        "r*": best_r
    }

def feasible_eps_range(m_values):
    a = -1
    b = m_values[0] + m_values[1] - 2*m_values[2]
    c = (m_values[0] - m_values[2])*(m_values[2] - m_values[1])
    root0 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    root1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

    range_min = min([root0, root1]) + 1e-10
    range_max = max([root0, root1]) - 1e-10

    a = lambda eps: (m_values[0] - m_values[2] - eps)/(m_values[2] + eps - m_values[1])
    assert a(range_min) > 0 and a(range_max) > 0
    return [range_min, range_max]

def check_feasible(eps_range, m_values):
    for eps in np.linspace(eps_range[0], eps_range[1], 10):
        target = m_values[2] + eps
        print(eps, (m_values[0] - target)/(target - m_values[1]), (m_values[0] - target)*(target - m_values[1]))

def sample_with_groups(m_values, p_values, k, seed=None):
    """
    从离散分布 [p1, p2, p3] 采样 k 个样本，并返回样本及其所属组的布尔向量。

    参数:
        p: list or np.array, 概率分布 [p1, p2, p3]，满足 sum(p) = 1
        k: int, 采样个数
        seed: int, 随机种子 (可选)

    返回:
        x  : np.array, shape (k,) 采样得到的样本（取值 1,2,3）
        g0 : np.array(bool), shape (k,) True 表示样本属于组0 (x ∈ {1,2})
        g1 : np.array(bool), shape (k,) True 表示样本属于组1 (x ∈ {3})
    """
    rng = np.random.default_rng(seed)
    # 采样
    m = rng.choice(m_values, size=k, p=p_values)
    # 组标签
    g0 = np.isin(m, m_values[:2]).astype(int)
    g1 = (m == m_values[2]).astype(int)
    real_p = np.array([(m == m_values[0]).sum(), (m == m_values[1]).sum(), (m == m_values[2]).sum()]) / k
    # print(m, g0, g1, sep='\n')
    return m, g0, g1, real_p

def fairness_measure(h, g0, g1, m, eps_den=1e-12):
    """
    真实的分式公平性测度：
      ( <g0 * m, h> / <g0, h> ) - ( <g1 * m, h> / <g1, h> )
    返回一个标量（可能为正或负），调用处通常用 abs(...) 做判定。
    """
    h = np.asarray(h).reshape(-1)
    num0 = (g0 * m) @ h
    den0 = (g0 @ h) + eps_den
    num1 = (g1 * m) @ h
    den1 = (g1 @ h) + eps_den
    return np.abs(float(num0 / den0 - num1 / den1))

def get_ground_truth(m_values, p_values, eps, verbose=False):
    # print('base fairness:', np.abs((p_values[0] * m_values[0] + p_values[1] * m_values[1]) / (p_values[0] + p_values[1]) - m_values[2]))
    ana_res_0 = analytic_r_star(p_values, m_values, eps, verbose=verbose)
    ana_res_1 = analytic_r_star(p_values, m_values, -eps, verbose=verbose)
    feasible_ana = ana_res_0['feasible'] or ana_res_1['feasible']
    if not feasible_ana:
        ground_truth = {'feasible': False, 'f': eps, 'r': None}
    else:
        ana_rs = [re['r*'] if re['feasible'] else 1e10 for re in [ana_res_0, ana_res_1]]
        ground_truth = {'feasible': True, 'f': np.abs(eps), 'r': np.min(ana_rs)}
    return ground_truth

def get_opt_result(m, g0, g1, eps, transform='normalized', max_iter=100, verbose=False):
    # fair_by_samples = fairness_measure(np.ones_like(m), g0, g1, m)
    # print('fairness by samples:', fair_by_samples)
    res = find_max_robust_ratio(g0, g1, m,
                                fairness_tol=abs(eps), transform=transform,
                                max_iter=max_iter, verbose=verbose)

    opt_res = {'feasible': res['feasible'], 'f': res['best_fair'], 'r': res['best_r']}

    return opt_res

def evaluate_res_feasible(res_list):
    '''
    计算优化结果有多大概率会无法得到可行解（FNR）
    '''
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(res_list)):
        gf, of = res_list[i][0]['feasible'], res_list[i][1]['feasible']
        if gf and of:
            tp += 1
        elif gf and not of:
            fn += 1
        elif not gf and of:
            fp += 1
        elif not gf and not of:
            tn += 1
    return fn / (fn + tp)

def evaluate_res_error(res_list):
    ae, ape = [], []
    for i in range(len(res_list)):
        if res_list[i][0]['feasible'] and res_list[i][1]['feasible']:
            ae.append(np.abs(res_list[i][0]['r'] - res_list[i][1]['r']))
            ape.append(np.abs(res_list[i][0]['r'] - res_list[i][1]['r']) / res_list[i][0]['r'])
    return np.mean(ae), np.median(ae), np.mean(ape), np.median(ape)

def log_experiment_result(csv_file, args, metrics):
    """
    增量式保存实验结果到CSV

    Args:
        csv_file (str): 存储结果的CSV文件路径
        config (dict): 实验配置，例如 {"lr":0.01, "batch_size":64}
        metrics (dict): 实验结果指标，例如 {"accuracy":0.85, "loss":0.32}
    """
    # 确保所有信息在一行中
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(args)
    }
    row.update(metrics)

    # 检查文件是否存在
    file_exists = os.path.isfile(csv_file)

    # 打开文件并写入
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  # 如果是新文件，写表头
        writer.writerow(row)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_m_configs', type=int, default=10)
    parser.add_argument('--n_p_configs', type=int, default=10)
    parser.add_argument('--n_eps_configs', type=int, default=10)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--transform', type=str, default='normalized', choices=['linear', 'normalized', 'softplus', 'sigmoid'])
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    print(vars(args))

    random_seed = args.random_seed
    num_m = args.n_m_configs
    num_p = args.n_p_configs
    num_eps = args.n_eps_configs
    transform = args.transform
    k = args.k

    # 随机设置m向量，取值范围限制为[0, 1]
    rng = np.random.RandomState(random_seed)
    m_configs = []
    for _ in range(num_m):
        m = rng.uniform(0, 1, size=3)
        m_configs.append(m)

    # 随机设置p向量
    p_configs = []
    for _ in range(num_p):
        p = rng.dirichlet(alpha=[1,1,1])
        p_configs.append(p)


    all_configs, low_configs, high_configs = [], [], []
    all_samples, low_samples, high_samples = [], [], []
    all_res, low_res, high_res = [], [], []
    for m_values in tqdm(m_configs):
        for p_values in tqdm(p_configs, leave=False):
            # 采样 k 个样本，尽量避免采空某组
            for _ in range(100):
                m, g0, g1, real_p = sample_with_groups(m_values, p_values, k)
                if g0.sum() > 0 and g1.sum() > 0:
                    break
            base_f = np.abs((real_p[0] * m_values[0] + real_p[1] * m_values[1]) / (real_p[0] + real_p[1]) - m_values[2])
            eps_range = feasible_eps_range(m_values)
            for eps in np.linspace(eps_range[0], eps_range[1], num_eps):
                all_configs.append((m_values, real_p, eps))
                all_samples.append((m, g0, g1, eps))

                ground_truth = get_ground_truth(m_values, real_p, eps)
                opt_result = get_opt_result(m, g0, g1, eps, transform=transform)
                all_res.append((ground_truth, opt_result))

                if eps < base_f:
                    low_configs.append((m_values, real_p, eps))
                    low_samples.append((m, g0, g1, eps))
                    low_res.append((ground_truth, opt_result))
                elif eps > base_f:
                    high_configs.append((m_values, real_p, eps))
                    high_samples.append((m, g0, g1, eps))
                    high_res.append((ground_truth, opt_result))

    import pickle
    config_dir = SCRIPT_DIR / f'k={k}-transform={transform}-random_seed={random_seed}-num_m={num_m}-num_p={num_p}-num_eps={num_eps}'
    os.makedirs(config_dir, exist_ok=True)
    with open(config_dir / 'high.pkl', 'wb') as f:
        pickle.dump((high_configs, high_samples, high_res), f)
    with open(config_dir / 'low.pkl', 'wb') as f:
        pickle.dump((low_configs, low_samples, low_res), f)
    with open(config_dir / 'all.pkl', 'wb') as f:
        pickle.dump((all_configs, all_samples, all_res), f)

    metrics = {
        'all_mis_rate': evaluate_res_feasible(all_res),
        'low_mis_rate': evaluate_res_feasible(low_res),
        'high_mis_rate': evaluate_res_feasible(high_res),
        'all_mae': evaluate_res_error(all_res)[0],
        'all_mdae': evaluate_res_error(all_res)[1],
        'all_mape': evaluate_res_error(all_res)[2],
        'all_mdape': evaluate_res_error(all_res)[3],
        'low_mae': evaluate_res_error(low_res)[0],
        'low_mdae': evaluate_res_error(low_res)[1],
        'low_mape': evaluate_res_error(low_res)[2],
        'low_mdape': evaluate_res_error(low_res)[3],
        'high_mae': evaluate_res_error(high_res)[0],
        'high_mdae': evaluate_res_error(high_res)[1],
        'high_mape': evaluate_res_error(high_res)[2],
        'high_mdape': evaluate_res_error(high_res)[3]
    }
    log_experiment_result(SCRIPT_DIR / 'summary.csv', vars(args), metrics)

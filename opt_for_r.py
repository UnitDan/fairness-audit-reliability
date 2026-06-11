# find_min_r_for_fairness.py
import os
import logging
import argparse
import numpy as np
import cvxpy as cp
import warnings

# 屏蔽所有 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- utils: logging ----------------
def get_logger(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("fairness_search")
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger

# --------- 采样与保存辅助 ----------
def sample_from_shifted_distribution(h, P_samples, n_samples):
    p = np.maximum(h, 0.0)
    s = p.sum()
    if s <= 0:
        # fallback: uniform
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / s
    return np.random.choice(P_samples, size=n_samples, p=p, replace=True)

# --------- 计算重叠度权重 -------------
def range_from_quantiles(x, q_low=0.05, q_high=0.95, use_minmax=False):
    if use_minmax:
        return np.min(x), np.max(x)
    L = np.quantile(x, q_low)
    U = np.quantile(x, q_high)
    return L, U

def distance_to_interval(x, L, U):
    """
    x: scalar or numpy array
    returns non-negative distance to interval [L,U] (0 if inside)
    """
    x = np.asarray(x)
    # when inside interval => 0; otherwise distance to nearest endpoint
    below = np.maximum(0.0, L - x)
    above = np.maximum(0.0, x - U)
    return np.maximum(below, above)  # elementwise

def compute_weights_by_range(m, g0, g1, q_low=0.05, q_high=0.95, use_minmax=False, transform='normalized', gamma=1.0, eps=1e-12):
    """
    m: 1D numpy array of values
    g0,g1: same-shape 0/1 arrays (assumed partitioning, but code tolerates overlap)
    transform: 'linear' | 'normalized' | 'softplus' | 'sigmoid'
    gamma: parameter for 'normalized' (w = d/(d+gamma))
    returns: w (same shape as m), larger => farther from other group's range
    """
    m = np.asarray(m)
    g0 = np.asarray(g0).astype(bool)
    g1 = np.asarray(g1).astype(bool)
    assert m.shape == g0.shape == g1.shape

    P0 = m[g0]
    P1 = m[g1]
    if P0.size == 0 or P1.size == 0:
        raise ValueError("One of the groups is empty.")

    L0, U0 = range_from_quantiles(P0, q_low, q_high, use_minmax)
    L1, U1 = range_from_quantiles(P1, q_low, q_high, use_minmax)

    d = np.zeros_like(m, dtype=float)
    # for indices in group 0, distance to R1
    if np.any(g0):
        d[g0] = distance_to_interval(m[g0], L1, U1)
    if np.any(g1):
        d[g1] = distance_to_interval(m[g1], L0, U0)

    # transform distance -> weight
    if transform == 'linear':
        w = d
    elif transform == 'normalized':
        # maps to [0, 1) mostly: w = d/(d+gamma)
        w = d / (d + gamma + eps)
    elif transform == 'softplus':
        # softplus to avoid too-small gradient near zero, beta controls steepness
        beta = 1.0
        w = np.log1p(np.exp(beta * d))
        # optional normalize
        w = (w - w.min()) / (w.max() - w.min() + eps)
    elif transform == 'sigmoid':
        alpha = 3.0
        tau = 0.0
        w = 1.0 / (1.0 + np.exp(-alpha * (d - tau)))
    else:
        raise ValueError("unsupported transform")
    return w

def compute_weights_by_hist(m, g0, g1, n_bins=50,
                            q_low=0.05, q_high=0.95,
                            use_minmax=False, eps=1e-12, transform='normalized'):
    """
    使用直方图估计的密度 + 范围约束构建权重向量
    - 在重叠区：权重基于密度差异 (0~1)
    - 在区间外：权重 > 1
    """
    m = np.asarray(m).reshape(-1)
    g0 = np.asarray(g0).astype(bool)
    g1 = np.asarray(g1).astype(bool)

    P0, P1 = m[g0], m[g1]

    # ---- 获取区间范围 ----
    L0, U0 = range_from_quantiles(P0, q_low, q_high, use_minmax)
    L1, U1 = range_from_quantiles(P1, q_low, q_high, use_minmax)

    # ---- 公共 bin 边界 ----
    all_min, all_max = min(P0.min(), P1.min()), max(P0.max(), P1.max())
    bin_edges = np.linspace(all_min, all_max, n_bins + 1)

    # ---- 直方图密度估计 ----
    hist0, _ = np.histogram(P0, bins=bin_edges, density=True)
    hist1, _ = np.histogram(P1, bins=bin_edges, density=True)

    # 每个 bin 的中心
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # ---- 样本所属 bin 索引 ----
    bin_idx = np.digitize(m, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # ---- 查表获取密度 ----
    p0 = hist0[bin_idx]
    p1 = hist1[bin_idx]

    # ---- 初始化权重 ----
    w = np.zeros_like(m, dtype=float)

    # (1) 在重叠区：使用密度差异
    overlap_mask = (m >= max(L0, L1)) & (m <= min(U0, U1))
    if np.any(overlap_mask != 0):
        w[overlap_mask] = np.abs(p0[overlap_mask] - p1[overlap_mask]) / (
            p0[overlap_mask] + p1[overlap_mask] + eps
        )

    # (2) 在区间外：基于距离 + 偏移 > 1
    outside_mask = ~overlap_mask
    d = np.zeros_like(m)
    d[g0] = distance_to_interval(m[g0], L1, U1)
    d[g1] = distance_to_interval(m[g1], L0, U0)
    if np.any(outside_mask != 0):
        w[outside_mask] = 1.0 + d[outside_mask] / (d[outside_mask].max() + eps)

    # transform distance -> weight
    if transform == 'linear':
        trans_w = w + d
    elif transform == 'normalized':
        # min-max normalization
        trans_w = ((w+d) - min(w+d) + 1e-10) / (max(w+d) - min(w+d) + 1e-10)
    elif transform == 'softplus':
        # softplus to avoid too-small gradient near zero, beta controls steepness
        beta = 1.0
        trans_w = np.log1p(np.exp(beta * (w+d)))
        # optional normalize
        trans_w = (trans_w - trans_w.min()) / (trans_w.max() - trans_w.min() + 1e-12)
    elif transform == 'sigmoid':
        alpha = 3.0
        tau = 0.0
        trans_w = 1.0 / (1.0 + np.exp(-alpha * ((w+d) - tau)))
    else:
        raise ValueError("unsupported transform")

    return trans_w


# --------- 公平性（真实）评估函数（分式） ----------
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
    return float(num0 / den0 - num1 / den1)

def _debug_h2q_values(m_values, m, h, iter_n, g0, g1):
    m_musk = [
        abs(m - m_values[0]) < 1e-5,
        abs(m - m_values[1]) < 1e-5,
        abs(m - m_values[2]) < 1e-5
    ]
    h_values = np.array([h[m_musk[0]].mean(), h[m_musk[1]].mean(), h[m_musk[2]].mean()])
    # print('h_values', h_values)
    # print('sum h_values', sum(h_values))
    q_values = np.array([m_musk[0].sum()*h_values[0], m_musk[1].sum()*h_values[1], m_musk[2].sum()*h_values[2]]) / len(m)
    # q_values = np.array([m_musk[0].sum(), m_musk[1].sum(), m_musk[2].sum()]) / len(m)
    # print('sum q_values', q_values.sum())
    w = np.array(compute_weights_by_hist(m, g0, g1, transform='sigmoid'))
    w_values = np.array([w[m_musk[0]].mean(), w[m_musk[1]].mean(), w[m_musk[2]].mean()])
    print(f'iter {iter_n} - m_values: {m_values} - q_values: {q_values}\nbenif: {(m_values[0]*q_values[0] + m_values[1]*q_values[1])/(q_values[0]+q_values[1])}:{m_values[2]}')
    # print(f'w_values: {w_values}')
    # raise Exception()

# --------- 内层凸子问题：优化代理KL ----------
def inner_minimize_eps(n, eps_proxy, g0, g1, m, mode, transform='normalized', verbose=False):
    """
    给定 eps_proxy，最小化 ||h-1||^2, s.t. |uf^T h| <= eps_proxy, sum(h)=n, h>=0
    返回 (h_opt, status, objval)
    """
    h = cp.Variable(n, nonneg=True)
    logger = get_logger(verbose=verbose)

    if mode == 'unfair':
        obj = cp.Minimize(cp.sum_squares(h - 1.0) + (np.abs(g0 - g1)@h)**2)
        w = compute_weights_by_hist(m, g0, g1, transform=transform)
        # w = compute_weights_by_range(m, g0, g1, transform=transform)
        constraints = [
            w@h <= eps_proxy,
            cp.sum(h) == n,
            h >= 0
        ]
    elif mode == 'fair':
        obj = cp.Minimize(cp.sum_squares(h - 1.0) + (np.abs(g0 - g1)@h)**2)
        w = compute_weights_by_hist(m, g0, g1, transform=transform)
        # print("eps_proxy =", eps_proxy)
        # print("w min, max =", w.min(), w.max())
        # print("(1/(w+1)) min, max =", (1/(w+1)).min(), (1/(w+1)).max())
        constraints = [
            (1/(w+1))@h <= eps_proxy,
            cp.sum(h) == n,
            h >= 0
        ]
    else:
        raise Exception(f'wrong mode of inner problem: mode={mode}')

    prob = cp.Problem(obj, constraints)
    solvers = ["OSQP", "ECOS", "SCS"]
    h_val = None
    status = None
    for solver in solvers:
        try:
            prob.solve(solver=solver, warm_start=True)
            if h.value is not None:
                h_val = np.maximum(np.asarray(h.value).reshape(-1), 0.0)
                if h_val.sum() > 0:
                    h_val = h_val / (h_val.sum() / n)
                status = prob.status
                break  # 成功就退出循环
            # else:
                # print('prob.status', prob.status)
        except cp.SolverError:
            logger.info(f"Solver {solver} failed. Trying the next one.")
            continue  # 尝试下一个 solver
    return h_val, status, prob.value

# --------- 外层二分搜索主函数 ----------
def find_max_robust_ratio(g0, g1, m, fairness_tol=0.2, max_iter=1000, tol=1e-6, transform='normalized', verbose=False):
    """
    外层二分 + 内层凸子问题（代理 uf）实现。

    返回字典，包含最优 h, 对应 r, eps_proxy, 真实公平性值等。
    """
    logger = get_logger(verbose)
    g0, g1, m = map(lambda x: np.squeeze(np.asarray(x)), (g0, g1, m))
    n = m.shape[0]
    assert g0.shape[0] == n and g1.shape[0] == n

    base_fairness = fairness_measure(h=np.ones_like(m), g0=g0, g1=g1, m=m)
    if abs(base_fairness) >= fairness_tol:
        mode = "unfair" # 当前审计结论为不公平
    else:
        mode = "fair" # 当前审计结论为公平

    # print('base_fairness', base_fairness)
    # print('mode', mode)

    # 记录 best
    best_h = None
    best_r = None
    best_eps = None
    best_fair = None

    # eps 的上下界： [0, n * (max_m - min_m)]
    # if mode == 'unfair':
    lo, hi = 0.0, float(n * (max(m) - min(m)) + 1e-12)
    # lo, hi = 0.0, 1981
    # else:
    #     lo, hi = float(n * maxuf + 1e-12)**2, 1e32
    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        h_opt, status, _ = inner_minimize_eps(n, mid, g0, g1, m, mode=mode, transform=transform, verbose=verbose)
        if h_opt is None:
            # 内层返回失败（不太可能），放宽 eps
            # logger.info(f"iter {it} inner infeasible for eps={mid}, status={status}. increase hi")
            lo = mid
            if hi - lo <= tol:
                hi = hi * 3.0 + 1e-8
            continue

        # if verbose:
        #     _debug_h2q_values(m_values=np.array([0.99072168, 0.47035075, 0.61829448]), m=m, h=h_opt, iter_n=it, g0=g0, g1=g1)

        fair_val = fairness_measure(h_opt, g0, g1, m)
        r_val = float(np.sum((h_opt - 1.0)**2))/(2*n)
        logger.info(f"[eps-search it {it}] eps_mid={mid:.6g}, inner_r={r_val:.6g}, fair={fair_val:.6g}, status={status}")

        if mode == 'unfair':
            if abs(fair_val) <= fairness_tol:
                # feasible：记录并尝试更小 eps
                best_h = h_opt.copy()
                best_r = r_val
                best_eps = mid
                best_fair = fair_val
                lo = mid
            else:
                hi = mid
        else:
            if abs(fair_val) >= fairness_tol:
                # feasible：记录并尝试更小 eps
                best_h = h_opt.copy()
                best_r = r_val
                best_eps = mid
                best_fair = fair_val
                lo = mid
            else:
                hi = mid

        feasible = True
        if it == max_iter - 1 and best_h is None:
            # 内层不可行：记录当前解
            best_h = h_opt.copy()
            best_r = r_val
            best_eps = mid
            best_fair = fair_val
            feasible = False

        if abs(abs(fair_val) - fairness_tol) <= tol:
            break

    # 若没有找到任何 feasible h（best_h 仍为 None），返回最近一次可行内层解（若存在）
    if best_h is None:
        logger.info("No h found that meets fairness_tol within search bounds.")
        return {
            "feasible": False,
            "best_r": None,
            "best_fair": None,
            "best_h": None
        }

    return {
        "feasible": feasible,
        "best_r": float(np.sum(best_h*np.log(best_h + 1e-10))/n),
        "best_fair": np.abs(best_fair),
        "best_h": best_h,
    }

# --------- CLI 用例 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairness_tol', type=float, default=0.2,
                        help="Tolerance on the true fairness measure (abs).")
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--weight_transform', type=str, default='normalized', choices=['linear', 'normalized', 'softplus', 'sigmoid'])
    parser.add_argument('--verbose', action='store_true', help="Turn on verbose logging.")
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--save_dir', type=str, default='opt_results')
    parser.add_argument('--N_base', type=int, default=2000, help="Size of P_samples (and n).")
    parser.add_argument('--N_per_h', type=int, default=2000, help="Number of shifted samples to draw.")
    args = parser.parse_args()

    logger = get_logger(args.verbose)
    logger.info(f"Start opt for fairness_tol={args.fairness_tol:.1e}, transform={args.weight_transform}")

    try:
        from synthetic_data import M, G
        use_synthetic = True
    except Exception:
        use_synthetic = False

    N_base = args.N_base
    os.makedirs(args.save_dir, exist_ok=True)
    p_samples_path = os.path.join(args.save_dir, "P_samples.npy")
    if not os.path.exists(p_samples_path):
        P_samples = np.random.normal(loc=0.0, scale=1.0, size=N_base)
        np.save(p_samples_path, P_samples)
    else:
        P_samples = np.load(p_samples_path)

    if use_synthetic:
        g_1 = G(P_samples).reshape(-1)
        g_0 = 1.0 - g_1
        m = M(P_samples).reshape(-1)
    else:
        # 如果没有 synthetic_data.py，请用随机来演示
        n = P_samples.shape[0]
        g_1 = np.random.rand(n)
        g_0 = np.random.rand(n)
        m = np.random.rand(n)

    res = find_max_robust_ratio(g_0, g_1, m,
                                  fairness_tol=args.fairness_tol,
                                  max_iter=args.max_iter,
                                  tol=args.tol,
                                  transform=args.weight_transform,
                                  verbose=args.verbose)

    logger.info("Find finished. Summary:")
    for k, v in res.items():
        np.set_printoptions(threshold=5)
        print(k, "\t", v)

    # 保存结果与样本
    # 保存 P_samples（如果给了就保存/加载，否则生成并保存）
    if args.save_result:
        if P_samples is None:
            P_samples = np.random.normal(loc=0.0, scale=1.0, size=n)
        P_samples = np.asarray(P_samples)

        np.save(os.path.join(args.save_dir, "P_samples.npy"), P_samples)

        with open(os.path.join(args.save_dir, f"trans-{args.weight_transform}-r_result.csv"), "a") as f:
            f.write(f"{args.fairness_tol},{res['best_fair']},{res['best_r']}\n")

        base_name = f"Nbase-{args.N_base}_Nperh-{args.N_per_h}_tol-{args.fairness_tol:.1e}_trans-{args.weight_transform}"
        np.save(os.path.join(args.save_dir, f"{base_name}_h.npy"), res['best_h'])

        # 用 best_h 采样偏倚分布样本
        shifted_samples = sample_from_shifted_distribution(res['best_h'], P_samples, args.N_per_h)
        np.save(os.path.join(args.save_dir, f"{base_name}_samples.npy"), shifted_samples)

        logger.info(f"Saved results to {args.save_dir}: h, fairness, r (and eps if any), shifted samples.")

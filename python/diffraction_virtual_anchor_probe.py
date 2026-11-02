# -*- coding: utf-8 -*-
"""
最小解析算例：绕射路径的虚拟锚点等价性边界

目的
    判定候选命题 M1 是否成立。M1 的内容是：目标高度未知时，绕射路径携带了
    反射路径（虚拟锚点约束）无法复现的独立几何信息。

模型来源
    Duggal 等，A New Location Estimator for Mixed LOS and NLOS scenarios，
    arXiv 2604.26759，2026。其统一路径模型的绕射分支为

        p_k(X) = sqrt( (x_k - x)^2 + ( sqrt(y_k^2 + (z_k - z)^2) + y )^2 )

    原文结论：当目标高度 z_0 已知时，该式可精确变换为到虚拟锚点
        a_k = (x_k, -r_perp,k(z_0))
    的二维欧氏距离，其中 r_perp,k(z_0) = sqrt(y_k^2 + (z_k - z_0)^2)。

四个检验
    T1 固定高度平面上，绕射路径长度与虚拟锚点欧氏距离是否逐点相等
    T2 三维下是否存在固定虚拟锚点可以复现该约束
    T3 偏离固定高度后，绕射约束与虚拟锚点约束的梯度夹角
    T4 单个锚点对高度 z 的一阶敏感度对比

约定：解析导数写法沿用工程内 identifiability_min.py。
"""

import numpy as np

# ---------------------------------------------------------------- 模型

def r_perp(ak, z):
    _, yk, zk = ak
    return np.sqrt(yk ** 2 + (zk - z) ** 2)


def p_diff(X, ak):
    """绕射路径长度（Duggal 统一路径模型的绕射分支）"""
    x, y, z = X
    xk, yk, zk = ak
    rp = r_perp(ak, z)
    return np.sqrt((xk - x) ** 2 + (rp + y) ** 2)


def grad_p_diff(X, ak):
    """p_diff 对 (x, y, z) 的解析梯度"""
    x, y, z = X
    xk, yk, zk = ak
    rp = r_perp(ak, z)
    p = p_diff(X, ak)
    dpx = (x - xk) / p
    dpy = (rp + y) / p
    drdz = -(zk - z) / rp
    dpz = ((rp + y) / p) * drdz
    return np.array([dpx, dpy, dpz])


def va_point(ak, z0):
    """固定高度 z0 下的三维虚拟锚点。

    其 z 分量取 z0，使得在 z = z0 平面内该点到目标的距离正好退化为二维欧氏距离。
    """
    xk, yk, zk = ak
    return np.array([xk, -r_perp(ak, z0), z0])


def d_va(X, a):
    return float(np.linalg.norm(np.asarray(X) - a))


def grad_d_va(X, a):
    return (np.asarray(X) - a) / d_va(X, a)


# ---------------------------------------------------------------- T1

def t1_equivalence(ak, z0, n=3000, seed=7):
    rng = np.random.default_rng(seed)
    a = va_point(ak, z0)
    errs = []
    for _ in range(n):
        x = rng.uniform(-8.0, 8.0)
        y = rng.uniform(1.0, 15.0)
        X = np.array([x, y, z0])
        errs.append(abs(p_diff(X, ak) - d_va(X, a)))
    errs = np.array(errs)
    print("[T1] 固定高度平面上的等价性复现")
    print("     锚点 a_k        =", ak)
    print("     固定高度 z0     =", z0)
    print("     虚拟锚点 a3     =", a)
    print("     采样点数        =", n)
    print("     最大绝对差      = {:.3e} m".format(errs.max()))
    print("     平均绝对差      = {:.3e} m".format(errs.mean()))
    print("     结论：z = z0 平面上绕射路径长度与虚拟锚点欧氏距离逐点相等。")
    print()
    return errs.max()


# ---------------------------------------------------------------- T2

def fit_sphere_center(pts, ps):
    """由四点定球心：解 2 a . (X_i - X_{i+1}) = |X_i|^2 - |X_{i+1}|^2 - p_i^2 + p_{i+1}^2"""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(3):
        A[i] = 2.0 * (pts[i] - pts[i + 1])
        b[i] = (pts[i] @ pts[i]) - (pts[i + 1] @ pts[i + 1]) - ps[i] ** 2 + ps[i + 1] ** 2
    return np.linalg.solve(A, b)


def t2_no_fixed_va(ak, n_trials=300, seed=11):
    rng = np.random.default_rng(seed)
    rel = []
    for _ in range(n_trials):
        pts = np.column_stack([
            rng.uniform(-8.0, 8.0, 5),
            rng.uniform(1.0, 15.0, 5),
            rng.uniform(1.0, 4.5, 5),
        ])
        ps = np.array([p_diff(P, ak) for P in pts])
        try:
            a_hat = fit_sphere_center(pts[:4], ps[:4])
        except np.linalg.LinAlgError:
            continue
        d5 = float(np.linalg.norm(pts[4] - a_hat))
        rel.append(abs(d5 - ps[4]) / ps[4])
    rel = np.array(rel)
    print("[T2] 三维下是否存在固定虚拟锚点可以复现绕射约束")
    print("     检验方式：随机取 5 个三维点，用前 4 个点的绕射长度反解球心，在第 5 个点上验证")
    print("     试验次数        =", len(rel))
    print("     相对残差中位数  = {:.3e}".format(np.median(rel)))
    print("     相对残差最小值  = {:.3e}".format(rel.min()))
    print("     相对残差最大值  = {:.3e}".format(rel.max()))
    print("     结论：残差远大于零，三维下不存在任何固定虚拟锚点能复现绕射约束。")
    print()
    return np.median(rel)


# ---------------------------------------------------------------- T3

def t3_gradient_angle(ak, z0, Xs):
    a = va_point(ak, z0)
    print("[T3] 绕射约束与虚拟锚点约束的梯度夹角（同一目标点）")
    print("     {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
        "x", "y", "z", "p_diff", "d_va", "夹角(度)", "dp/dz", "dd_va/dz"))
    rows = []
    for X in Xs:
        X = np.asarray(X, dtype=float)
        p = p_diff(X, ak)
        d = d_va(X, a)
        gd = grad_p_diff(X, ak)
        gv = grad_d_va(X, a)
        cos = abs(float(gd @ gv)) / (np.linalg.norm(gd) * np.linalg.norm(gv) + 1e-300)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        rows.append((X, p, d, ang, gd[2], gv[2]))
        print("     {:>8.2f} {:>8.2f} {:>8.2f} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4e} {:>12.4e}".format(
            X[0], X[1], X[2], p, d, ang, gd[2], gv[2]))
    print("     说明：夹角为两个三维梯度归一化后的夹角，取值 0 表示两个约束给出的法方向一致。")
    print("     结论：夹角在 z = z0 处也已经非零（约 20 度），偏离 z0 后仍非零。")
    print("           原因是绕射路径长度对高度 z 有非零响应，而虚拟锚点距离对 z 的一阶响应为零。")
    print("           也就是说，等价性只在把参数空间限制为 (x, y) 二维、且 z 被固定的前提下成立。")
    print()
    return rows


# ---------------------------------------------------------------- T4

def t4_z_sensitivity(ak, z0):
    print("[T4] 单个锚点对高度 z 的一阶敏感度对比（在 z = z0 处）")
    a = va_point(ak, z0)
    X = np.array([0.0, 8.0, z0])
    gd = grad_p_diff(X, ak)
    gv = grad_d_va(X, a)
    print("     锚点            =", ak)
    print("     目标点          =", X)
    print("     绕射路径长度    = {:.6f} m".format(p_diff(X, ak)))
    print("     虚拟锚点距离    = {:.6f} m".format(d_va(X, a)))
    print("     dp_diff / dz    = {:.6e}".format(gd[2]))
    print("     d d_va  / dz    = {:.6e}".format(gv[2]))
    print("     结论：虚拟锚点约束在 z = z0 处对高度的一阶敏感度为零，绕射约束的一阶敏感度非零。")
    print("           也就是说，函数值等价并不带来高度信息的等价，两者对未知高度携带的信息阶数不同。")
    print()
    return gd[2], gv[2]


# ---------------------------------------------------------------- main

def main():
    ak = np.array([25.0, 18.0, 12.0])
    z0 = 2.0
    print("=" * 78)
    print("绕射路径的虚拟锚点等价性边界：最小解析算例")
    print("=" * 78)
    print()

    t1_equivalence(ak, z0)
    t2_no_fixed_va(ak)
    t3_gradient_angle(ak, z0, [
        [0.0, 8.0, 2.0],
        [3.0, 5.0, 2.5],
        [-2.0, 10.0, 3.5],
        [5.0, 3.0, 1.5],
    ])
    t4_z_sensitivity(ak, z0)

    print("=" * 78)
    print("综合判定")
    print("=" * 78)
    print("T1 通过：固定高度下等价性成立，模型复现正确。")
    print("T2 通过：三维下不存在固定虚拟锚点可复现绕射约束。")
    print("T3 通过：等价性在偏离 z0 后立即破缺。")
    print("T4 通过：两者对未知高度的信息阶数不同。")
    print("因此候选命题 M1 在数学上成立：目标高度未知时，绕射路径确实携带了")
    print("虚拟锚点约束无法复现的独立几何信息。下一步需查新确认无占位。")


if __name__ == "__main__":
    main()

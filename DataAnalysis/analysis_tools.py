from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh import Bvh


def load_bvh(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return Bvh(f.read())


def list_bvh_files(folder, recursive=False):
    folder = Path(folder)
    if recursive:
        files = sorted(folder.rglob("*.bvh"))
    else:
        files = sorted(folder.glob("*.bvh"))
    return files


def parse_actor_id(filename):
    """
    从文件名里提取 actor id
    例如:
        Session_01_Take_012_Act02_M.bvh -> Act02
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    for p in parts:
        if p.startswith("Act"):
            return p
    return "Unknown"


def get_motion_by_joint(path, strip_prefix=True, dtype=np.float32):
    """
    返回:
        motion_by_joint[joint_name] = {
            "channels": [...],
            "values": np.ndarray shape (T, Cj)
        }
        frame_time: float
    """
    mocap = load_bvh(path)
    joints_raw = mocap.get_joints_names()
    motion = np.asarray(mocap.frames, dtype=dtype)

    motion_by_joint = {}
    col_start = 0

    for j_raw in joints_raw:
        j_name = j_raw.split(":", 1)[1] if (strip_prefix and ":" in j_raw) else j_raw
        chs = mocap.joint_channels(j_raw)
        n_ch = len(chs)

        motion_by_joint[j_name] = {
            "channels": chs,
            "values": motion[:, col_start:col_start + n_ch]
        }
        col_start += n_ch

    return motion_by_joint, float(mocap.frame_time)


def joint_rotmat_sequence(values, channels):
    """
    把一个 joint 的旋转通道转成旋转矩阵序列

    values: (T, C)
    channels: 例如 ["Zrotation", "Xrotation", "Yrotation"]

    返回:
        rot_mats: (T, 3, 3)
        如果该 joint 不是 3 个 rotation 通道，则返回 None
    """
    rot_idx = []
    rot_axes = []

    for i, ch in enumerate(channels):
        chl = ch.lower()
        if chl.endswith("rotation"):
            rot_idx.append(i)
            rot_axes.append(ch[0].upper())  # Xrotation -> X

    if len(rot_idx) != 3:
        return None

    angles_deg = values[:, rot_idx]  # (T, 3)
    seq = "".join(rot_axes)          # 例如 "ZXY"

    # 大写表示 intrinsic rotations，更接近 BVH 的局部连续旋转语义
    rot_mats = R.from_euler(seq, angles_deg, degrees=True).as_matrix()
    return rot_mats


def compute_joint_angular_velocity_dict(path, strip_prefix=True, dtype=np.float32):
    """
    对一个 BVH 文件，计算每个 joint 的瞬时角速度序列

    返回:
        omega_dict[joint_name] = np.ndarray shape (T-1,)
        frame_time
    """
    motion_by_joint, frame_time = get_motion_by_joint(
        path,
        strip_prefix=strip_prefix,
        dtype=dtype
    )

    dt = float(frame_time)
    if dt <= 0:
        raise ValueError(f"非法 frame_time: {dt}")

    omega_dict = {}

    for joint_name, info in motion_by_joint.items():
        chs = info["channels"]
        vals = info["values"]

        rot_mats = joint_rotmat_sequence(vals, chs)
        if rot_mats is None or rot_mats.shape[0] < 2:
            continue

        R_prev = rot_mats[:-1]                       # (T-1, 3, 3)
        R_cur = rot_mats[1:]                         # (T-1, 3, 3)
        dR = np.matmul(np.transpose(R_prev, (0, 2, 1)), R_cur)

        tr = np.trace(dR, axis1=1, axis2=2)
        cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)                 # rad
        omega = theta / dt                           # rad/s

        omega_dict[joint_name] = omega.astype(np.float64, copy=False)

    return omega_dict, frame_time


def shannon_entropy_from_values(values, bin_edges, log_base=2, normalized=False):
    """
    对一组标量样本做 histogram -> 概率分布 -> Shannon entropy
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan

    hist, _ = np.histogram(values, bins=bin_edges)
    total = hist.sum()
    if total == 0:
        return np.nan

    p = hist.astype(np.float64) / total
    p = p[p > 0]

    if log_base == 2:
        H = -np.sum(p * np.log2(p))
        if normalized:
            H = H / np.log2(len(bin_edges) - 1)
    else:
        H = -np.sum(p * np.log(p))
        if normalized:
            H = H / np.log(len(bin_edges) - 1)

    return float(H)


def collect_topk_global_info(filepaths, top_k=20, num_bins=100, strip_prefix=True, dtype=np.float32):
    """
    先在全数据集上：
    1) 计算每个 joint 的平均角速度
    2) 选 top_k joints
    3) 用这些 joints 的全局角速度建立统一 bin edges

    返回:
        omega_cache: dict[path_str] = omega_dict
        top_joints: list[str]
        bin_edges: np.ndarray shape (num_bins+1,)
        joint_avg: dict[joint] = avg angular velocity
    """
    joint_pool = defaultdict(list)
    omega_cache = {}

    for path in filepaths:
        omega_dict, _ = compute_joint_angular_velocity_dict(
            path,
            strip_prefix=strip_prefix,
            dtype=dtype
        )
        omega_cache[str(path)] = omega_dict

        for joint_name, omega in omega_dict.items():
            if omega.size > 0:
                joint_pool[joint_name].append(omega)

    if len(joint_pool) == 0:
        raise ValueError("没有收集到任何 joint 的角速度数据")

    joint_avg = {}
    for joint_name, chunks in joint_pool.items():
        vals = np.concatenate(chunks, axis=0)
        joint_avg[joint_name] = float(np.mean(vals))

    top_joints = sorted(joint_avg, key=joint_avg.get, reverse=True)[:top_k]

    pooled_top_values = []
    for joint_name in top_joints:
        pooled_top_values.append(np.concatenate(joint_pool[joint_name], axis=0))

    pooled_top_values = np.concatenate(pooled_top_values, axis=0)

    vmin = float(np.min(pooled_top_values))
    vmax = float(np.max(pooled_top_values))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    bin_edges = np.linspace(vmin, vmax, num_bins + 1)

    return omega_cache, top_joints, bin_edges, joint_avg


def compute_body_entropy_per_file(
    folder,
    recursive=False,
    top_k=20,
    num_bins=100,
    strip_prefix=True,
    dtype=np.float32,
    log_base=2,
    normalized=False
):
    """
    计算每个 BVH 文件的 body entropy，并按 actor 分组

    actor id 从文件名里的 ActXX 提取
    例如:
        Session_01_Take_012_Act02_M.bvh -> actor = Act02

    返回:
        file_results: [
            {"file": "...", "actor": "Act02", "entropy": 4.38},
            ...
        ]

        actor_results: [
            {
                "actor": "Act02",
                "num_files": 10,
                "mean_entropy": 4.41,
                "std_entropy": 0.12,
                "min_entropy": 4.20,
                "max_entropy": 4.60
            },
            ...
        ]

        meta: {
            "top_joints": [...],
            "num_bins": 100
        }
    """
    filepaths = list_bvh_files(folder, recursive=recursive)
    if len(filepaths) == 0:
        raise ValueError(f"在 {folder} 中没有找到 .bvh 文件")

    omega_cache, top_joints, bin_edges, joint_avg = collect_topk_global_info(
        filepaths=filepaths,
        top_k=top_k,
        num_bins=num_bins,
        strip_prefix=strip_prefix,
        dtype=dtype
    )

    file_results = []

    for path in filepaths:
        omega_dict = omega_cache[str(path)]

        values = []
        for joint_name in top_joints:
            if joint_name in omega_dict and omega_dict[joint_name].size > 0:
                values.append(omega_dict[joint_name])

        if len(values) == 0:
            H = np.nan
        else:
            values = np.concatenate(values, axis=0)
            H = shannon_entropy_from_values(
                values,
                bin_edges=bin_edges,
                log_base=log_base,
                normalized=normalized
            )

        actor_id = parse_actor_id(path.name)

        file_results.append({
            "file": path.name,
            "actor": actor_id,
            "entropy": None if np.isnan(H) else round(float(H), 6)
        })

    # 按 actor 分组
    actor_to_entropy = defaultdict(list)
    for item in file_results:
        if item["entropy"] is not None:
            actor_to_entropy[item["actor"]].append(item["entropy"])

    actor_results = []
    for actor, vals in sorted(actor_to_entropy.items()):
        vals = np.asarray(vals, dtype=np.float64)
        actor_results.append({
            "actor": actor,
            "num_files": int(len(vals)),
            "mean_entropy": round(float(np.mean(vals)), 6),
            "std_entropy": round(float(np.std(vals)), 6),
            "min_entropy": round(float(np.min(vals)), 6),
            "max_entropy": round(float(np.max(vals)), 6),
        })

    meta = {
        "top_joints": top_joints,
        "num_bins": num_bins,
        "joint_avg_angvel": {k: round(v, 6) for k, v in joint_avg.items()}
    }

    return file_results, actor_results, meta


def save_results_json(file_results, actor_results, meta, out_path="body_entropy_results.json"):
    out = {
        "file_results": file_results,
        "actor_results": actor_results,
        "meta": meta
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    folder = "../bvhs"

    file_results, actor_results, meta = compute_body_entropy_per_file(
        folder=folder,
        recursive=False,
        top_k=20,
        num_bins=100,
        normalized=False
    )

    print("=== Per-file entropy ===")
    for x in file_results[:10]:
        print(x)

    print("\n=== Grouped by actor ===")
    for x in actor_results:
        print(x)

    save_results_json(file_results, actor_results, meta, "body_entropy_results.json")
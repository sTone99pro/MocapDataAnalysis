from pathlib import Path
import numpy as np 
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
class bvhtools:
    def __init__(self,bvh_path):
        self.mypath = bvh_path
    #内部使用的加载bvh
    def load_bvh(self, filename):
        bvh_path = Path(self.mypath) / filename
        with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
            mocap = Bvh(f.read())
        return mocap

    def get_meta(self, filename):
    # filename 是文件名，例如 "Session_01_Take_012_Act02_M.bvh"
        mocap = self.load_bvh(filename)

        joints_raw = mocap.get_joints_names()
        joints = [j.split(":", 1)[1] if ":" in j else j for j in joints_raw]  # 去前缀

        num_frames = len(mocap.frames)
        frame_time = float(mocap.frame_time)
        fps = round(1.0 / frame_time)

        return {
            "file": filename,
            "num_joints": len(joints),
            "joints": joints,
            "num_frames": num_frames,
            "frame_time": frame_time,
            "fps": fps,
        }

    def get_motion(self, filename, as_dict=False, strip_prefix=True, dtype=np.float32):
        """
        返回 BVH 的 motion 数据

        参数:
            filename: 文件名
            as_dict: 
                False -> 返回整个 motion 矩阵 (T, D)
                True  -> 按 joint 拆成 dict
            strip_prefix:
                是否去掉 joint 名里的前缀，例如 "Act02_M:Hips" -> "Hips"
            dtype:
                numpy 数据类型，默认 float32

        返回:
            如果 as_dict=False:
                {
                    "file": filename,
                    "motion": np.ndarray,   # shape (T, D)
                    "columns": list[str],   # 每一列对应的名字
                    "num_frames": T,
                    "num_channels": D
                }

            如果 as_dict=True:
                {
                    "file": filename,
                    "motion_by_joint": {
                        joint_name: {
                            "channels": [...],
                            "values": np.ndarray  # shape (T, Cj)
                        },
                        ...
                    },
                    "num_frames": T
                }
        """
        mocap = self.load_bvh(filename)

        joints_raw = mocap.get_joints_names()

        # 整个 motion 数据，shape: (T, D)
        motion = np.asarray(mocap.frames, dtype=dtype)

        if not as_dict:
            columns = []
            for j_raw in joints_raw:
                j_name = j_raw.split(":", 1)[1] if (strip_prefix and ":" in j_raw) else j_raw
                chs = mocap.joint_channels(j_raw)
                for ch in chs:
                    columns.append(f"{j_name}_{ch}")

            return {
                "file": filename,
                "motion": motion,
                "columns": columns,
                "num_frames": motion.shape[0],
                "num_channels": motion.shape[1],
            }

        else:
            motion_by_joint = {}
            col_start = 0

            for j_raw in joints_raw:
                j_name = j_raw.split(":", 1)[1] if (strip_prefix and ":" in j_raw) else j_raw
                chs = mocap.joint_channels(j_raw)
                n_ch = len(chs)

                motion_by_joint[j_name] = { #第二层key
                    "channels": chs,
                    "values": motion[:, col_start:col_start + n_ch]
                }

                col_start += n_ch

            return {
                "file": filename,
                "motion_by_joint": motion_by_joint, #第一层key
                "num_frames": motion.shape[0],
            }
    
    def export_no_finger_bvh(self, filename, out_filename=None, dtype=np.float32):
        bvh_path = Path(self.mypath) / filename
        with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        mocap = Bvh(text)
        joints_raw = mocap.get_joints_names()
        motion = np.asarray(mocap.frames, dtype=dtype)
        lines = text.splitlines()

        finger_keys = ["thumb", "index", "middle", "ring", "pinky", "little", "finger"]

        keep_cols = []
        col_start = 0
        for j_raw in joints_raw:
            chs = mocap.joint_channels(j_raw)
            n_ch = len(chs)

            j_name = j_raw.split(":", 1)[1] if ":" in j_raw else j_raw
            if not any(k in j_name.lower() for k in finger_keys):
                keep_cols.extend(range(col_start, col_start + n_ch))

            col_start += n_ch

        motion_new = motion[:, keep_cols]

        motion_idx = lines.index("MOTION")
        hier_lines = lines[:motion_idx]
        motion_lines = lines[motion_idx:]

        new_hier = []
        skip = False
        depth = 0

        for line in hier_lines:
            s = line.strip()

            if not skip and s.startswith("JOINT "):
                raw_name = s[6:].strip()
                name = raw_name.split(":", 1)[1] if ":" in raw_name else raw_name
                if any(k in name.lower() for k in finger_keys):
                    skip = True
                    depth = 0
                    continue

            if skip:
                depth += line.count("{")
                depth -= line.count("}")
                if depth <= 0:
                    skip = False
                continue

            new_hier.append(line)

        header = motion_lines[:3]
        new_frames = [" ".join(f"{x:.6f}" for x in row) for row in motion_new]
        new_bvh_text = "\n".join(new_hier + header + new_frames) + "\n"

        if out_filename is None:
            out_filename = Path(filename).stem + "_nofinger.bvh"

        out_path = Path("bvhsnofingers") / out_filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_bvh_text)

        return {
            "file": filename,
            "motion": motion_new,
            "num_frames": motion_new.shape[0],
            "num_channels": motion_new.shape[1],
            "out_bvh": str(out_path),
        }
            
    def _layout(self, filename, strip_prefix=True):
        mocap = self.load_bvh(filename)
        joints_raw = mocap.get_joints_names()

        layout = []
        col_start = 0
        for j_raw in joints_raw:
            j_name = j_raw.split(":", 1)[1] if (strip_prefix and ":" in j_raw) else j_raw
            chs = mocap.joint_channels(j_raw)
            n_ch = len(chs)

            pos_idx = [i for i, c in enumerate(chs) if "position" in c.lower()]
            rot_idx = [i for i, c in enumerate(chs) if "rotation" in c.lower()]
            rot_order = "".join(chs[i][0].upper() for i in rot_idx)

            layout.append({
                "name": j_name,
                "channels": chs,
                "start": col_start,
                "end": col_start + n_ch,
                "pos_idx": pos_idx,
                "rot_idx": rot_idx,
                "rot_order": rot_order,
            })
            col_start += n_ch

        return layout


    def motion_to_bvh(self, template_filename, motion, out_filename, dtype=np.float32):
        bvh_path = Path(self.mypath) / template_filename
        with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        mocap = Bvh(text)
        motion = np.asarray(motion, dtype=dtype)
        if motion.shape[1] != len(mocap.frames[0]):
            raise ValueError(f"motion 通道数不匹配: got {motion.shape[1]}, expect {len(mocap.frames[0])}")

        lines = text.splitlines()
        motion_idx = lines.index("MOTION")
        hier_lines = lines[:motion_idx]
        frame_time_line = lines[motion_idx + 2]

        new_frames = [" ".join(f"{x:.6f}" for x in row) for row in motion]
        new_bvh_text = "\n".join(
            hier_lines +
            ["MOTION", f"Frames: {motion.shape[0]}", frame_time_line] +
            new_frames
        ) + "\n"

        out_path = Path(out_filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_bvh_text)

        return {
            "motion": motion,
            "num_frames": motion.shape[0],
            "num_channels": motion.shape[1],
            "out_bvh": str(out_path),
        }


    def euler_motion_to_rot6d(self, template_filename, motion, degrees=True, dtype=np.float32):
        layout = self._layout(template_filename)
        motion = np.asarray(motion, dtype=dtype)

        parts = []
        columns = []

        for info in layout:
            block = motion[:, info["start"]:info["end"]]

            joint_parts = []

            for i in info["pos_idx"]:
                joint_parts.append(block[:, i:i+1])
                columns.append(f'{info["name"]}_{info["channels"][i]}')

            if len(info["rot_idx"]) > 0:
                euler = block[:, info["rot_idx"]]
                mats = R.from_euler(info["rot_order"], euler, degrees=degrees).as_matrix()
                rot6d = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=1).astype(dtype, copy=False)
                joint_parts.append(rot6d)
                columns.extend([f'{info["name"]}_rot6d_{k}' for k in range(6)])

            if len(joint_parts) > 0:
                parts.append(np.concatenate(joint_parts, axis=1))

        motion6d = np.concatenate(parts, axis=1).astype(dtype, copy=False)

        return {
            "motion": motion6d,
            "columns": columns,
            "num_frames": motion6d.shape[0],
            "num_channels": motion6d.shape[1],
        }


    def rot6d_motion_to_euler(self, template_filename, motion6d, degrees=True, dtype=np.float32):
        layout = self._layout(template_filename)
        motion6d = np.asarray(motion6d, dtype=dtype)

        T = motion6d.shape[0]
        in_col = 0
        parts = []
        columns = []

        for info in layout:
            joint_out = np.zeros((T, len(info["channels"])), dtype=dtype)

            for i in info["pos_idx"]:
                joint_out[:, i:i+1] = motion6d[:, in_col:in_col+1]
                in_col += 1

            if len(info["rot_idx"]) > 0:
                a1 = motion6d[:, in_col:in_col+3]
                a2 = motion6d[:, in_col+3:in_col+6]
                in_col += 6

                b1 = a1 / np.clip(np.linalg.norm(a1, axis=1, keepdims=True), 1e-8, None)
                u2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
                b2 = u2 / np.clip(np.linalg.norm(u2, axis=1, keepdims=True), 1e-8, None)
                b3 = np.cross(b1, b2)

                mats = np.stack([b1, b2, b3], axis=2)
                euler = R.from_matrix(mats).as_euler(info["rot_order"], degrees=degrees).astype(dtype, copy=False)
                joint_out[:, info["rot_idx"]] = euler

            parts.append(joint_out)

            for ch in info["channels"]:
                columns.append(f'{info["name"]}_{ch}')

        motion_euler = np.concatenate(parts, axis=1).astype(dtype, copy=False)

        return {
            "motion": motion_euler,
            "columns": columns,
            "num_frames": motion_euler.shape[0],
            "num_channels": motion_euler.shape[1],
        }





    
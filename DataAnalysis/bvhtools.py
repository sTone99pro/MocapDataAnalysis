from pathlib import Path
import numpy as np 
from bvh import Bvh

class bvhtools:
    def init(self,bvh_path):
        self.mypath = bvh_path

    def get_meta(self, filename):
    # filename 是文件名，例如 "Session_01_Take_012_Act02_M.bvh"
        bvh_path = Path(self.mypath) / filename   # self.mypath 是目录

        with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
            mocap = Bvh(f.read())

        joints_raw = mocap.get_joints_names()
        joints = [j.split(":", 1)[1] if ":" in j else j for j in joints_raw]  # 去前缀

        num_frames = len(mocap.frames)
        frame_time = float(mocap.frame_time)
        fps = 1.0 / frame_time

        return {
            "file": filename,
            "num_joints": len(joints),
            "joints": joints,
            "num_frames": num_frames,
            "frame_time": frame_time,
            "fps": fps,
        }





    
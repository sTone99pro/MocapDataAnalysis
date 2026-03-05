#this is the tool box
from pathlib import Path
import re
import json

class Preprocessor:
    _re_st = re.compile(r"Session[_\-]?(\d+).*Take[_\-]?(\d+)", re.IGNORECASE)

    def __init__(self, wav_dir, bvh_dir):
        self.wav_dir = Path(wav_dir)
        self.bvh_dir = Path(bvh_dir)

    def count_bvhs(self):
        s = sum(1 for x in self.bvh_dir.iterdir() if x.is_file())
        print("there are {s} bvhs")
        return s
    def count_wavs(self):
        s = sum(1 for x in self.bvh_dir.iterdir() if x.is_file())
        print("there are {s} wavs")
        return s

    def _parse_st(self, name: str):
        m = self._re_st.search(name)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _index_one_level(self, folder: Path, pattern: str):
        idx = {}
        for p in folder.glob(pattern):  # 只一层
            key = self._parse_st(p.name)
            if key is None:
                continue
            idx.setdefault(key, []).append(p.name)  # 只存文件名
        for k in idx:
            idx[k].sort()
        return idx

    def find_quadruples(self, out_json: str, unmatched_json: str = "unmatched.json"):#可以获得配对的
        wav_idx = self._index_one_level(self.wav_dir, "*.wav")
        bvh_idx = self._index_one_level(self.bvh_dir, "*.bvh")

        quads = []
        unmatched = []

        # 以 bvh 为准：遍历所有 bvh 出现过的 (session,take)
        for st in sorted(bvh_idx.keys(), key=lambda x: (x[0], x[1])):
            wavs = wav_idx.get(st, [])
            bvhs = bvh_idx.get(st, [])

            if len(wavs) >= 2 and len(bvhs) >= 2:
                quads.append({
                    "session": st[0],
                    "take": st[1],
                    "wav0": wavs[0],
                    "wav1": wavs[1],
                    "bvh0": bvhs[0],
                    "bvh1": bvhs[1],
                })
            else:
                unmatched.append({
                    "session": st[0],
                    "take": st[1],
                    "num_wav": len(wavs),
                    "num_bvh": len(bvhs),
                    "wavs": wavs,
                    "bvhs": bvhs,
                })

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(quads, f, ensure_ascii=False, indent=2)

        with open(unmatched_json, "w", encoding="utf-8") as f:
            json.dump(unmatched, f, ensure_ascii=False, indent=2)

        return {"saved_quads": len(quads), "unmatched_groups": len(unmatched)}

    #返回所有文件名
    def list_allfiles(self,folder):
        return [f.name for f in Path(folder).iterdir() if f.is_file()]
    def list_bvhs(self,folder = self.bvh_dir):
        return [f.name for f in Path(folder).iterdir() if f.is_file()]
    def list_wavs(self,folder = self.wav_dir):
        return [f.name for f in Path(folder).iterdir() if f.is_file()]
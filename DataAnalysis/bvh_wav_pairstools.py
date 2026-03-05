#this is the tool box
class Preprocessor:

    def __init__(self, wav_dir, bvh_dir):
        self.wav_dir = wav_dir
        self.bvh_dir = bvh_dir

    def parse_key(self, name):
        import re
        m = re.search(r"Session_(\d+)_Take_(\d+)", name)
        return int(m.group(1)), int(m.group(2))

    def build_index(self, files):
        index = {}
        for f in files:
            key = self.parse_key(f.name)
            index.setdefault(key, []).append(f)
        return index

    def build_pairs(self):

        from pathlib import Path

        wavs = list(Path(self.wav_dir).glob("*.wav"))
        bvhs = list(Path(self.bvh_dir).glob("*.bvh"))

        wav_index = self.build_index(wavs)
        bvh_index = self.build_index(bvhs)

        pairs = []

        for key in bvh_index:
            if key in wav_index:
                if len(bvh_index[key]) == 2 and len(wav_index[key]) == 2:
                    pairs.append((key,
                                  wav_index[key],
                                  bvh_index[key]))

        return pairs
from pathlib import Path

import scipy.io

from . import config


VAL_ROOT = Path(config["paths"]["val_root"])
VAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
META_PATH = Path(config["paths"]["meta_path"])
META_PATH.parent.mkdir(parents=True, exist_ok=True)
LABEL_FILE = Path(config["paths"]["label_file"])
LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(config["paths"]["val_path"])
OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

meta = scipy.io.loadmat(META_PATH, squeeze_me=True)
ilsvrc2012_id_to_wnid = {m[0]: m[1] for m in meta["synsets"]}

with LABEL_FILE.open("r") as f:
    labels = [int(line.strip()) for line in f.readlines()]

OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

for i, label in enumerate(labels):
    filename = VAL_ROOT / f"ILSVRC2012_val_{str(i+1).zfill(8)}.JPEG"
    wnid = ilsvrc2012_id_to_wnid[label]
    class_dir = OUTPUT_DIR / wnid
    class_dir.parent.mkdir(parents=True, exist_ok=True)

    filename.rename(class_dir / filename.name)

print("Validation images have been organized into class-specific folders.")

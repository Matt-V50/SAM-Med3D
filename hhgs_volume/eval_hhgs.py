"""
SAM-Med3D on HHGS cases with the HHGS clicks (2026-11-13). Same procedure as the r2 baseline script
(eval_seg_3D_sam_med3d.py of the r2 rebuttal): per click a 128^3 window centred on the click (zero-padded outside the
volume, no resampling: the model's native input), ZNormalization over the voxels > 0 inside the window (its
training convention), one positive point at the window centre (utils/infer_utils.py sam_model_infer without GT
clicks the centre), sigmoid > 0.5, the window pasted back on the grid. The image is the HHGS volume vol_<vol>.npy
(rc_v by default, raw values, as nnInteractive). Scored by evaluating/baselines/common.py of the hhgs repo (Dice
with the clicked GT node over the valid voxels).

usage:  python hhgs_volume/eval_hhgs.py --case <DATA_ROOT case> --out <OUT_ROOT case>   (environment sam_med3d)
output: <out>/baselines/sam_med3d.json
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # the SAM-Med3D repo: utils.infer_utils
import medim  # noqa: E402
from utils.infer_utils import sam_model_infer  # noqa: E402

spec = importlib.util.spec_from_file_location("hhgs_common", HERE.parents[2] / "evaluating/baselines/common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

CKPT = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
ROI = 128


def window(center, shape, extent=ROI):
    src, dst = [], []
    for c, n in zip(center, shape):
        lo = int(c) - extent // 2
        s_lo, s_hi = max(lo, 0), min(lo + extent, int(n))
        src.append(slice(s_lo, s_hi))
        dst.append(slice(s_lo - lo, s_hi - lo))
    return tuple(src), tuple(dst)


def main(case, out):
    clicks = common.Clicks(out)
    vol = torch.as_tensor(np.load(Path(case) / f"vol_{clicks.vol}.npy").astype(np.float32), device="cuda")
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=CKPT).cuda().eval()
    dice, t0 = [], time.time()
    for i in range(len(clicks)):
        src, dst = window(clicks.xyz[i], vol.shape)
        roi = torch.zeros((ROI,) * 3, device="cuda")
        roi[dst] = vol[src]
        roi = roi[None, None]
        m = roi > 0
        if m.any():
            v = roi[m]
            sd = v.std()
            roi = (roi - v.mean()) / (sd if float(sd) >= 1e-8 else torch.ones_like(sd))
        pred, _ = sam_model_infer(model, roi, roi_gt=None, num_clicks=1)
        full = torch.zeros(clicks.shape, dtype=torch.bool, device="cuda")
        full[src] = torch.as_tensor(pred, device="cuda")[dst] > 0.5
        dice.append(clicks.dice(i, full))
    res = clicks.write_result("sam_med3d", dice, seconds=round(time.time() - t0, 1), roi=ROI, ckpt=CKPT)
    print(f"{case}: leaf {res['leaf']}  own {res['own']:.4f}  (n {len(dice)})  [{res['seconds']} s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="+", required=True)
    ap.add_argument("--out", nargs="+", required=True)
    a = ap.parse_args()
    for c, o in zip(a.case, a.out):
        main(c, o)

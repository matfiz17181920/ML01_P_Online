import cv2
import numpy as np
from kornia.feature import LoFTR
import torch
from LoFTR.src.loftr import LoFTR, default_cfg
from copy import deepcopy
from LoFTR.src.utils.plotting import make_matching_figure
import matplotlib.cm as cm

_default_cfg = deepcopy(default_cfg)
_default_cfg['coarse']['temp_bug_fix'] = True
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("indoor_ds_new.ckpt")['state_dict'])
matcher = matcher.eval().cpu()
# Load example images
img0_pth = "IMG_1.JPG"
img1_pth = "IMG_2.JPG"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (640, 880))
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = torch.from_numpy(img0_raw)[None][None].cpu() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cpu() / 255.
batch = {'image0': img0, 'image1': img1}

with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
print(fig)
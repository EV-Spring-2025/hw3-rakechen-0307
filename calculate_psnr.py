import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = np.max(img1)
    return 20 * np.log10(pixel_max / np.sqrt(mse))

parser = argparse.ArgumentParser(description="Calculate and plot PSNR of a video")
parser.add_argument("--target_path", required=True, type=str)
parser.add_argument("--gt_path", required=True, type=str)
parser.add_argument("--save_plot", type=str, default=None, help="Optional path to save the plot image")

args = parser.parse_args()
imgs = sorted(glob.glob(os.path.join(args.target_path, "*.png")))
gt_imgs = sorted(glob.glob(os.path.join(args.gt_path, "*.png")))
assert len(imgs) == len(gt_imgs), "Mismatch in number of frames."

psnrs = []
for i in range(len(imgs)):
    img = cv2.imread(imgs[i], cv2.IMREAD_COLOR)
    gt = cv2.imread(gt_imgs[i], cv2.IMREAD_COLOR)
    psnr = calculate_psnr(gt, img)
    psnrs.append(psnr)

plt.figure(figsize=(10, 5))
plt.plot(psnrs, marker='o', linestyle='-', color='blue')
plt.title("PSNR over Frames")
plt.xlabel("Frame Index")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.tight_layout()

if args.save_plot:
    plt.savefig(args.save_plot)
    print(f"PSNR plot saved to {args.save_plot}")
else:
    plt.show()
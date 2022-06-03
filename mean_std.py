import argparse
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--dataroot', type=str)
parser.add_argument('--output_dir', type=str, default='.')
args = parser.parse_args()

imageFilesDir = Path(args.dataroot)
files = list(imageFilesDir.rglob('*/*.png'))

mean = np.array([0.,0.,0.])
stdTemp = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])

numSamples = len(files)

print('Calculate mean...')

for i in tqdm(range(numSamples)):
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    
    for j in range(3):
        mean[j] += np.mean(im[:,:,j])

mean = (mean/numSamples)

print('Calculate std...')

for i in tqdm(range(numSamples)):
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    for j in range(3):
        stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

std = np.sqrt(stdTemp/numSamples)

print('Mean:', mean)
print('Standard deviation:', std)

with open(Path(args.output_dir) / 'mean_std.txt', 'w') as f:
    f.write(f"dataroot: {args.dataroot}\n")
    f.write(f"mean: {mean}\n")
    f.write(f"std: {std}\n")

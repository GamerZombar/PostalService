import argparse
import pickle

import cv2
import lmdb
from path import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, required=True)
args = parser.parse_args()


assert not (args.data_dir / 'lmdb').exists()

env = lmdb.open('lmdb', map_size=1024 * 1024 * 1024 * 2)  # 2gb
imgs = (args.data_dir / 'images').walkfiles('*.png')
with env.begin(write=True) as conn:
     for idx, img in enumerate(imgs):
         read_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
         bd_filename = '/'.join(str(img).split('/')[-3:]).encode("ascii")
         conn.put(bd_filename, pickle.dumps(read_img))
env.close()

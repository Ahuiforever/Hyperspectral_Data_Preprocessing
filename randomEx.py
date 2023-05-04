'''
Author: your name
Date: 2021-12-13 15:25:06
LastEditTime: 2021-12-13 16:19:17
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \Python-universe\hyperspectral band extraction\randomEx.py
'''
import glob
import math
import os
import shutil

import numpy as np
from tqdm import tqdm


def path_check(dstpath):
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)
        os.makedirs(dstpath)
    else:
        os.makedirs(dstpath)


def extract(srcpath, dstpath, ex_rate):
    files = []
    os.chdir(srcpath)
    path_check(dstpath)
    for file in glob.glob('*.tif'):
        files.append(file)

    l = len(files)

    e = math.ceil(l * ex_rate)

    files_ = np.random.choice(files, e, replace=False)

    for file_ in files_:
        src = os.path.join(srcpath, file_)
        dst = os.path.join(dstpath, file_)
        shutil.move(src, dst)


if __name__ == '__main__':
    bands = ['450']
    srcpath = r'F:\works\ENVI_items\try7\540'
    dstpath = r'F:\works\ENVI_items\try7\540\images\val'
    ex_rate = 0.4

    for band in tqdm(bands):
        # srcpaths = os.path.join(srcpath, band)
        # dstpaths = os.path.join(dstpath, band)
        extract(srcpath, dstpath, ex_rate)

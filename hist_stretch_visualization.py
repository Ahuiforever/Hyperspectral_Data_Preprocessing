# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/13 9:24
# @Author  : Ahuiforever
# @File    : hist_stretch_visualization.py
# @Software: PyCharm

"""
Visualize accumulative histogram of single image to test different gray stretching methods.
"""
import os
from typing import Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from extract_stretch_save import Tif, Processor


class Visual:
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self._show = {}
        self._imshow = []
        self._title = []
        self._hists = []

    def add(self, _key: str, _value: Union[int, float, np.ndarray]) -> None:
        self._show[_key] = _value

    def plot(self):
        # - All the images will be transformed to uint8.
        # todo 2: Complete this method.
        for _key, _value in zip(self._show.keys(), self._show.values()):
            self._imshow.append(np.hstack((self.arr, _value)))
            self._title.append(_key)
            self._hists.append(_value.flatten())

    def save(self, filename, save_path: str = "./results"):
        os.mkdir(save_path) if not os.path.exists(save_path) else None
        for plt_imshow, plt_title, plt_hist in zip(
            self._imshow, self._title, self._hists
        ):
            plt.title(plt_title)
            plt.imshow(plt_imshow, cmap="gray", vmin=0, vmax=255)
            plt.savefig(f"{save_path}/{filename}_{plt_title}.jpg", dpi=300)
            plt.clf()
            plt.title(plt_title)
            plt.hist(plt_hist, bins=255, color="blue", alpha=0.7)
            plt.savefig(f"{save_path}/{filename}_{plt_title}_hist.jpg", dpi=300)
            plt.clf()

    def show(self):
        for plt_imshow, plt_title, plt_hist in zip(
            self._imshow, self._title, self._hists
        ):
            plt.title(plt_title)
            plt.imshow(plt_imshow, cmap="gray", vmin=0, vmax=255)
            plt.show()
            plt.title(plt_title)
            plt.hist(plt_hist, bins=255, color="blue", alpha=0.7)
            plt.show()


class Operator(Processor):
    def __init__(self, arr, data_type):
        super().__init__(data_type)
        self.arr = arr

    def hist_equ(self):
        return super()._hist_equ(self.arr).squeeze()

    def optim_stretch(self):
        return super()._optim_stretch(self.arr).squeeze()

    def clahe_hist_equ(self):
        return super()._clahe_hist_equ(self.arr).squeeze()

    def linear_stretch(self):
        return super()._linear_stretch(self.arr).squeeze()

    def percentage_linear_stretch(self):
        return super()._percentage_linear_stretch(self.arr).squeeze()


if __name__ == "__main__":
    tif = Tif(filename=r".\test\D_2021_10_30_09-31-51.img", data_type=np.uint8)
    tif.bands()
    for band_name, band_array in tif:
        operator = Operator(band_array, np.uint8)
        visual = Visual(Operator.trans2_uint8(band_array))
        visual.add("linear_stretch", operator.linear_stretch())
        visual.add("percentage_linear_stretch", operator.percentage_linear_stretch())
        visual.add("optim_stretch", operator.optim_stretch())
        visual.add("hist_equ", operator.hist_equ())
        visual.add("clahe_hist_equ", operator.clahe_hist_equ())
        visual.plot()
        # visual.show()
        visual.save(os.path.split(tif.img_file)[-1].split(".")[0] + '_' + band_name)
        # >>> D_2021_10_30_09-31-51_450
    print("=======================All Done.=======================")
    print(f"Got {len(tif.band_arrays)} bands from {tif.img_file}.")

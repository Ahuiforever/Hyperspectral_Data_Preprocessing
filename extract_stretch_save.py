# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 15:39
# @Author  : Ahuiforever
# @File    : extract_stretch_save.py
# @Software: PyCharm

import glob
import os
from typing import Optional, Union

import cv2

import numpy as np
from osgeo import gdal
from wjh.utils import LogWriter
from tqdm import tqdm


class Tif:
    def __init__(self, filename: str, data_type: np.dtype):
        self.img_file = filename
        self.ds = gdal.Open(self.img_file)
        self.band_names = []
        self.band_arrays = []

        if data_type is np.uint8:
            self.gdal_dtype = gdal.GDT_Byte
        elif data_type is np.uint16:
            self.gdal_dtype = gdal.GDT_UInt16
        else:
            raise AttributeError(
                f"Class 'Tif' has no corresponding osgeo.gdal attribute for '{data_type}'."
            )
        # ! No matter what the data type is, this class always read the '*.img' file with dtype of np.uint16.
        # ! Go to the self.bands() for more details.

    def _filename(self) -> str:
        # >>> E:\Work\Hyperspectral Images\D_2021_10_30_09-31-51.img
        return self.img_file

    def _count(self) -> int:
        return self.ds.RasterCount

    def _width(self) -> int:
        return self.ds.RasterXSize

    def _height(self) -> int:
        return self.ds.RasterYSize

    def _geotrans(self) -> tuple:
        return self.ds.GetGeoTransform()

    def _projection(self) -> str:
        return self.ds.GetProjection()

    def bands(self) -> None:
        for _ in range(1, self._count() + 1):
            band = self.ds.GetRasterBand(_)
            self.band_names.append(band.GetDescription()[-15:-12])
            self.band_arrays.append(band.ReadAsArray())
        self.band_arrays = np.array(self.band_arrays, dtype=np.uint16)
        # >>> self.band_arrays.shape = (self.count, self.height, self.width)

    def __getitem__(self, item: int) -> tuple:
        # ! Before calling __getitem__ method, self.bands must be called.
        return self.band_names[item], self.band_arrays[item]

    def save(self, filetype: str = "tif") -> None:
        """
        @ filetype: tif, jpg, png, tiff
        Args:
            filetype (str):  The filetype to save.

        Returns:
            No return.
        """
        # // todo 1: Complete method of saving array to certain file.
        suffix = "." + filetype
        dir_path, img_name = os.path.split(self.img_file)
        # >>> save_path = E:\Work\Hyperspectral Images, file_name = D_2021_10_30_09-31-51.img

        for band_array, band_name in zip(self.band_arrays, self.band_names):
            save_dir = dir_path + f"/{band_name}"
            os.mkdir(save_dir) if not os.path.exists(save_dir) else None
            file_name = f"{band_name}_" + img_name.replace(".img", suffix)
            save_name = os.path.join(save_dir, file_name)
            # >>> E:\Work\Hyperspectral Images\450\450_D_2021_10_30_09-31-51.tif

            self._save(save_name, band_array)

    def _save(self, save_name: str, band_array: np.ndarray) -> None:
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(save_name, self._width(), self._height(), 1, self.gdal_dtype)
        ds.SetProjection(self._projection())
        ds.SetGeoTransform(self._geotrans())
        ds.GetRasterBand(1).WriteArray(band_array)
        ds.FlushCache()
        del ds


class Processor:
    def __init__(self, data_type: np.dtype):
        self.img_dir = None
        self.log = LogWriter("extract_stretch_save.txt")
        self.tif = []
        self.data_type = data_type
        # * Get the maximum value of the input data type.
        self.max_value = np.iinfo(self.data_type).max

    def get_img_list(self, directory: str) -> None:
        self.img_dir = directory
        # >>> E:\Work\Hyperspectral Images
        for img_file in glob.glob(f"{self.img_dir}/*.img"):
            hdr_file = img_file.replace(".img", ".hdr")
            if os.path.exists(hdr_file):
                self._read_img(
                    img_file
                )  # ? This may be able to be replaced by source code.
            else:
                self.log(f"{hdr_file} is missing.", printf=True)

    def _read_img(self, img_file: str) -> None:
        self.tif.append(Tif(img_file, self.data_type))

    def stretch(self, stretch_mode: int = 0) -> None:
        """To perform corresponding stretch on a given image.
        @ -1: No stretch
        @  0: Optimized stretch
        @  1: histogram equalization
        @  2: CLAHE histogram equalization
        @  3: linear stretch
        @  4: percentage linear truncation stretch

        Args:
            stretch_mode (int): The method you would like to stretch the histogram.

        Returns:
            No return.
        """

        for tif in tqdm(self.tif):
            tif.bands()
            if stretch_mode == -1:
                tif.band_arrays = self._no_stretch(tif.band_arrays)
            elif stretch_mode == 0:
                tif.band_arrays = self._optim_stretch(tif.band_arrays)
            elif stretch_mode == 1:
                tif.band_arrays = self._hist_equ(tif.band_arrays)
            elif stretch_mode == 2:
                tif.band_arrays = self._clahe_hist_equ(tif.band_arrays)
            elif stretch_mode == 3:
                tif.band_arrays = self._linear_stretch(tif.band_arrays)
            elif stretch_mode == 4:
                tif.band_arrays = self._percentage_linear_stretch(
                    tif.band_arrays, percentage=2
                )
            tif.save(filetype="tif")

    def _optim_stretch(self, arr: np.ndarray) -> np.ndarray:
        """
        ` minimum percentile: default value 0.025
        ` maximum percentile: default value 0.99
        ` minimum adjustment percentage: default value 0.1
        ` maximum adjustment percentage: default value 0.5
        - a, b are the data values corresponding to the minimum and maximum percentages in the relative cumulative
        - histogram, respectively.
        * black point: c = a - 0.1 * (b - a)
        * white point: d = b + 0.5 * (b - a)
        """

        # * Determine the 2.5% and 99% percentile, denoted by a and b, respectively.
        a, b = np.percentile(
            arr, (2.5, 99), axis=(-2, -1)
        )  # Along the last 2 dimensions.
        # ? In oder to calculate the percentile for every channel separately, axis is assigned to (-2, -1),
        # ? which means that percentile is calculated along the tif.band_arrays.shape[-2] = 1920 and
        # ? tif.band_arrays.shape[-1] = 1080.

        # Calculate the black and white points, denoted by c and b, respectively.
        c = (a - 0.1 * (b - a)).reshape(-1, 1, 1)
        d = (b + 0.5 * (b - a)).reshape(-1, 1, 1)
        # >>> shape = (self.count, 1, 1)
        # ? The points values are reshaped to match tif.band_arrays.

        # The pixel values between the black and white points are linearly stretched to a range of 0-65535.
        arr = (arr - c) / (d - c) * self.max_value

        # Pixel values greater than white point are assigned to 65535, and the ones smaller than black to 0.
        arr = np.clip(arr, 0, self.max_value).astype(self.data_type)
        return arr

    def _linear_stretch(self, arr: np.ndarray) -> np.ndarray:
        a = arr.min(axis=(-2, -1)).reshape(-1, 1, 1)
        b = arr.max(axis=(-2, -1)).reshape(-1, 1, 1)
        arr = (arr - a) / (b - a) * self.max_value
        arr = np.clip(arr, 0, self.max_value).astype(self.data_type)
        return arr

    def _percentage_linear_stretch(
        self, arr: np.ndarray, percentage: Optional[Union[int, float]] = 2
    ) -> np.ndarray:
        a, b = np.percentile(arr, (percentage, 100 - percentage), axis=(-2, -1))
        a, b = a.reshape(-1, 1, 1), b.reshape(-1, 1, 1)
        arr = (arr - a) / (b - a) * self.max_value
        arr = np.clip(arr, 0, self.max_value).astype(self.data_type)
        return arr

    def _hist_equ(self, arr: np.ndarray) -> np.ndarray:
        # // todo 2: Complete method of histogram equalization.
        # - Histogram equalization requires uint8 data type.
        arr = self.trans2_uint8(arr)
        arr = (
            cv2.equalizeHist(arr)
            if len(arr.shape) == 2
            else np.reshape(
                [cv2.equalizeHist(arr[channel]) for channel in range(arr.shape[0])],
                arr.shape,
            )
        )
        # ? Set dtype to np.uint8 to make sure the result is saved correctly.
        self.data_type = np.uint8
        return arr

    def _clahe_hist_equ(self, arr: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization, CLAHE
        ? 一种称为CLAHE（Contrast Limited Adaptive Histogram Equalization）的方法，它是直方图均衡的一种变体，
        ? 具有自适应性和对比度限制。CLAHE在每个小块（称为“tiles”）中独立应用直方图均衡，然后通过插值将这些均衡化的小块合并。
        ? CLAHE还引入了对比度限制，以防止过度增强对比度，这使得该方法适用于各种图像。

        @ 1. clipLimit：
        * clipLimit 控制对比度的限制。它定义了在每个小块（tile）中对比度的最大增加量。如果某个小块的直方图均衡化导致对比度的增加超过了
        * clipLimit，那么对该块的像素值将会被截断，以限制对比度的过度增加。通常，clipLimit 的选择需要在试验中找到一个合适的值。
        * 较小的值会导致更保守的对比度增强，而较大的值可能导致过度增强。经验上，常见的起始值可以选择在1.0到3.0之间。

        @ 2. tileGridSize：
        * tileGridSize 定义了图像被分成的小块（tiles）的大小。CLAHE独立地在每个小块上进行直方图均衡化，然后通过插值将这些小块合并。
        * 因此，tileGridSize 影响了均衡化的局部性。通常，tileGridSize 的选择也需要在试验中找到一个合适的值。
        * 较小的值会导致更细粒度的均衡化，而较大的值可能导致过度平滑。典型的值可能在(4, 4)到(16, 16)的范围内选择。
        """
        arr = self.trans2_uint8(arr)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = (
            clahe.apply(arr)
            if len(arr.shape) == 2
            else np.reshape(
                [clahe.apply(arr[channel]) for channel in range(arr.shape[0])],
                arr.shape,
            )
        )
        self.data_type = np.uint8
        return arr

    def _no_stretch(self, arr: np.ndarray) -> np.ndarray:
        arr = self.trans2_uint8(arr) if self.data_type is np.uint8 else arr
        return arr.astype(self.data_type)

    @staticmethod
    def trans2_uint8(arr: np.ndarray) -> np.ndarray:
        arr = arr / 65535.0 * 255.0
        # arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.
        return arr.astype(np.uint8)


if __name__ == "__main__":
    processor = Processor(data_type=np.uint16)
    processor.get_img_list(directory=r".\test")
    processor.stretch(stretch_mode=0)

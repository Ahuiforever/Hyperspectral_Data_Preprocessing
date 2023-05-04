"""
This Python module is used to implement pre-processing of the hyperspectral dataset.
Including band extraction, gray level stretch, format conversion and classification of different bands.
"""

import glob
import os
import shutil
import sys

# from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from osgeo import gdal
from tqdm import tqdm


def file_obtain(dir_path: str) -> dict:
    # * dir_path = F:\works\ENVI_items\try9-dataset2.0
    os.chdir(dir_path)
    file_dict = {}
    for file in glob.glob(r'*.img'):
        file_path = os.path.join(dir_path, file)
        # * >>> F:\works\ENVI_items\try9-dataset2.0\D_2022_05_12_15-51-14.img
        file_name = file.split('.i')[0]
        # * >>> D_2022_05_12_15-51-14
        if os.path.isfile(file_name + '.hdr'):
            file_dict[file_name] = file_path
        else:
            print(f'\033[1;34m{file_name}.hdr is missing.\033[0m')
    sys.exit(0) if file_dict == {} else None
    return file_dict


def stretch(band_array: np.ndarray, stretch_mode: int = 0) -> np.ndarray:
    """
    :rtype: np.ndarray
    """
    # - |0: linear 2% |1: std stretch |2: exponential transform |3: log transform |4:...
    if stretch_mode == 0:
        truncated_value = 2
        max_out = 65535
        min_out = 0
        down = np.percentile(band_array, truncated_value)
        up = np.percentile(band_array, 100 - truncated_value)
        band_array = (band_array - down) / (up - down) * (max_out - min_out) + min_out
        band_array = np.clip(band_array, min_out, max_out)
        band_array = np.uint16(band_array)
    elif stretch_mode == 1:
        mean = np.mean(band_array)
        std = np.std(band_array)
        vmin = mean - std * 2
        vmax = mean + std * 2
        band_array = np.clip(band_array, vmin, vmax)
        band_array = np.array((band_array - vmin) / (vmax - vmin) * 255, dtype=np.uint16)
    elif stretch_mode == 2:
        epsilon = 0.
        gamma = 1.5
        band_array = np.clip(band_array, 0, 65535)
        band_array = np.divide(band_array, 65535)
        band_array = np.array(np.power(band_array + epsilon, gamma) * 65535, dtype=np.uint16)
    elif stretch_mode == 3:
        c = 1.5
        band_array = np.clip(band_array, 0, 65535)
        band_array = np.divide(band_array, 65535)
        band_array = np.array(np.log(1 + band_array) * c * 65535, dtype=np.uint16)
    return band_array


def extract(img_name: str, img_file: str) -> dict:
    """
    :rtype: dict
    """
    # * img_file = F:\works\ENVI_items\try9-dataset2.0\D_2022_05_12_15-51-14.img
    # * img_name = D_2022_05_12_15-51-14
    ds = gdal.Open(img_file)
    width = ds.RasterXSize
    height = ds.RasterYSize
    num: int = ds.RasterCount
    projection = ds.GetProjection()
    geotransform = ds.GetGeoTransform()
    band_dict = {'name': img_name,
                 'num': num,
                 'width': width,
                 'height': height,
                 'projection': projection,
                 'geotransform': geotransform}
    for n in range(1, num + 1):
        band = ds.GetRasterBand(n)
        band_name = band.GetDescription()[-15:-12]
        band_array = band.ReadAsArray()
        band_dict[band_name] = band_array
    return band_dict


def save(dir_path: str, band_dict: dict, suffix: str = '.tif') -> None:
    """
    :rtype: None
    """
    name = band_dict['name']
    num = band_dict['num']
    width = band_dict['width']
    height = band_dict['height']
    projection = band_dict['projection']
    geotransform = band_dict['geotransform']
    band_names = list(band_dict.keys())[6:]
    band_arrays = list(band_dict.values())[6:]
    supported_types = ['.tif', '.jpg', '.png', '.tiff']
    for n in range(num):
        driver = gdal.GetDriverByName('GTiff')
        save_dir = os.path.join(dir_path, band_names[n])
        save_path = os.path.join(save_dir, str(band_names[n]) + '_' + name + suffix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dataset = driver.Create(save_path[:-len(suffix)] + '.tif', width, height, 1, gdal.GDT_UInt16)
        if dataset is not None:
            dataset.SetProjection(projection)
            dataset.SetGeoTransform(geotransform)
        dataset.GetRasterBand(1).WriteArray(band_arrays[0])
        dataset.FlushCache()
        del dataset
        if suffix in supported_types and suffix != '.tif':
            # // dataset = driver.CreateCopy(save_path, dataset, strict=1, options=["TILED=YES", "COMPRESS=LZW"])
            img = Image.open(save_path[:-len(suffix)] + '.tif').copy()
            img.save(save_path, quality=100)
            os.remove(save_path[:-len(suffix)] + '.tif')
        elif suffix not in supported_types:
            print(f'Suffix \033[1;31m*{suffix}\033[0m not supported')
            del dataset
            shutil.rmtree(save_dir)
            sys.exit(0)


if __name__ == '__main__':
    dirpath = r'F:\works\ENVI_items\try9-dataset2.0'
    filedict = file_obtain(dirpath)
    for filename, filepath in tqdm(filedict.items(), total=len(filedict)):
        banddict = extract(filename, filepath)
        bandnames = list(banddict.keys())[6:]
        bandarrays = list(banddict.values())[6:]
        num = banddict['num']
        for n in range(num):
            bandarray = stretch(bandarrays[n], 0)
            banddict[bandnames[n]] = bandarray
        save(dirpath, banddict, '.tif')

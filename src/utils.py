# for general purposes
import os, sys
from glob import glob
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx
from branca.element import MacroElement
from datetime import date, datetime
import pickle
import gc

#import requests
#import json
#import getpass
#import re
#import branca
#import time
#from sklearn.preprocessing import MinMaxScaler
#import matplotlib.ticker as ticker

import warnings
warnings.filterwarnings('ignore')

# for geographic data
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio.mask import mask
import rioxarray
import folium
from folium.plugins import DualMap

#from folium import Choropleth
#from shapely.geometry import box

# for processing Landsat
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import earthpy.mask as em

# for Google Earth Engine
import ee

# for torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch.autograd import detect_anomaly
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import tempfile

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    return device

def crop_images(year: int,
                mesh_num: int,
                key_id: int,
                gdf,
                mesh_dir='../data/census/mesh',
                land_dir='../data/Landsat',
                ntl_dir ='../data/VIIRS/GeoTiff',
                all_dir ='../data/all_combined',
                size=224,
                mode=None
               ):
    """
    A function of cropping images with the extent of mesh.

    Args:
    year               : An integer indicating the mesh of census boundary.
    mesh_num           : An integer indicating the number of mesh extent.
    key_id             : An integer indicating the id of mesh cell.
    gdf                : An GeoDataDrame object which contains vector data of aggregated mesh and demographics.
    mesh_dir           : The directory of mesh boundary. By default, it is '../data/census/mesh'.
    land_dir           : The directory of landsat images. By default, it is '../data/Landsat'.
    ntl_dir            : The directory of VIIRS images. By default, it is '../data/VIIRS/GeoTiff'.
    all_dir            : The directory of final output. By default, it is '../data/all_combined'.
    size               : An integer of final output pixel size. By default, it is 224.
    mode               : An string indicating either 'save' or 'return' the output.

    Returns:
        stacked_layers : A np.ndarray object of stacked layers from Landsat-8 and VIIRS.
    """

    # obtain the geometry of mesh cell
    try:
        key_code = gdf.iloc[key_id]['KEY_CODE']
        geom = [gdf.geometry[key_id]]
    except:
        key_code = gdf['KEY_CODE']
        geom = [gdf.geometry]

    # create 'tmp' directory if it does not exist
    if not os.path.exists(os.path.join(all_dir, 'tmp')):
        os.mkdir(os.path.join(all_dir, 'tmp'))

    # crop Landsat-8 (Daytime image)
    with rasterio.open(os.path.join(land_dir, str(year), str(mesh_num)+'.tif')) as img_lands:
        lands_cropped, lands_transf = mask(img_lands, geom, crop=True, all_touched=True)
        lands_meta = img_lands.meta

    # update the metadata
    lands_meta.update({"driver": "GTiff",
                       "height": lands_cropped.shape[1],
                       "width": lands_cropped.shape[2],
                       "transform": lands_transf})

    # store the cropped Landsat-8/OLI
    with rasterio.open(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'), "w", **lands_meta) as dest:
        dest.write(lands_cropped)

    # crop Suomi NPP/VIIRS-DNS
    with rasterio.open(os.path.join(ntl_dir, 'processed', str(year), str(year)+'.tif')) as img_viirs:
        viirs_cropped, viirs_transf = mask(img_viirs, geom, crop=True, all_touched=True)
        viirs_meta = img_viirs.meta

    # update the metadata
    viirs_meta.update({"driver": "GTiff",
                       "height": viirs_cropped.shape[1],
                       "width": viirs_cropped.shape[2],
                       "transform": viirs_transf})

    # store the cropped Suomi NPP/VIIRS-DNS
    with rasterio.open(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'), "w", **viirs_meta) as dest:
        dest.write(viirs_cropped)

    # open the cropped GeoTIFF files
    xds_lands = rioxarray.open_rasterio(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
    xds_viirs = rioxarray.open_rasterio(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

    # reproject the Suomi NPP/VIIRS-DNS to Landsat-8/OLI for adjusting resolution
    xds_viirs = xds_viirs.rio.reproject_match(xds_lands)

    # resize pixels
    xds_viirs = xds_viirs.rio.reproject('EPSG:4326', shape=(size,size))
    xds_lands = xds_lands.rio.reproject('EPSG:4326', shape=(size,size))

    if mode == 'save':
        # save the reprojected VIIRS into GeoTIFF file
        xds_lands.rio.to_raster(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
        xds_viirs.rio.to_raster(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

    elif mode == 'return':
        # stack the layers into a single np.ndarray object
        stacked_layers = np.vstack([xds_lands.values, xds_viirs.values])

        # delete tmp files
        os.remove(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
        os.remove(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

        return stacked_layers
    
def append_index_layers(image: np.ndarray):
    """
    A function of appending layers of three indices (NDWI, NDBI, NDVI).

    Args:
        image: A np.ndarray object containing 7 layers of Landsat-8 and 1 layer of Suomi NPP.

    Returns:
        image: A np.ndarray object which additionally contains index layers
    """

    # check whether the image has proper number of layers or not
    assert image.shape[0] == 8

    # store relevant bands
    band_3 = image[2,:,:]
    band_4 = image[3,:,:]
    band_5 = image[4,:,:]
    band_6 = image[5,:,:]

    # calculate indices
    NDBI = (band_6 - band_5) / (band_6 + band_5)
    NDVI = (band_5 - band_4) / (band_5 + band_4)
    NDWI = (band_3 - band_5) / (band_3 + band_5)

    # stack index layers
    index_layers = np.stack([NDBI,NDVI,NDWI],axis=0)

    # append index layers to the original image
    image = np.vstack([image,index_layers])

    return image

def visualise_img(features, labels, figsize=10, fig_num=1):
    """
    A function of vilusalising relevant images as samples.
    Args:
        features
        labels
        figsize
        fig_num

    Returns:
        None
    """

    fig, axes = plt.subplots(fig_num, 5, figsize=(figsize, figsize))
    plt.rcParams.update({'font.size': 9})

    for i, ax in enumerate(axes):

        image = features[i]
        label = labels[i]

        # convert tensor into numpy.ndarray
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy() 

        if isinstance(label, torch.Tensor):
            label = label.cpu().detach().numpy()

        # plot landsat with band 4, 3, 2 for RGB
        ep.plot_rgb(image, rgb=(3, 2, 1), stretch=True, ax=ax[0])
        ax[0].set_title('RGB: Band 4, 3, 2')

        # plot landsat with band 7, 6, 5 for RGB
        ep.plot_rgb(image, rgb=(6, 5, 4), stretch=True, ax=ax[1])
        ax[1].set_title('RGB: Band 7, 6, 5')

        # plot landsat with NDVI
        ep.plot_rgb(image, rgb=(-3, -2, -1), stretch=True, ax=ax[2])
        ax[2].set_title('RGB: NDBI,NDVI,NDWI')
        ax[2].set_xticks([])
        ax[2].set_yticks([])

        # plot VIIRS
        ax[3].imshow(image[7,:,:], cmap='gray')
        ax[3].set_title('NTL')
        ax[3].set_xticks([])
        ax[3].set_yticks([])

        # plot label
        ax[4].bar(x=['0-14', '15-64', '65-'], height=10**label)
        ax[4].set_title('Population by ages')
        ax[4].set_xticklabels(['Btw. 0-14', 'Btw. 15-64', 'Over 64'])
        ax[4].set_yscale("log") 
        ax[4].set_ylim(10**0,10**6)

    plt.show()
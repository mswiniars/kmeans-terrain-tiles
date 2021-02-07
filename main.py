import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from math import log, tan, pi
import matplotlib
from sklearn.cluster import KMeans

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config

DATA_TYPE = "terrarium"
BUCKET_NAME = 'elevation-tiles-prod'   # replace with your bucket name
KEY = DATA_TYPE + '/{z}/{x}/{y}.png'    # replace with your object key

s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))

EUROPE_BOUNDS = (71.0, -10.0, 35.0, 56.0)
ZOOM = 3

DATA_FOLDER = f"./data/{DATA_TYPE}"
ZOOM_FOLDER = f"{DATA_FOLDER}/{ZOOM}"

if not os.path.exists(ZOOM_FOLDER):
    os.makedirs(ZOOM_FOLDER)

def transform_graph_coords_to_tiles(lat, lon, zoom):
    """
    Convert latitude, longitude to z/x/y tile coordinate at given zoom.
    """
    # convert to radians
    lon_r, lat_r = lon * pi / 180, lat * pi / 180

    # project to mercator format
    x_m, y_m = lon_r, log(tan(0.25 * pi + 0.5 * lat_r))

    # transform to tile space
    tiles, diameter = 2 ** zoom, 2 * pi
    x, y = int(tiles * (x_m + pi) / diameter), int(tiles * (pi - y_m) / diameter)

    return zoom, x, y


def get_tiles(zoom, lat1, lon1, lat2, lon2):
    """
    Convert geographic bounds into a list of tile coordinates at given zoom.
    """
    # convert to geographic bounding box
    min_lat, min_lon = min(lat1, lat2), min(lon1, lon2)
    max_lat, max_lon = max(lat1, lat2), max(lon1, lon2)

    # convert to tile-space bounding box
    _, x_min, y_min = transform_graph_coords_to_tiles(max_lat, min_lon, zoom)
    _, x_max, y_max = transform_graph_coords_to_tiles(min_lat, max_lon, zoom)

    # generate a list of tiles
    xs, ys = range(x_min, x_max + 1), range(y_min, y_max + 1)
    tiles = [(zoom, x, y) for (y, x) in product(ys, xs)]

    return tiles


def download(tiles):
    try:
        for i, (z, x, y) in enumerate(tiles):
            column_folder = f"{ZOOM_FOLDER}/{x}"
            if not os.path.exists(column_folder):
                os.mkdir(column_folder)
                print("Downloading " + str(i+1) + "/" + str(len(tiles)))
                url = KEY.format(z=z, x=x, y=y)
                s3.Bucket(BUCKET_NAME).download_file(url, './{}//{}//{}.png'.format(ZOOM_FOLDER, x, y))

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    except Exception as ex:
        print("Exception occurred: " + str(ex))

def save_decoded_img(c, r, img):
    decoded_folder = f"./data/{DATA_TYPE}_decoded/{ZOOM}/{c}"
    if not os.path.exists(decoded_folder):
        os.makedirs(decoded_folder)
    plt.imsave(f"{decoded_folder}/{r}.png", img, cmap='terrain')

def decode_terrain_img(img):
    height_arr = np.zeros(img.shape[:2])
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            bgr = img[row][column]
            height = (bgr[2] * 256.0 + bgr[1] + bgr[1] / 256.0) - 32768.0
            height_arr[row][column] = height if height > 0 else 0
    
    return height_arr

def concat_vh(list_2d):
    return cv2.hconcat([cv2.vconcat(list_h) for list_h in list_2d]) 

def get_tiles_from_file(decode=True, group=True):
    column_folder_paths = glob.glob(f"{ZOOM_FOLDER}/*")
    imgs = []
    for c, column_folder in enumerate(column_folder_paths):
        img_paths = glob.glob(f"{column_folder}/*.png")
        column_imgs = []
        for r, img_path in enumerate(img_paths):
            print(f"Reading {(r + c*len(img_paths))} img out of {len(column_folder_paths)**2} imgs from {img_path}")
            img = cv2.imread(img_path)
            if decode: 
                img = decode_terrain_img(img)
            column_imgs += [img]
        imgs.append(column_imgs)
    print("Done!")
    
    if group:
        imgs = concat_vh(imgs)
        plt.imshow(imgs, cmap='terrain')
        plt.show()
    plt.imsave(f"./data/{DATA_TYPE}_decoded/{ZOOM}/data.png", imgs, cmap='terrain')
    return imgs

    
def main():
    tiles = get_tiles(ZOOM, *EUROPE_BOUNDS)
    download(tiles)
    img = get_tiles_from_file(decode=True, group=True)
    data = np.asarray(img)
    # np.save(f'./data/{DATA_TYPE}_decoded/{ZOOM}/data.npy', data)
    # data = np.load(f'./data/{DATA_TYPE}_decoded/{ZOOM}/data.npy')
    init_shape = data.shape
    print(f"Data shape: {data.shape}")
    data = data.reshape(-1, 1)
    ### TU MAMY URUCHOMIC PYSPARKA ###
    'data - 2D image -> txt 0 -> 00, 01'
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10, tol=1e-4).fit(data)
    labels = kmeans.labels_
    '-> 1D -> 2D'
    img = labels.reshape(init_shape)
    cluster_centers = kmeans.cluster_centers_.reshape(-1)
    cmap = matplotlib.cm.viridis
    bounds = list(range(0, 6))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(img, norm=norm)
    cbar = plt.colorbar()
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    cbar.set_ticklabels([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(cluster_centers)
    plt.show()

if __name__ == '__main__':
    main()
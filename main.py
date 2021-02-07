import os
import glob
import math
import argparse
from itertools import product
from timeit import default_timer as timer

import cv2
import boto3
import botocore
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description='Data analysis of altitude profile in Europe.')
parser.add_argument('--zoom', metavar='Z', type=int, default=3, help='Zoom ratio of processed data.')
parser.add_argument('--decoded-data-path', metavar="D", type=str, help='Path to already decoded data.')
args = parser.parse_args()

DATA_TYPE = "terrarium"
EUROPE_BOUNDS = (71.0, -10.0, 35.0, 56.0)
ZOOM = args.zoom

DATA_FOLDER = f"./data/{DATA_TYPE}"
ZOOM_FOLDER = f"{DATA_FOLDER}/{ZOOM}"

DOWNLOAD_DATA = False

def generate_tiles_from_graph_coordinates(zoom, lat1, lon1, lat2, lon2):
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

def transform_graph_coords_to_tiles(lat, lon, zoom):
    """
    Convert latitude, longitude to z/x/y tile coordinate at given zoom.
    """
    # convert to radians
    lon_r, lat_r = lon * math.pi / 180, lat * math.pi / 180

    # project to mercator format
    x_m, y_m = lon_r, math.log(math.tan(0.25 * math.pi + 0.5 * lat_r))

    # transform to tile space
    tiles, diameter = 2 ** zoom, 2 * math.pi
    x, y = int(tiles * (x_m + math.pi) / diameter), int(tiles * (math.pi - y_m) / diameter)

    return zoom, x, y

def download_tiles(tiles):
    BUCKET_NAME = 'elevation-tiles-prod'
    KEY = DATA_TYPE + '/{z}/{x}/{y}.png'
    s3 = boto3.resource('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    for i, (z, x, y) in enumerate(tiles):
        column_folder = f"{ZOOM_FOLDER}/{x}"
        if not os.path.exists(column_folder):
            os.mkdir(column_folder)
        img_path = f'./{ZOOM_FOLDER}/{x}/{y}.png'
        if not os.path.exists(img_path):
            print(f"Downloading {i+1}/{len(tiles)} to {img_path}")
            DOWNLOAD_DATA = True
            url = KEY.format(z=z, x=x, y=y)
            s3.Bucket(BUCKET_NAME).download_file(url, img_path)

def decode_terrain_img(img):
    height_profile = np.zeros(img.shape[:2])
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            bgr = img[row][column]
            height = (bgr[2] * 256.0 + bgr[1] + bgr[1] / 256.0) - 32768.0
            height_profile[row][column] = height if height > 0 else 0
    
    return height_profile

def save_decoded_img(c, r, img):
    decoded_folder = f"./data/{DATA_TYPE}_decoded/{ZOOM}/{c}"
    if not os.path.exists(decoded_folder):
        os.makedirs(decoded_folder)
    plt.imsave(f"{decoded_folder}/{r}.png", img, cmap='terrain')

def concat_vh(list_2d):
    return cv2.hconcat([cv2.vconcat(list_h) for list_h in list_2d]) 

def get_tiles_from_file(decode=True, group=True):
    column_folder_paths = sorted(sorted(glob.glob(f"{ZOOM_FOLDER}/*"), key=len))
    imgs = []
    for c, column_folder in enumerate(column_folder_paths):
        img_paths = sorted(sorted(glob.glob(f"{column_folder}/*.png"), key=len))
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
        folder_path_decoded = f"./data/{DATA_TYPE}_decoded/{ZOOM}"
        if not os.path.exists(folder_path_decoded):
            os.makedirs(folder_path_decoded)
        plt.imsave(f"{folder_path_decoded}/data.png", imgs, cmap='terrain')
    return imgs

    
def main():
    start_processing = timer()
    if not args.decoded_data_path:
        tiles = generate_tiles_from_graph_coordinates(ZOOM, *EUROPE_BOUNDS)
        download_tiles(tiles)
        img = get_tiles_from_file(decode=True, group=True)
        data = np.asarray(img)
        np.save(f'./data/{DATA_TYPE}_decoded/{ZOOM}/data.npy', data)
    else:
        print("Skipping reading and decoding images!")
        data = np.load(f'./data/{DATA_TYPE}_decoded/{ZOOM}/data.npy')

    init_shape = data.shape
    data_reshaped = data.reshape(-1, 1)
    print(f"Starting KMeans algorithm for {data.size} data points...")
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10, tol=1e-4).fit(data_reshaped)
    stop_processing = timer()
    print("Done!")
    print(f"Time to cluster the data: {stop_processing - start_processing}s for zoom {ZOOM}, "
          f"decoding data: {not(args.decoded_data_path)}, downloading data: {DOWNLOAD_DATA}.")

    labels = kmeans.labels_
    img = labels.reshape(init_shape)
    cluster_centers = kmeans.cluster_centers_.reshape(-1)

    # Plotting results
    cmap = matplotlib.cm.viridis
    bounds = list(range(0, 6))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.subplot(121)
    plt.imshow(data, cmap='terrain')
    plt.subplot(122)
    plt.imshow(img, norm=norm)
    plt.imsave(f"./data/{DATA_TYPE}_decoded/{ZOOM}/kmeans.png", img, cmap='terrain')
    cbar = plt.colorbar()
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    cbar.set_ticklabels([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(cluster_centers)
    plt.show()
    plt.savefig(f"result_{ZOOM}.png", bbox_inches='tight')

if __name__ == '__main__':
    if not os.path.exists(ZOOM_FOLDER):
        os.makedirs(ZOOM_FOLDER)
    print(f"Starting terrain-tiles data analysis for Europe zoom: {ZOOM}, data type: {DATA_TYPE}")
    main()
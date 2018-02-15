import argparse
import os
from PIL import Image

import pandas as pd

from pynsia.pointcloud import PointCloud
from settings import DATASET_NAME, DATASET_CHOICES

POINTCLOUDS_PATH = 'lidar'


def point_cloud_key(pcl_name):
    return int(pcl_name[4:-4])


DEEPMAP_MODES = 'deep', 'coords'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curates the data in a the masters datasets.')
    parser.add_argument('subject', help='The subject of the experiment.')
    parser.add_argument('dataset', choices=DATASET_CHOICES, help='One of the dataset types')
    parser.add_argument('path', help='The directory where are located the subjects folders')
    parser.add_argument('output', help='The directory where the curated data should be saved')
    args = parser.parse_args()

    input_dir = os.path.join(args.path, args.subject, args.dataset)
    output_dir = os.path.join(args.output, args.subject, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(os.path.join(input_dir, DATASET_NAME))
    df['pointclouds_path'] = df['pointclouds_path'].str.replace('/media/blazaid/Saca/Phd/data',
                                                                '/home/blazaid/Projects/data-phd')
    df['snapshots_path'] = df['snapshots_path'].str.replace('/media/blazaid/Saca/Phd/data',
                                                            '/home/blazaid/Projects/data-phd')
    # 1. Generate the the deepmap images for the point clouds
    deepmaps_dir = os.path.join(output_dir, 'deepmaps')
    if not os.path.exists(deepmaps_dir):
        os.makedirs(deepmaps_dir)

    for pointcloud_path in df['pointclouds_path']:
        pc = PointCloud.load(pointcloud_path)
        deepmap = pc.to_deepmap(h_range=(0, 360), v_range=(-15, 15), h_res=1, v_res=1, max_dist=25, normalize=True)
        img = Image.fromarray(deepmap, 'L')
        img.show()
        # Creo el deepmap
        # Lo salvo en la salida esperada
    # 1. Extract full sequences.
    # There could be point clouds not available, so we separate into sequences of well formed data.
    # Sacar los mapas de profundidad
    # Pasar la velocidad del CANBus a metros por segundo
    # Extraer las seguncias con datos enteros (quitandonos los huecos no capturados)
    # Enriquecer los datos (para la proxima)

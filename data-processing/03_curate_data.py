#!/usr/bin/python2
#

""" sync_sensors.py

Get's all the sensors csv's and creates a new dataframe with all the transformed data.

Usage: python2 02_sync_sensors.py subject validation 10 data/raw_csvs data/sync_csv
"""
from __future__ import print_function

import glob
import multiprocessing
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from pynsia import latlon
from pynsia.pointcloud import PointCloud
from utils import load_subject_df, DATASETS_INFO, load_master_df

BASE_PATH = '/home/blazaid/Projects/data-phd/sync'
# BASE_PATH = '/media/blazaid/Saca/Phd/data/sync'
OUTPUT_PATH = '/home/blazaid/Projects/data-phd/curated'
# OUTPUT_PATH = '/media/blazaid/Saca/Phd/data/curated'
SUBJECTS = 'miguel',  # 'edgar', 'jj', 'miguel'
DATASETS = 'validation', 'training'
SPEED_ROLLING_WINDOW = 10

MAX_LEADER_DISTANCE = 50
MAX_RELATIVE_SPEED = 27.7  # +-100km/h -> +-1m/s
MAX_TLS_DISTANCE = 100
MAX_DRIVABLE_DISTANCE = 100
NUM_SHAKEN_DEEPMAPS = 10
MIRRORED_DEEPMAP = True
SHAKEN_SHIFTS = {'shift_x': 0.05, 'shift_y': 0.05, 'shift_z': 0.05}


def narrow(df, ini_frame, end_frame):
    return df[ini_frame:end_frame + 1].reset_index(drop=True)


def generate_lane_changes(df, lc_frames):
    lane_change = pd.Series(0, index=range(len(df)))
    for ini_frame, end_frame, change in lc_frames:
        lane_change[ini_frame:end_frame] = change
    one_hot = pd.get_dummies(lane_change)
    df[['Lane change left', 'Lane change none', 'Lane change right']] = one_hot[[1, 0, -1]]
    return df


def generate_max_speed(df, ms_frames):
    df['Max speed'] = 0
    for frame, speed in ms_frames:
        df.loc[frame:, 'Max speed'] = speed * 10 / 36
    return df


def adjust_vehicle_speed(df):
    limits = SPEED_ROLLING_WINDOW // 2, -SPEED_ROLLING_WINDOW // 2 - 1

    df['gps_speeds_speed'].fillna(df['canbus_Speed (km/h)'], inplace=True)
    df['Speed'] = df['gps_speeds_speed'].rolling(SPEED_ROLLING_WINDOW, center=True).mean()
    df.drop(['canbus_Speed (km/h)', 'gps_speeds_speed'], axis=1, inplace=True)

    return df[limits[0]:limits[1]]


def generate_acceleration(df):
    df['Acceleration'] = df['Speed'].shift(-1) - df['Speed']

    return df[:-1]


def generate_tls(df, tls_data, master_tls_data):
    df['Next TLS distance'] = 0.0
    status = pd.Series(index=range(len(df)))

    for i, row in tls_data.iterrows():
        status[row['frame']:] = row['status']

        lat, lon = master_tls_data.loc[row['next_tls'], ['lat', 'lon']]
        df.loc[row['frame']:, 'Next TLS distance'] = df.loc[row['frame']:,
                                                     ['gps_positions_latitude', 'gps_positions_longitude']
                                                     ].apply(lambda r: latlon.distance(
            (r['gps_positions_latitude'], r['gps_positions_longitude']),
            (lat, lon)
        ), axis=1)

    one_hot = pd.get_dummies(status)
    df[['Next TLS green', 'Next TLS yellow', 'Next TLS red']] = one_hot[['g', 'y', 'r']]

    return df


def generate_lanes(df, lane_distances):
    df['Distance +1'] = df['Distance 0'] = df['Distance -1'] = np.inf

    for frame, left, current, right in lane_distances:
        # Is the same code for each column, so we do this for over all of them
        for lane, column in zip([left, current, right],
                                ['Distance +1', 'Distance 0', 'Distance -1']):
            # Three options, inf (default), tuple (calc. distance) or 0
            if lane != np.inf:
                if isinstance(lane, tuple):
                    lat, lon = lane
                    df.loc[frame:, column] = df.loc[frame:,
                                             ['gps_positions_latitude',
                                              'gps_positions_longitude']].apply(
                        lambda r: latlon.distance(
                            (r['gps_positions_latitude'],
                             r['gps_positions_longitude']),
                            (lat, lon)
                        ), axis=1
                    )
                else:
                    df.loc[frame:, column] = 0
    return df


def generate_cf_dist(df, cf_dist):
    df['Leader distance'] = np.nan
    for frame_ini, frame_end, epsilon, min_samples in cf_dist:
        for frame, row in df[frame_ini:frame_end].iterrows():
            pointcloud_path = row['pointclouds_path']
            if not pd.isnull(pointcloud_path):
                pc = PointCloud.load(os.path.join(BASE_PATH, pointcloud_path))
                ps = pc.transform(**calibration_data).points

                # Extract the points in the specified bounding box
                masked_points = ps[
                                (ps[:, 0] < 35) &
                                (ps[:, 0] > 0.35) &
                                (ps[:, 2] > -1.5) &
                                (ps[:, 2] < 0.5) &
                                (ps[:, 1] < 1) &
                                (ps[:, 1] > -1), :
                                ]

                # Look for clusters and extract the distance to them
                dist = np.nan
                if len(masked_points) > 0:
                    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(
                        masked_points)
                    labels = db.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    for cluster in range(n_clusters):
                        obs = masked_points[labels == cluster]
                        centroid_dist = np.sqrt(
                            np.mean(obs[:, 0]) ** 2 + np.mean(obs[:, 1]) ** 2)
                        dist = centroid_dist if pd.isnull(dist) else min(dist,
                                                                         centroid_dist)

                df.loc[frame, 'Leader distance'] = dist
        df.loc[frame_ini:frame_end, 'Leader distance'] = df.loc[frame_ini:frame_end, 'Leader distance'].interpolate()
    return df


def relative_bounded(value, maximum):
    return min(1, value / maximum)


def extract_cf_sequences(df):
    cf_sequences = []

    for frame_ini, frame_end, _, _ in dataset_info['cf_dist']:
        sequence = df[frame_ini:frame_end].copy().reset_index(drop=True)

        leader_distance = sequence['Leader distance'].apply(lambda x: relative_bounded(x, MAX_LEADER_DISTANCE))
        next_tls_distance = sequence['Next TLS distance'].apply(lambda x: relative_bounded(x, MAX_TLS_DISTANCE))
        speed_to_leader = (sequence['Leader distance'] - sequence['Leader distance'].shift(1))

        car_following_df = pd.DataFrame({
            'Leader distance': leader_distance,
            'Next TLS distance': next_tls_distance,
            'Next TLS green': sequence['Next TLS green'],
            'Next TLS yellow': sequence['Next TLS yellow'],
            'Next TLS red': sequence['Next TLS red'],
            'Speed': sequence['Speed'],
            'Speed to leader': speed_to_leader,
            'Acceleration': sequence['Acceleration']
        })

        cf_sequences.append(car_following_df)
    return cf_sequences


def extract_lc_sequences(df):
    temp_lc_sequences = []

    frame_ini = 0
    for frame_end in pd.isnull(df['pointclouds_path']).nonzero()[0]:
        sequence = df[frame_ini:frame_end]
        if len(sequence) > 20:
            temp_lc_sequences.append(sequence.dropna(how='any').reset_index(drop=True))
        frame_ini = frame_end + 1

    lc_sequences = []
    for sequence in temp_lc_sequences:
        distance_l_lane = sequence['Distance +1'].apply(lambda x: relative_bounded(x, MAX_DRIVABLE_DISTANCE))
        distance_c_lane = sequence['Distance 0'].apply(lambda x: relative_bounded(x, MAX_DRIVABLE_DISTANCE))
        distance_r_lane = sequence['Distance -1'].apply(lambda x: relative_bounded(x, MAX_DRIVABLE_DISTANCE))
        next_tls_distance = sequence['Next TLS distance'].apply(lambda x: relative_bounded(x, MAX_TLS_DISTANCE))

        lane_change_df = pd.DataFrame({
            'Acceleration': sequence['Acceleration'],
            'Distance +1': distance_l_lane,
            'Distance 0': distance_c_lane,
            'Distance -1': distance_r_lane,
            'Lane change left': sequence['Lane change left'],
            'Lane change none': sequence['Lane change none'],
            'Lane change right': sequence['Lane change right'],
            'Speed': sequence['Speed'],
            'Pointcloud': sequence['pointclouds_path'],
            'Next TLS distance': next_tls_distance,
            'Next TLS green': sequence['Next TLS green'],
            'Next TLS yellow': sequence['Next TLS yellow'],
            'Next TLS red': sequence['Next TLS red'],
        })

        lc_sequences.append(lane_change_df)
    return lc_sequences


def save_deepmap(dm, path):
    dm = dm.normalize(orig=[-25, 25], dest=[1, 0])
    dm.save(path)


if __name__ == '__main__':
    files_pool = multiprocessing.Pool(processes=128)

    for subject in SUBJECTS:
        print('Subject: {}'.format(subject))
        for dataset in DATASETS:
            print('\tDataset: {}'.format(dataset))
            master_tls_df = load_master_df(BASE_PATH, dataset, 'tls')
            master_tls_df = master_tls_df.set_index('tls')
            print('\t\tLoading subject\'s dataframe... ')
            # Load the subject's data
            master_df = load_subject_df(BASE_PATH, subject, dataset, 'dataset')
            # Subject tls's dataframe
            tls_df = load_subject_df(BASE_PATH, subject, dataset, 'tls')
            # Select user info
            dataset_info = DATASETS_INFO[subject][dataset]
            # Subject's path
            SUBJECT_PATH = os.path.join(BASE_PATH, subject, dataset)
            # Calibration data
            calibration_data = dataset_info['calibration_data']

            print('\t\t\tNarrowing data')
            starting_frame = dataset_info['starting_frame'] or 0
            ending_frame = dataset_info['ending_frame'] or (len(master_df) - 1)
            master_df = narrow(master_df, starting_frame, ending_frame)
            print('\t\t\tStarting frame:\t{}'.format(starting_frame))
            print('\t\t\tEnding frame:\t{}'.format(ending_frame))
            print('\t\t\tNew route len:\t{}'.format(ending_frame + 1 - starting_frame))

            print('\t\t\tGenerating lane change data')
            master_df = generate_lane_changes(master_df, dataset_info['lane_changes'])

            print('\t\t\tAdjusting speed')
            master_df = adjust_vehicle_speed(master_df)

            print('\t\t\tGenerating acceleration')
            master_df = generate_acceleration(master_df)

            print('\t\t\tGenerating dists. and status of incomming nearest TLS')
            master_df = generate_tls(master_df, tls_df, master_tls_df)

            print('\t\t\tGenerating drivable distance data')
            master_df = generate_lanes(master_df, dataset_info['lanes_distances'])

            print('\t\t\tGenerating distance to next obstacles')
            master_df = generate_cf_dist(master_df, dataset_info['cf_dist'])

            print('\t\t\tCar following sequences')
            print('\t\t\t\tDeleting previous sequences')
            if not os.path.isdir(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
            filename_prefix = 'cf_{}_{}'.format(subject, dataset)
            for filename in glob.glob(os.path.join(OUTPUT_PATH, filename_prefix + '*')):
                os.remove(filename)

            print('\t\t\t\tExtracting new data: ', end='')
            cf_sequences = extract_cf_sequences(master_df)
            print('{} sequences'.format(len(cf_sequences)))

            print('\t\t\t\tSaving', end='')
            for i, sequence in enumerate(cf_sequences):
                filename = '{}_{:0>5}.csv'.format(filename_prefix, i)
                output_path = os.path.join(OUTPUT_PATH, filename)
                print('.', end='')
                sequence.dropna(how='any').to_csv(output_path, index=False)
            print('')

            print('\t\t\tLane change sequences')
            deepmaps_path = os.path.join(OUTPUT_PATH, 'deepmaps')
            if not os.path.isdir(deepmaps_path):
                os.makedirs(deepmaps_path)
            deepmap_prefix = 'lc_{}_{}'.format(subject, dataset)
            print('\t\t\t\tDeleting previous sequences')
            for filename in glob.glob(os.path.join(deepmaps_path, deepmap_prefix + '*')):
                os.remove(filename)
            for filename in glob.glob(os.path.join(OUTPUT_PATH, deepmap_prefix + '*')):
                os.remove(filename)

            print('\t\t\t\tExtracting new data: ', end='')
            lc_sequences = extract_lc_sequences(master_df)
            print('{} sequences'.format(len(lc_sequences)))

            mirrored_deepmap = MIRRORED_DEEPMAP if dataset != 'validation' else False
            num_shaken_deepmaps = NUM_SHAKEN_DEEPMAPS if dataset != 'validation' else 0
            print('\t\t\t\tAugmenting sequences')
            print('\t\t\t\t\tMirroring: {}'.format(mirrored_deepmap))
            print('\t\t\t\t\tShaking: {}'.format(num_shaken_deepmaps))
            num_generated_sequences = (1 + num_shaken_deepmaps) * (2 if mirrored_deepmap else 1)
            total_sequences = num_generated_sequences * len(lc_sequences)
            print('\t\t\t\t\tTotal sequences: {}'.format(total_sequences))

            sequence_index = 0
            deepmap_index = 0
            master_path = os.path.join(deepmaps_path, deepmap_prefix + '_{:0>10}.dat')
            for num, master_sequence in enumerate(lc_sequences):
                sequences = [[] for _ in range(num_generated_sequences)]

                for index, row in master_sequence.iterrows():
                    # Save the original and the shaken sequences
                    orig_pc = PointCloud.load(os.path.join(BASE_PATH, row['Pointcloud']))
                    orig_pc = orig_pc.transform(**calibration_data)
                    orig_shaken_pcs = [orig_pc.shake(**SHAKEN_SHIFTS) for _ in range(num_shaken_deepmaps)]
                    for i, pc in enumerate([orig_pc] + orig_shaken_pcs):
                        # Save the deepmap
                        path = master_path.format(deepmap_index)
                        dm = pc.to_deepmap(h_range=(0, 360), v_range=(-15, 3), h_res=1, v_res=2)
                        files_pool.apply_async(save_deepmap, (dm, path,))
                        deepmap_index += 1
                        # Add a row to the sequence
                        sequences[i].append([
                            row['Acceleration'],
                            row['Distance +1'],
                            row['Distance -1'],
                            row['Distance 0'],
                            row['Lane change left'],
                            row['Lane change none'],
                            row['Lane change right'],
                            row['Next TLS distance'],
                            row['Next TLS green'],
                            row['Next TLS yellow'],
                            row['Next TLS red'],
                            path.replace(OUTPUT_PATH, '.'),
                            row['Speed'],
                        ])

                    if mirrored_deepmap:
                        mirrored_pc = orig_pc.mirror(fix_x=True, fix_z=True)
                        mirrored_shaken_pcs = [
                            mirrored_pc.shake(**SHAKEN_SHIFTS) for _ in range(num_shaken_deepmaps)]
                        for i, pc in enumerate([mirrored_pc] + mirrored_shaken_pcs, start=num_shaken_deepmaps + 1):
                            # Save the deepmap
                            path = master_path.format(deepmap_index)
                            dm = pc.to_deepmap(h_range=(0, 360), v_range=(-15, 3), h_res=1, v_res=2)
                            files_pool.apply_async(save_deepmap, (dm, path,))
                            deepmap_index += 1
                            # Add a row to the sequence

                            sequences[i].append([
                                row['Acceleration'],
                                row['Distance -1'],  # Mirror
                                row['Distance +1'],  # Mirror
                                row['Distance 0'],
                                row['Lane change right'],  # Mirror
                                row['Lane change none'],
                                row['Lane change left'],  # Mirror
                                row['Next TLS distance'],
                                row['Next TLS green'],
                                row['Next TLS yellow'],
                                row['Next TLS red'],
                                path.replace(OUTPUT_PATH, '.'),
                                row['Speed'],
                            ])
                for sequence in sequences:
                    print('\t\t\t\t\t\tSequence: {} / {}'.format(sequence_index + 1, total_sequences))
                    filename = os.path.join(OUTPUT_PATH, deepmap_prefix + '_{:0>5}.csv')
                    sequence_df = pd.DataFrame(sequence, columns=[
                        'Acceleration', 'Distance +1', 'Distance -1', 'Distance 0',
                        'Lane change left', 'Lane change none', 'Lane change right',
                        'Next TLS distance', 'Next TLS green', 'Next TLS yellow', 'Next TLS red',
                        'Deepmap', 'Relative speed'
                    ])
                    sequence_df.to_csv(filename.format(sequence_index), index=False)
                    sequence_index += 1

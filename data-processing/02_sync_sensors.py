#!/usr/bin/python2
#

""" sync_sensors.py

Get's all the sensors csv's and creates a new dataframe with all the transformed data.

Usage: python2 02_sync_sensors.py subject validation 10 data/raw_csvs data/sync_csv
"""

import argparse
import errno
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

# Each key is the message id. The rest are list of tuples, each of one are specified in the form (indicator, msg_data)
# Each data is specified in the form (init_byte, num_bits, first_bit, resolution, offset, unsigned).
CAN_MESSAGES = {
    # '208': [
    #     ('Brake pedal (%)', 3, 12, 0, 0.1, 0, True),
    # ],
    # '231': [
    #     ('Stoplight (0, 1)', 4, 1, 1, 0, 0, True),
    # ],
    # '236': [
    #     ('Steering angle (deg)', 1, 16, 0, 0.5, -4104, True),
    # ],
    # '2F2': [
    #     ('Steering torque (Nm)', 0, 8, 00.1, 0, True),
    # ],
    # '346': [
    #     ('Autonomy (km)', 7, 8, 0, 1, 0, True),
    #     ('EV power (W)', 1, 16, 0, 10, -10000, True),
    # ],
    # '374': [
    #     ('Charge state (%)', 1, 8, 0, 0.5, 10, True),
    # ],
    '412': [
        ('Speed (km/h)', 1, 9, 0, 1, 0, True),
        #        ('Odometry (km)', 4, 24, 0, 1, 0, True),
    ],
    # '418': [
    #     ('Gears (deg/s)', 0, 8, 0, 0, 0, True),
    # ],
    # '424': [
    #     ('Left blinker (0,1)', 1, 1, 1, 0, 0, True),
    #     ('Right blinker (0,1)', 1, 1, 0, 0, 0, True),
    #     ('Low beam (0,1)', 1, 1, 5, 0, 0, True),
    #     ('High beam (0,1)', 4, 1, 2, 0, 0, True),
    #     ('Position lights (0,1)', 1, 1, 6, 0, 0, True),
    #     ('Front fog (0,1)', 0, 1, 3, 0, 0, True),
    #     ('Rear fog (0,1)', 0, 1, 4, 0, 0, True),
    #     ('Driver\'s door (0,1)', 2, 1, 1, 0, 0, True),
    # ],
    # '696': [
    #     ('Throttle angle (0-5 V)', 1, 16, 0, 0.001, 0, True),
    #     ('Throttle angle (0-2.5 V)', 3, 16, 0, 0.001, 0, True),
    # ]
}

KINECT_IMAGES_DIR = 'snapshots'
LIDAR_IMAGES_DIR = 'pointclouds'


def time_transformer(dfs):
    for i, df in enumerate(dfs):
        df['time'] = df['secs'] + df['nsecs'] * pow(10, -9)
        df.drop(['secs', 'nsecs'], axis=1, inplace=True)
        dfs[i] = df
    return dfs


def canbus_transformer(df):
    def translate_can_message(message, init_byte, num_bits, first_bit, resolution, offset, unsigned):
        end_bit = (init_byte + 1) * 8 - first_bit
        init_bit = end_bit - num_bits

        bin_data = bin(int(message, 16))
        bin_value = bin_data[2 + init_bit: 2 + end_bit]

        value = int(bin_value, 2)
        if not unsigned:
            value -= 1 << num_bits

        return str(float(value * resolution + offset))

    df['id'] = df['frame'].str[1:4]

    dfs = defaultdict(lambda: pd.DataFrame())
    for index, row in df.iterrows():
        frame_id = row['id']
        if frame_id in CAN_MESSAGES.keys():
            size = int(row['frame'][4], 16)
            message = row['frame'][5:5 + 2 * size]
            for column, init_byte, num_bits, first_bit, resolution, offset, unsigned in CAN_MESSAGES[frame_id]:
                value = translate_can_message(message, init_byte, num_bits, first_bit, resolution, offset, unsigned)
                dfs[column] = dfs[column].append({
                    'secs': row['secs'],
                    'nsecs': row['nsecs'],
                    column: value
                }, ignore_index=True)

    return time_transformer([new_df for key, new_df in dfs.iteritems()])


def gps_positions_transformer(df):
    return time_transformer([df])


def gps_speeds_transformer(df):
    new_df = pd.DataFrame()
    new_df['secs'] = df['secs']
    new_df['nsecs'] = df['nsecs']
    new_df['speed'] = np.sqrt(df['v_x'] ** 2 + df['v_y'] ** 2 + df['v_z'] ** 2)
    return time_transformer([new_df])


def snapshots_transformer(df):
    return time_transformer([df])


def pointclouds_transformer(df):
    return time_transformer([df])


def starting_indices(dfs, time_columns):
    def error(dfs, rows, cols):
        return sum(
            pow(dfs[i].loc[rows[i], cols[i]] - dfs[i + 1].loc[rows[i + 1], cols[i + 1]], 2)
            for i in range(len(dfs) - 1)
        )

    # We start in 0 index for all the dataframes. This will be the best position (for now).
    indices = [0 for _ in dfs]
    min_error = error(dfs, indices, time_columns)
    possible_indices = [(min_error, indices)]
    while possible_indices:
        del possible_indices[:]  # .clear() doesn't exists in python2
        # We go one by one over all the dfs.
        for i_df, df in enumerate(dfs):
            # If there is a row over the current one, we check it's contents
            if indices[i_df] < len(df.index):
                new_indices = indices[:]
                new_indices[i_df] += 1
                # Is the new time difference better?
                this_error = error(dfs, new_indices, time_columns)
                if this_error <= min_error:
                    possible_indices.append((this_error, new_indices))

        # Si hay filas mejores que la actual, cogemos la mejor
        if possible_indices:
            possible_indices.sort()
            min_error, indices = possible_indices[0]

    return indices


def syncronize_dataframes(dfs, time_columns, freq=10, exclude_columns=None):
    master_df = pd.DataFrame(columns=[col for df in dfs for col in df])
    rows = [0 for _ in dfs]
    step = 0
    time = 1. / freq
    half_time = time / 2
    while all(row < len(df) - 1 for df, row in zip(dfs, rows)):
        data_row = []
        for df_i, (df, row, col) in enumerate(zip(dfs, rows, time_columns)):
            possible_values = []
            for i in range(len(df) - row):
                value = df.loc[row + i, col]
                diff = step * time - value
                if -half_time < diff < half_time:
                    # We're inside the thresshold so we take the value
                    possible_values.append((value, row + i))
                elif diff < -half_time:
                    # We're over the threshold, so no more values should be taken
                    break

            # We remove all the possible nan and None values
            possible_values = [val for val in possible_values if not pd.isnull(val)]
            if possible_values:
                possible_values.sort()
                _, row = possible_values[0]
                # possible_values.clear()  # python2 doesn't have clear method
                del possible_values[:]

                data_row.extend(list(df.loc[row, :]))
                rows[df_i] = row + 1
            else:
                data_row.extend([np.nan for _ in df.columns])

        master_df.loc[step] = data_row

        step += 1

    # If there are starting or ending rows with null data, we remove them too
    while master_df[time_columns].loc[0, :].isnull().any():
        master_df = master_df[1:]
        master_df.reset_index(drop=True, inplace=True)
    while master_df[time_columns].loc[len(master_df) - 1, :].isnull().any():
        master_df = master_df[:-1]
        master_df.reset_index(drop=True, inplace=True)

    # If it's necessary to remove columns, now it's the moment
    if exclude_columns:
        master_df = master_df[[col for col in master_df.columns if col not in exclude_columns]]

    return master_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transforms a the bag files to a set of csvs, one for each sensor.')
    parser.add_argument('subject', help='The subject of the experiment.')
    parser.add_argument('dataset', choices=('training', 'validation'), help='One of the dataset types')
    parser.add_argument('freq', type=int, help='The rate of the data to be synchronized.')
    parser.add_argument('path', help='The directory where are located the sensors csv with the raw data.')
    parser.add_argument('output', help='The directory where the data is extracted.')
    args = parser.parse_args()

    input_dir = os.path.join(args.path, args.subject, args.dataset)

    sensors_with_transformers = [
        ('canbus', canbus_transformer),
        ('gps_positions', gps_positions_transformer),
        ('gps_speeds', gps_speeds_transformer),
        ('snapshots', snapshots_transformer),
        ('pointclouds', pointclouds_transformer),
    ]

    dfs = []
    for sensor, transformer in sensors_with_transformers:
        print('Sensor ' + sensor)
        print('\tLoading raw CSV')
        df = pd.read_csv(os.path.join(input_dir, sensor + '.csv'))
        print('\tExtracting datasets')
        new_dfs = transformer(df)
        print('\tRenaming columns: ' + ', '.join(df.columns))
        for new_df in new_dfs:
            mapping = {column: sensor + '_' + column for column in new_df.columns}
            new_df.rename(columns=mapping, inplace=True)
        print('\tNew names: ' + ', '.join(df.columns))
        dfs.extend(new_dfs)

    print('Inferring starting indices')
    time_columns = [[c for c in df.columns if c.endswith('_time')][0] for df in dfs]
    indices = starting_indices(dfs, time_columns)
    print('\t' + ', '.join(map(str, indices)))

    print('Reduce the values of the times. The smallest will be 0s')
    minimum_value = min(df[tc].min() for df, tc in zip(dfs, time_columns))
    print('\tMinimum time between sensors: ' + str(minimum_value))
    for df, tc in zip(dfs, time_columns):
        df[tc] -= minimum_value

    print('Synchronizing datasets (and removing time columns in the process)')
    master_df = syncronize_dataframes(dfs, time_columns, freq=args.freq, exclude_columns=time_columns)

    print('Saving external files for synced dataframes')
    output_dir = os.path.join(args.output, args.subject, args.dataset)
    for path_column in master_df[[c for c in master_df.columns if c.endswith('_path')]].columns:
        for path in master_df[path_column]:
            if not pd.isnull(path):
                dest_path = output_dir + path.replace(os.path.join(args.path, args.subject, args.dataset), '')
                try:
                    os.makedirs(os.path.dirname(dest_path))
                except OSError as e:  # Guard against race condition
                    if e.errno != errno.EEXIST:
                        raise
                shutil.copyfile(path, dest_path)
        master_df[path_column] = master_df[path_column].str.replace(input_dir, output_dir)

    master_df.to_csv(os.path.join(output_dir, 'dataset.csv'))

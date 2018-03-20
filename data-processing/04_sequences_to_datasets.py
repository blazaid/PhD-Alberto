from __future__ import print_function

import glob
import multiprocessing
import os
import shutil

import pandas as pd

ORIG_PATH = '/home/blazaid/Projects/data-phd/curated'
DEST_PATH = '/home/blazaid/Projects/data-phd/datasets'
DMS_DIR = 'deepmaps'
SUBJECTS = 'edgar', 'jj', 'miguel'
MOMENTS_BEFORE = [5, 10, 20]

DMS_ORIG_DIR = os.path.join(ORIG_PATH, 'deepmaps')
DMS_DEST_DIR = os.path.join(DEST_PATH, 'deepmaps')
CF = 'cf'
LC = 'lc'
MOMENTS_BEFORE.sort()


def load_sequences(path, subjects):
    sequences = {}
    for dataset in ('training', 'validation'):
        print('{} dataset'.format(dataset))
        sequences[dataset] = {}
        for submodel in (CF, LC):
            print('\t{} model'.format(submodel))
            sequences[dataset][submodel] = {}
            for subject in subjects:
                print('\t\tLoading data for {} .. '.format(subject), end='')
                # Establish the pattern of the files to read
                pattern = '{}_{}_{}_*.csv'.format(submodel, subject, dataset)
                file_pattern = os.path.join(path, pattern)
                # Read the files and store them into the list
                sequences[dataset][submodel][subject] = [
                    pd.read_csv(filepath, index_col=None, engine='python')
                    for filepath in glob.glob(file_pattern)
                ]
                # Tell how many sequences have been loaded
                sequences_loaded = len(sequences[dataset][submodel][subject])
                print('{} loaded'.format(sequences_loaded))

    # Create a new dataset with the sequences of all of the subjects
    for dataset in ('training', 'validation'):
        for submodel in (LC, CF):
            sequences[dataset][submodel]['all'] = []
            for subject in subjects:
                sequences[dataset][submodel]['all'].extend(
                    sequences[dataset][submodel][subject])

    return sequences


def copy_file(orig_path, dest_path):
    if not os.path.exists(dest_path):
        shutil.copy(os.path.join(orig_path, filename), dest_path)


if __name__ == '__main__':
    if not os.path.isdir(DMS_DEST_DIR):
        os.makedirs(DMS_DEST_DIR)

    for filename in glob.glob(os.path.join(DEST_PATH, '*')):
        if not os.path.isdir(filename):
            os.remove(filename)
    for filename in glob.glob(os.path.join(DEST_PATH, DMS_DIR, '*')):
        os.remove(filename)

    base_sequences = load_sequences(ORIG_PATH, SUBJECTS)

    files_pool = multiprocessing.Pool(processes=8)

    for dataset in base_sequences:
        for submodel in base_sequences[dataset]:
            for subject in base_sequences[dataset][submodel]:
                print('Building datasets')
                dfs = base_sequences[dataset][submodel][subject]
                # And now, construct the dataset
                print('\tBuilding {} {} dataset for moments {} ...'.format(submodel, dataset, subject), end='')
                filename = '{}-{}-{}.csv'.format(submodel, subject, dataset)

                if submodel == CF:
                    datasets = pd.concat(dfs, ignore_index=True)
                else:
                    datasets = []
                    temporal_columns = ['Acceleration', 'Next TLS green', 'Next TLS yellow', 'Next TLS red', 'Deepmap', 'Relative speed']
                    for df in dfs:
                        # Generate the dataframes with the shifted times
                        subset = df
                        for moment in MOMENTS_BEFORE:
                            temp_df = df.shift(moment)

                            suffix = ' t_{}'.format(moment)
                            for column in temporal_columns:
                                subset[column + suffix] = temp_df[column]
                        subset = subset[max(MOMENTS_BEFORE):]
                        datasets.append(subset)

                    datasets = pd.concat(datasets, ignore_index=True)

                print('done')
                print('\tSaving dataset {} ... '.format(filename), end='')
                datasets.to_csv(os.path.join(DEST_PATH, filename), index=False)
                print('done')
                if submodel == LC:
                    print('\tSaving deepmaps ... ', end='')
                    dm_columns = [c for c in datasets.columns if c.startswith('Deepmap')]
                    for index, row in datasets.iterrows():
                        for column in dm_columns:
                            orig_path = os.path.join(ORIG_PATH, row[column])
                            dest_path = os.path.join(DEST_PATH, row[column])
                            files_pool.apply_async(copy_file, (orig_path, dest_path,))
                    print('done')

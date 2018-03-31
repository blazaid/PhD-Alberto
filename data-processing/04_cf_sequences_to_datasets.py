from __future__ import print_function

import glob
import os

import pandas as pd

ORIG_PATH = '/media/blazaid/Saca/Phd/data/curated'
DEST_PATH = '/media/blazaid/Saca/Phd/data/datasets'
# ORIG_PATH = '/home/blazaid/Projects/data-phd/curated'
# DEST_PATH = '/home/blazaid/Projects/data-phd/datasets'
SUBJECTS = 'edgar', 'jj', 'miguel'

if __name__ == '__main__':
    # Remove previous datasets for car following
    for filename in glob.glob(os.path.join(DEST_PATH, 'cf-*')):
        if not os.path.isdir(filename):
            os.remove(filename)

    # Load the datasets
    for dataset in ('training', 'validation'):
        dfs = []
        print('Making {} dataset'.format(dataset))
        for subject in SUBJECTS:
            print('\tLoading data for {} ... '.format(subject), end='', flush=True)

            # Establish the pattern of the files to read
            pattern = 'cf_{}_{}_*.csv'.format(subject, dataset)
            file_pattern = os.path.join(ORIG_PATH, pattern)

            # Read the files and store them into the list
            sequences = [pd.read_csv(filepath, index_col=None, engine='python') for filepath in glob.glob(file_pattern)]
            print('{} sequences'.format(len(sequences)))

            # Concat every sequence into a bit one dataset and save it
            df = pd.concat(sequences, ignore_index=True)
            print('\tSaving dataset ... ', end='', flush=True)
            filename = 'cf-{}-{}.csv'.format(subject, dataset)
            df.to_csv(os.path.join(DEST_PATH, filename), index=False)
            print('{} saved'.format(filename))

            # Add the dataset to the list of datasets
            dfs.append(df)

        # After creating all the subjects datasets, save all of them into a whole new dataset
        print('\tSaving dataset for all subjects ... ', end='', flush=True)
        df = pd.concat(dfs, ignore_index=True)
        filename = 'cf-all-{}.csv'.format(dataset)
        df.to_csv(os.path.join(DEST_PATH, filename), index=False)
        print('{} saved'.format(filename))
    print('Finished')

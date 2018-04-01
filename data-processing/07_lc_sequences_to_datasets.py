from __future__ import print_function

import glob
import os
import timeit

import numpy as np
import pandas as pd

ORIG_PATH = '/media/blazaid/Saca/Phd/data/curated'
DEST_PATH = '/media/blazaid/Saca/Phd/data/datasets'
# ORIG_PATH = '/home/blazaid/Projects/data-phd/curated'
# DEST_PATH = '/home/blazaid/Projects/data-phd/datasets'
DMS_DIR = 'deepmaps'
SUBJECTS = 'edgar', 'jj', 'miguel'
MOMENTS_BEFORE = list(sorted([10, 20]))

temporal_columns = ['Next TLS green', 'Next TLS yellow', 'Next TLS red']
temporal_columns += ['Dm {:0>4}'.format(i) for i in range(360 * 8)]

if __name__ == '__main__':
    # Remove previous datasets for car following
    for filename in glob.glob(os.path.join(DEST_PATH, 'lc-*')):
        if not os.path.isdir(filename):
            os.remove(filename)

    # Load the datasets
    columns = None
    all_subjects_file = None
    subject_file = None
    for dataset in ('training', 'validation'):
        print('Making {} dataset'.format(dataset))
        for subject in SUBJECTS:
            print('\tLoading data for {}'.format(subject))

            # Establish the pattern of the files to read
            pattern = 'lc_{}_{}_*.csv'.format(subject, dataset)
            file_pattern = os.path.join(ORIG_PATH, pattern)

            # Read the files and store them into the list. The data from the files will be converted to lessen the space
            # in memory
            filepaths = list(glob.glob(file_pattern))
            num_sequences = len(filepaths)
            print('\t\t{} sequences found'.format(num_sequences))
            df = []
            sequences = []
            for i, filepath in enumerate(filepaths, start=1):
                start = timeit.default_timer()
                print('\t\t\t{} / {} ({}) ... Loading ... '.format(i, num_sequences, filepath), end='', flush=True)
                # Read the file (all values will be converted to float32 by default)
                base_sequence = pd.read_csv(filepath, index_col=None, dtype=np.float32)
                # Cast the integer columns
                print('Casting ... ', end='', flush=True)
                base_sequence['Next TLS green'] = base_sequence['Next TLS green'].astype(np.uint8)
                base_sequence['Next TLS yellow'] = base_sequence['Next TLS yellow'].astype(np.uint8)
                base_sequence['Next TLS red'] = base_sequence['Next TLS red'].astype(np.uint8)
                base_sequence['Lane change left'] = base_sequence['Lane change left'].astype(np.uint8)
                base_sequence['Lane change none'] = base_sequence['Lane change none'].astype(np.uint8)
                base_sequence['Lane change right'] = base_sequence['Lane change right'].astype(np.uint8)
                # Generate the shifted times
                print('Moments ', end='', flush=True)
                sequence = base_sequence
                for moment in MOMENTS_BEFORE:
                    print('.', end='', flush=True)
                    temp_df = base_sequence.shift(moment)
                    moment_columns = ['{} t_{}'.format(c, moment) for c in temporal_columns]
                    sequence[moment_columns] = temp_df[temporal_columns]
                subset = sequence[max(MOMENTS_BEFORE):]
                if columns is None:
                    columns = subset.columns

                print(' Writting (sbj) ... ', end='', flush=True)
                if subject_file is None:
                    subject_file = open(os.path.join(DEST_PATH, 'lc-{}-{}.csv'.format(subject, dataset)), 'w')
                    subject_file.write(','.join(columns) + '\n')
                for row in subset.as_matrix(columns=columns):
                    subject_file.write(','.join(map(str, row)) + '\n')

                print('Writting (all) ... ', end='', flush=True)
                if all_subjects_file is None:
                    all_subjects_file = open(os.path.join(DEST_PATH, 'lc-all-{}.csv'.format(dataset)), 'w')
                    all_subjects_file.write(','.join(columns) + '\n')
                for row in subset.as_matrix(columns=columns):
                    all_subjects_file.write(','.join(map(str, row)) + '\n')
                stop = timeit.default_timer()
                print('Done ({:.2f} seconds)'.format(stop - start))
            subject_file.close()
            subject_file = None
        all_subjects_file.close()
        all_subjects_file = None
    print('Finished')

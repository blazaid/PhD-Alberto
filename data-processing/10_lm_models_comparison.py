import os

import pandas as pd
import tensorflow as tf

BEST_ARCH = [7, 8, 2, 1]
BEST_ARCH_STR = '-'.join(map(str, BEST_ARCH))
MODELS_PATH = 'final-models'
MODEL_FILE_PATTERN = 'cf-mlp-{}-{}'
DATA_PATH = 'data'
DATA_FILE_PATTERN = 'cf-{}-validation.csv'
OUTPUT_DIRS = 'final-outputs'
OUTPUT_FILE_PATTERN = 'cf-subjects-{}.csv'

INPUT_COLS = [
    'Leader distance', 'Next TLS distance', 'Next TLS green', 'Next TLS yellow',
    'Next TLS red', 'Speed', 'Speed to leader'
]
OUTPUT_COL = 'Acceleration'

if __name__ == '__main__':
    for data_subject in 'edgar', 'jj', 'miguel':
        data_file = os.path.join(DATA_PATH, DATA_FILE_PATTERN.format(data_subject))
        dataset = pd.read_csv(data_file, index_col=None)

        df = pd.DataFrame()
        df['expected'] = dataset[OUTPUT_COL].values
        for infer_subject in 'edgar', 'jj', 'miguel':
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                filename = MODEL_FILE_PATTERN.format(infer_subject, BEST_ARCH_STR)
                model_meta_path = os.path.join(MODELS_PATH, filename + '.meta')

                # Restore saved session
                saver = tf.train.import_meta_graph(model_meta_path)
                saver.restore(session, os.path.join(MODELS_PATH, filename))

                # Get output tensor of the saved graph
                output = tf.get_collection('output')[0]

                # Execute against the data
                df[infer_subject] = session.run(output, feed_dict={
                    'input:0': dataset[INPUT_COLS].values,
                })
        save_file = os.path.join(OUTPUT_DIRS, OUTPUT_FILE_PATTERN.format(data_subject))
        df.to_csv(save_file, index=None)
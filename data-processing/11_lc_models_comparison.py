import os

import pandas as pd
import tensorflow as tf

from utils import LC_TARGET_COLS

BEST_ARCH_STR = 'c16-4-18-v-d128-d0.1'
MODELS_PATH = 'final-models'
MODEL_FILE_PATTERN = 'lc-cnn-{}-{}'
DATA_PATH = 'data'
DATA_FILE_PATTERN = 'lc-{}-validation.csv'
OUTPUT_DIRS = 'final-outputs'
OUTPUT_FILE_PATTERN = 'lc-subjects-{}.csv'

if __name__ == '__main__':
    for data_subject in 'edgar', 'jj', 'miguel':
        data_file = os.path.join(DATA_PATH, DATA_FILE_PATTERN.format(data_subject))
        dataset = pd.read_csv(data_file, index_col=None)

        df = pd.DataFrame()
        df[LC_TARGET_COLS] = dataset[LC_TARGET_COLS]
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
                vals = session.run(output, feed_dict={
                    'input:0': dataset.drop(LC_TARGET_COLS, axis=1).values,
                })
                df['Lane change left {}'.format(infer_subject)] = vals[:, 0]
                df['Lane change none {}'.format(infer_subject)] = vals[:, 1]
                df['Lane change right {}'.format(infer_subject)] = vals[:, 2]
        save_file = os.path.join(OUTPUT_DIRS, OUTPUT_FILE_PATTERN.format(data_subject))
        df.to_csv(save_file, index=None)

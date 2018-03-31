import os

import pandas as pd
import tensorflow as tf

subject = 'all'
BASE_PATH = '/media/blazaid/Saca/Phd/data/datasets'
steps = 100
layers = []
dropout = 0.0

LEARNING_RATE = 0.01


class C:
    OUTPUT_COLS_PREFIX = 'Lane change'
    IMAGE_COLS_PREFIX = 'Deepmap'

    def __init__(self, df, base_path):
        self.df = df
        self.base_path = tf.convert_to_tensor(base_path)

        # Those are the columns to use
        self.o_cols = [c for c in sorted(df.columns) if c.startswith(C.OUTPUT_COLS_PREFIX)]
        self.d_cols = [c for c in sorted(df.columns) if c.startswith(C.IMAGE_COLS_PREFIX)]
        self.i_cols = [c for c in sorted(df.columns) if not (c in self.o_cols or c in self.d_cols)]

        # The tensors with the actual input and output data
        self.numeric_inputs = tf.convert_to_tensor(df[self.i_cols].values, dtype=tf.float32)
        self.deepmap_inputs = tf.convert_to_tensor(df[self.d_cols].values, dtype=tf.string)
        self.labels = tf.convert_to_tensor(df[self.o_cols].values, dtype=tf.float32)

        # Current data as a Dataset
        self.data = tf.data.Dataset.from_tensor_slices((self.numeric_inputs, self.deepmap_inputs, self.labels))
        self.data = self.data.map(self._input_data_mapping)

    def _input_data_mapping(self, numeric_inputs, deepmap_inputs, labels):
        """Input parser for samples of the training set."""
        deepmap_inputs = self.base_path + deepmap_inputs
        filename_queue = tf.train.string_input_producer(deepmap_inputs)
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        print(key, value)
        return key, value


df = pd.read_csv(os.path.join(BASE_PATH, 'lc-{}-training.csv'.format(subject)), index_col=None)
with tf.device('/cpu:0'):
    training_set = C(df, BASE_PATH)
    iterator = tf.data.Iterator.from_structure(training_set.data.output_types, training_set.data.output_shapes)
    next_batch = iterator.get_next()
training_init_op = iterator.make_initializer(training_set.data)

with tf.Session() as sess:
    sess.run(training_init_op)
    print(sess.run(next_batch))

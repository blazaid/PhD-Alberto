import collections
import os
import re
import shutil
from multiprocessing import Process

import numpy as np
import pandas as pd
import tensorboard.default
import tensorboard.program
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tfz import IVar, OVar, fuzzy_controller

SUBJECT = 'all'
DATASETS_PATH = './data'
LEARNING_RATE = 0.01
TRAIN_STEPS = 1000
LOGS_STEPS = 1
NUM_FS = [3, 3, 2, 2, 2, 3, 3]

input_cols = [
    'Leader distance', 'Next TLS distance', 'Next TLS green', 'Next TLS yellow',
    'Next TLS red', 'Speed', 'Speed to leader'
]
output_col = 'Acceleration'

train_file = os.path.join(DATASETS_PATH, 'cf-{}-training.csv'.format(SUBJECT))
test_file = os.path.join(DATASETS_PATH, 'cf-{}-validation.csv'.format(SUBJECT))

num_fs_string = '-'.join(str(x) for x in NUM_FS)
summary_trn_path = 'tensorboard/{}/{}/training'.format(SUBJECT, num_fs_string)
if os.path.exists(summary_trn_path):
    shutil.rmtree(summary_trn_path)
summary_val_path = 'tensorboard/{}/{}/validation'.format(SUBJECT, num_fs_string)
if os.path.exists(summary_val_path):
    shutil.rmtree(summary_val_path)
summary_tst_path = 'tensorboard/{}/{}/test'.format(SUBJECT, num_fs_string)
if os.path.exists(summary_tst_path):
    shutil.rmtree(summary_tst_path)


def launch_tensorboard(tb_trn_path, tb_val_path, tb_tst_path):
    tensorboard.program.FLAGS.logdir = 'training:{},validation:{},test:{}'.format(tb_trn_path, tb_val_path, tb_tst_path)
    tensorboard.program.main(tensorboard.default.get_plugins(), tensorboard.default.get_assets_zip_provider())

    print('Tensorboard started on http://localhost:6006/'.format())


def extract_fuzzy_controller_data(session, fc_data, input_vars):
    all_variables = [n.name for n in tf.get_default_graph().as_graph_def().node]
    patterns = ['^{}/b$'.format(var) for var in input_vars]
    patterns.extend(['^{}/s\d+$'.format(var) for var in input_vars])
    patterns.extend(['^{}/sf\d+$'.format(var) for var in input_vars])
    patterns.extend(['^{}/sl\d+$'.format(var) for var in input_vars])
    patterns.append('^{}$'.format('fuzzy_output_weights'))
    for pattern in patterns:
        for var in all_variables:
            if re.match(pattern, var) is not None:
                tensor = tf.get_default_graph().get_tensor_by_name(var + ':0')
                values = session.run(tensor)
                if var == 'fuzzy_output_weights':
                    values = values.flatten()
                    values = ';'.join([str(x) for x in values])
                fc_data[var].append(values)


if __name__ == '__main__':
    input_var_names = [''.join(s[:1].upper() + s[1:] for s in i.split(' ')) for i in input_cols]
    output_var_name = output_col.lower().capitalize()

    # Fuzzy controller graph
    x, y_hat = fuzzy_controller(
        i_vars=[
            IVar(name=input_var_names[0], fuzzy_sets=NUM_FS[0], domain=(0., 1.)),
            IVar(name=input_var_names[1], fuzzy_sets=NUM_FS[1], domain=(0., 1.)),
            IVar(name=input_var_names[2], fuzzy_sets=NUM_FS[2], domain=(0., 1.)),
            IVar(name=input_var_names[3], fuzzy_sets=NUM_FS[3], domain=(0., 1.)),
            IVar(name=input_var_names[4], fuzzy_sets=NUM_FS[4], domain=(0., 1.)),
            IVar(name=input_var_names[5], fuzzy_sets=NUM_FS[5], domain=(0., 20.)),
            IVar(name=input_var_names[6], fuzzy_sets=NUM_FS[6], domain=(-20., 20.)),
        ],
        o_var=OVar(name=output_var_name, values=(-1, 1))
    )

    # Expected output
    y = tf.placeholder(tf.float32)

    # Training graph
    cost = tf.reduce_mean(tf.squared_difference(y, y_hat))

    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # Tensorboard outputs
    tf.summary.scalar('RMSE', cost)
    all_variables = [n.name for n in tf.get_default_graph().as_graph_def().node]
    patterns = ['^{}/b$'.format(var) for var in input_var_names]
    patterns.extend(['^{}/s\d+$'.format(var) for var in input_var_names])
    patterns.extend(['^{}/sf\d+$'.format(var) for var in input_var_names])
    patterns.extend(['^{}/sl\d+$'.format(var) for var in input_var_names])
    for pattern in patterns:
        for var in all_variables:
            if re.match(pattern, var) is not None:
                tensor = tf.get_default_graph().get_tensor_by_name(var + ':0')
                tf.summary.scalar(var, tensor)
    patterns = ['^{}$'.format('fuzzy_output_weights')]
    for pattern in patterns:
        for var in all_variables:
            if re.match(pattern, var) is not None:
                tensor = tf.get_default_graph().get_tensor_by_name(var + ':0')
                tf.summary.histogram(var, tensor)

    merged_summary = tf.summary.merge_all()
    writer_trn = tf.summary.FileWriter(summary_trn_path)
    writer_val = tf.summary.FileWriter(summary_val_path)
    writer_tst = tf.summary.FileWriter(summary_tst_path)

    train_df = pd.read_csv(train_file, index_col=False).astype(np.float32)
    test_df = pd.read_csv(test_file, index_col=False).astype(np.float32)

    # Launching tensorboard
    tb_process = Process(target=launch_tensorboard, args=(summary_trn_path, summary_val_path, summary_tst_path))
    tb_process.start()

    fc_data = collections.defaultdict(list)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer_trn.add_graph(session.graph)

        for step in range(TRAIN_STEPS):
            # Create a partition of the training dataset
            #train_partition, validation_partition = train_test_split(train_df, test_size=0.2, random_state=1)
            train_partition, validation_partition = train_test_split(train_df, test_size=0.2)

            # When logging, evaluate also with the validation partition and the test
            if TRAIN_STEPS % LOGS_STEPS == 0:
                extract_fuzzy_controller_data(session, fc_data, input_var_names)

                summary = session.run(merged_summary, feed_dict={
                    x: train_partition[input_cols].values,
                    y: train_partition[[output_col]].values
                })
                writer_trn.add_summary(summary, step)
                writer_trn.flush()
                summary = session.run(merged_summary, feed_dict={
                    x: validation_partition[input_cols].values,
                    y: validation_partition[[output_col]].values
                })
                writer_val.add_summary(summary, step)
                writer_val.flush()
                summary = session.run(merged_summary, feed_dict={
                    x: test_df[input_cols].values,
                    y: test_df[[output_col]].values
                })
                writer_tst.add_summary(summary, step)
                writer_tst.flush()

            # Train with the training partition
            session.run(train, feed_dict={
                x: train_partition[input_cols].values,
                y: train_partition[[output_col]].values
            })

        # Write results to a file so we can later make graphs
        pd.DataFrame({
            'expected': test_df[[output_col]].values.flatten(),
            'real': session.run(y_hat, feed_dict={
                x: test_df[input_cols].values,
                y: test_df[[output_col]].values
            }).flatten(),
        }).to_csv('fcs-outputs-{}-{}.csv'.format(SUBJECT, num_fs_string), index=None)
        pd.DataFrame(fc_data).to_csv('fcs-description-{}-{}.csv'.format(SUBJECT, num_fs_string), index=None)
        print('Finished training')

    tb_process.join()

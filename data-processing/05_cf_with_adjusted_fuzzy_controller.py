import os
import re
import shutil
from multiprocessing import Process

import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import pynsia.tensorflow.fuzzy as tfz
from utils import launch_tensorboard, extract_fuzzy_controller_data

SUBJECT = 'all'
DATASETS_PATH = './data'
RULES_LEARNING_RATE = 0.01
VARS_LEARNING_RATE = 0.001
TRAIN_VARS_STEPS = 1
TRAIN_RULES_STEPS = 1
TRAIN_STEPS = 500000
LOGS_STEPS = (TRAIN_VARS_STEPS + TRAIN_RULES_STEPS) * TRAIN_STEPS / 100 
# NUM_FS = [2, 2, 2, 2, 2, 2, 2]
# NUM_FS = [3, 3, 2, 2, 2, 3, 3]
# NUM_FS = [4, 3, 2, 2, 2, 3, 3]
NUM_FS = [5, 5, 2, 2, 2, 5, 5]

input_cols = [
    'Leader distance', 'Next TLS distance', 'Next TLS green', 'Next TLS yellow',
    'Next TLS red', 'Speed', 'Speed to leader'
]
output_col = 'Acceleration'

train_file = os.path.join(DATASETS_PATH, 'cf-{}-training.csv'.format(SUBJECT))
test_file = os.path.join(DATASETS_PATH, 'cf-{}-validation.csv'.format(SUBJECT))

num_fs_string = '-'.join(str(x) for x in NUM_FS)
summary_trn_path = 'tensorboard/{}/cf-fcs-{}/training'.format(SUBJECT, num_fs_string)
if os.path.exists(summary_trn_path):
    shutil.rmtree(summary_trn_path)
summary_val_path = 'tensorboard/{}/cf-fcs-{}/validation'.format(SUBJECT, num_fs_string)
if os.path.exists(summary_val_path):
    shutil.rmtree(summary_val_path)
summary_tst_path = 'tensorboard/{}/cf-fcs-{}/test'.format(SUBJECT, num_fs_string)
if os.path.exists(summary_tst_path):
    shutil.rmtree(summary_tst_path)

if __name__ == '__main__':
    input_var_names = [''.join(s[:1].upper() + s[1:] for s in i.split(' ')) for i in input_cols]
    output_var_name = output_col.lower().capitalize()

    # Fuzzy controller graph
    x, y_hat = tfz.fuzzy_controller(
        i_vars=[
            tfz.IVar(name=input_var_names[0], fuzzy_sets=NUM_FS[0], domain=(0., 1.)),
            tfz.IVar(name=input_var_names[1], fuzzy_sets=NUM_FS[1], domain=(0., 1.)),
            tfz.IVar(name=input_var_names[2], fuzzy_sets=NUM_FS[2], domain=(0., 1.)),
            tfz.IVar(name=input_var_names[3], fuzzy_sets=NUM_FS[3], domain=(0., 1.)),
            tfz.IVar(name=input_var_names[4], fuzzy_sets=NUM_FS[4], domain=(0., 1.)),
            tfz.IVar(name=input_var_names[5], fuzzy_sets=NUM_FS[5], domain=(0., 20.)),
            tfz.IVar(name=input_var_names[6], fuzzy_sets=NUM_FS[6], domain=(-40., 40.)),
        ],
        o_var=tfz.OVar(name=output_var_name, values=(-1, 1))
    )

    # Expected output
    y = tf.placeholder(tf.float32)

    # Cost
    cost = tf.sqrt(tf.losses.mean_squared_error(y, y_hat))

    # Training graphs
    vars_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'tfz/var/')
    vars_optimizer = tf.train.AdamOptimizer(VARS_LEARNING_RATE)
    vars_train = vars_optimizer.minimize(cost, var_list=vars_vars)
    rules_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'tfz/rules/')
    rules_optimizer = tf.train.AdamOptimizer(RULES_LEARNING_RATE)
    rules_train = rules_optimizer.minimize(cost, var_list=rules_vars)

    # Tensorboard outputs
    tf.summary.scalar('RMSE', cost)
    for var in vars_vars:
        tf.summary.scalar(var.name, var)
    for var in rules_vars:
        tf.summary.histogram(var.name, var)
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

        fcs_rms = {
            'training': [],
            'validation': [],
            'test': [],
        }
        train_partition, validation_partition = train_test_split(train_df, test_size=0.1, random_state=0)
        step = 0
        for main_step in range(TRAIN_STEPS):
            print('Main step: {}'.format(main_step))
            # Create a partition of the training dataset

            for nm, train, steps in (('rules', rules_train, TRAIN_RULES_STEPS), ('vars', vars_train, TRAIN_VARS_STEPS)):
                print('Training {}: {} steps'.format(nm, steps))
                for _ in range(steps):
                    # When logging, evaluate also with the validation partition and the test
                    if step % LOGS_STEPS == 0:
                        print('Step: {}'.format(step))
                        extract_fuzzy_controller_data(session, fc_data, input_var_names)

                        summary = session.run(merged_summary, feed_dict={
                            x: train_partition[input_cols].values,
                            y: train_partition[[output_col]].values
                        })
                        writer_trn.add_summary(summary, step)
                        writer_trn.flush()
                        fcs_rms['training'].append(session.run(cost, feed_dict={
                            x: train_partition[input_cols].values,
                            y: train_partition[[output_col]].values
                        }))
                        summary = session.run(merged_summary, feed_dict={
                            x: validation_partition[input_cols].values,
                            y: validation_partition[[output_col]].values
                        })
                        writer_val.add_summary(summary, step)
                        writer_val.flush()
                        fcs_rms['validation'].append(session.run(cost, feed_dict={
                            x: validation_partition[input_cols].values,
                            y: validation_partition[[output_col]].values
                        }))
                        summary = session.run(merged_summary, feed_dict={
                            x: test_df[input_cols].values,
                            y: test_df[[output_col]].values
                        })
                        writer_tst.add_summary(summary, step)
                        writer_tst.flush()
                        fcs_rms['test'].append(session.run(cost, feed_dict={
                            x: test_df[input_cols].values,
                            y: test_df[[output_col]].values
                        }))

                    # Train with the training partition
                    session.run(train, feed_dict={
                        x: train_partition[input_cols].values,
                        y: train_partition[[output_col]].values
                    })
                    step += 1

        # Write results to a file so we can later make graphs
        pd.DataFrame({
            'expected': test_df[[output_col]].values.flatten(),
            'real': session.run(y_hat, feed_dict={
                x: test_df[input_cols].values,
                y: test_df[[output_col]].values
            }).flatten(),
        }).to_csv('outputs/cf-fcs-outputs-{}-{}.csv'.format(SUBJECT, num_fs_string), index=None)
        pd.DataFrame(fc_data).to_csv('outputs/cf-fcs-description-{}-{}.csv'.format(SUBJECT, num_fs_string), index=None)
        pd.DataFrame(fcs_rms).to_csv('outputs/cf-fcs-rms-{}-{}.csv'.format(SUBJECT, num_fs_string), index=None)
        print('Finished training')
        print('Saving model ...')
        saver = tf.train.Saver()
        saver.save(session, 'models/cf-fcs-{}-{}'.format(SUBJECT, num_fs_string))
        saver.export_meta_graph('models/cf-fcs-{}-{}.meta'.format(SUBJECT, num_fs_string))
        print('Saved')

    tb_process.join()

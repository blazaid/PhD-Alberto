import os
import shutil
from multiprocessing import Process

import numpy as np
import pandas as pd
import tensorboard.default
import tensorboard.program
import tensorflow as tf
from sklearn.model_selection import train_test_split

SUBJECT = 'all'
DATASETS_PATH = './data'
LEARNING_RATE = 0.01
TRAIN_STEPS = 1000
LOGS_STEPS = 1
HIDDEN_UNITS = [10, 5, 3]  # [], [10], [10, 5], [10, 5, 3]
ACTIVATION_FUNCTION = tf.nn.tanh

input_cols = [
    'Leader distance', 'Next TLS distance', 'Next TLS green', 'Next TLS yellow',
    'Next TLS red', 'Speed', 'Speed to leader'
]
output_col = 'Acceleration'

train_file = os.path.join(DATASETS_PATH, 'cf-{}-training.csv'.format(SUBJECT))
test_file = os.path.join(DATASETS_PATH, 'cf-{}-validation.csv'.format(SUBJECT))

hidden_units_str = '-'.join(str(x) for x in HIDDEN_UNITS)
if not hidden_units_str:
    hidden_units_str = 'none'
summary_trn_path = 'tensorboard/{}/{}/training'.format(SUBJECT, hidden_units_str)
summary_val_path = 'tensorboard/{}/{}/validation'.format(SUBJECT, hidden_units_str)
summary_tst_path = 'tensorboard/{}/{}/test'.format(SUBJECT, hidden_units_str)

if os.path.exists(summary_trn_path):
    shutil.rmtree(summary_trn_path)
if os.path.exists(summary_val_path):
    shutil.rmtree(summary_val_path)
if os.path.exists(summary_tst_path):
    shutil.rmtree(summary_tst_path)


def launch_tensorboard(tb_trn_path, tb_val_path, tb_tst_path):
    tensorboard.program.FLAGS.logdir = 'training:{},validation:{},test:{}'.format(tb_trn_path, tb_val_path, tb_tst_path)
    tensorboard.program.main(tensorboard.default.get_plugins(), tensorboard.default.get_assets_zip_provider())

    print('Tensorboard started on http://localhost:6006/'.format())


def multilayer_perceptron(layers, activation_fn=None):
    activation_fn = activation_fn or tf.nn.relu
    x = tf.placeholder(tf.float32, name='input', shape=[None, layers[0]])
    output = x
    for layer_id, num_neurons in enumerate(layers[1:], start=1):
        output = tf.layers.dense(inputs=output, units=num_neurons, activation=activation_fn)
    return x, output


if __name__ == '__main__':
    tb_process = Process(target=launch_tensorboard, args=(summary_trn_path, summary_val_path, summary_tst_path))

    architecture = [len(input_cols)] + HIDDEN_UNITS + [1]
    architecture_str = '-'.join(str(x) for x in architecture)
    print('Architecture: {}'.format(architecture_str))

    # Fuzzy controller graph
    x, y_hat = multilayer_perceptron(architecture, activation_fn=ACTIVATION_FUNCTION)
    tf.add_to_collection('output', y_hat)

    # Expected output
    y = tf.placeholder(tf.float32)

    # Training graph
    cost = tf.reduce_mean(tf.squared_difference(y, y_hat))
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    tf.summary.scalar('RMSE', cost)
    merged_summary = tf.summary.merge_all()
    writer_trn = tf.summary.FileWriter(summary_trn_path)
    writer_val = tf.summary.FileWriter(summary_val_path)
    writer_tst = tf.summary.FileWriter(summary_tst_path)

    train_df = pd.read_csv(train_file, index_col=False).astype(np.float32)
    test_df = pd.read_csv(test_file, index_col=False).astype(np.float32)

    tb_process.start()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer_trn.add_graph(session.graph)

        mlp_rms = {
            'training': [],
            'validation': [],
            'test': [],
        }
        for step in range(TRAIN_STEPS):
            # Create a partition of the training dataset
            train_partition, validation_partition = train_test_split(train_df, test_size=0.1)

            # When logging, evaluate also with the validation partition and the test
            if step % LOGS_STEPS == 0:
                print('Step: {}'.format(step))
                summary = session.run(merged_summary, feed_dict={
                    x: train_partition[input_cols].values,
                    y: train_partition[[output_col]].values
                })
                writer_trn.add_summary(summary, step)
                writer_trn.flush()
                mlp_rms['training'].append(session.run(cost, feed_dict={
                    x: train_df[input_cols].values,
                    y: train_df[[output_col]].values
                }))

                summary = session.run(merged_summary, feed_dict={
                    x: validation_partition[input_cols].values,
                    y: validation_partition[[output_col]].values
                })
                writer_val.add_summary(summary, step)
                writer_val.flush()
                mlp_rms['validation'].append(session.run(cost, feed_dict={
                    x: validation_partition[input_cols].values,
                    y: validation_partition[[output_col]].values
                }))

                summary = session.run(merged_summary, feed_dict={
                    x: test_df[input_cols].values,
                    y: test_df[[output_col]].values
                })
                writer_tst.add_summary(summary, step)
                writer_tst.flush()
                mlp_rms['test'].append(session.run(cost, feed_dict={
                    x: test_df[input_cols].values,
                    y: test_df[[output_col]].values
                }))

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
                y: test_df[output_col].values
            }).flatten(),
        }).to_csv('outputs/mlp-outputs-{}-{}.csv'.format(SUBJECT, architecture_str), index=None)
        pd.DataFrame(mlp_rms).to_csv('outputs/mlp-rms-{}-{}.csv'.format(SUBJECT, architecture_str), index=None)
        print('Finished training')
        print('Saving model ...')
        saver = tf.train.Saver()
        saver.save(session, 'models/mlp-{}-{}'.format(SUBJECT, architecture_str))
        saver.export_meta_graph('models/mlp-{}-{}.meta'.format(SUBJECT, architecture_str))
        print('Saved')
        # with tf.Session() as session:
        # saver = tf.train.import_meta_graph('tmp/model.meta')
        # saver.restore(session, 'tmp/model')

        # output = tf.get_collection('output')[0]
        # print(session.run(output, feed_dict={
        #    x: test_df[input_cols].values[:2],
        # }))
        # output = tf.get_collection('output')[1]
        # print(session.run(output, feed_dict={
        #    x: test_df[input_cols].values[:2],
        # }))
    tb_process.join()

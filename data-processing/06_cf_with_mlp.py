import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import launch_tensorboard, multilayer_perceptron

SUBJECT = 'all'
DATASETS_PATH = './data'
MAX_LEARN_RATE = 0.01
MIN_LEARN_RATE = 0.0001 / 2
DECAY_SPEED = 5000 * 2
TRAIN_STEPS = 100000 * 2
DROPOUT_RATE = 0.1
LOGS_STEPS = 1000 * 2
HIDDEN_UNITS = [16]  # [16], [8, 2], [16, 8], [16, 16, 8]

ACTIVATION_FUNCTION = tf.nn.tanh

input_cols = [
    'Leader distance', 'Next TLS distance', 'Next TLS green', 'Next TLS yellow',
    'Next TLS red', 'Speed', 'Speed to leader'
]
output_col = 'Acceleration'

train_file = os.path.join(DATASETS_PATH, 'cf-{}-training.csv'.format(SUBJECT))
test_file = os.path.join(DATASETS_PATH, 'cf-{}-validation.csv'.format(SUBJECT))

if __name__ == '__main__':
    architecture = [len(input_cols)] + HIDDEN_UNITS + [1]
    architecture_str = '-'.join(str(x) for x in architecture)
    print('Architecture: {}'.format(architecture_str))

    # Create the tensorboard directories associated to this training configuration
    summary_trn_path = 'tensorboard/{}/{}/training'.format(SUBJECT, architecture_str)
    summary_val_path = 'tensorboard/{}/{}/validation'.format(SUBJECT, architecture_str)
    summary_tst_path = 'tensorboard/{}/{}/test'.format(SUBJECT, architecture_str)
    if os.path.exists(summary_trn_path):
        shutil.rmtree(summary_trn_path)
    if os.path.exists(summary_val_path):
        shutil.rmtree(summary_val_path)
    if os.path.exists(summary_tst_path):
        shutil.rmtree(summary_tst_path)

    # Network
    x, y_hat, dropout = multilayer_perceptron(architecture, activation_fn=ACTIVATION_FUNCTION, output_fn=None)
    tf.add_to_collection('output', y_hat)

    # Expected output
    y = tf.placeholder(tf.float32)

    # Training graph
    cost = tf.sqrt(tf.losses.mean_squared_error(y, y_hat))
    learning_rate = tf.placeholder_with_default(0.0, shape=())
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    tf.summary.scalar('RMSE', cost)
    tf.summary.scalar('learning_rate', learning_rate)
    merged_summary = tf.summary.merge_all()
    writer_trn = tf.summary.FileWriter(summary_trn_path)
    writer_val = tf.summary.FileWriter(summary_val_path)
    writer_tst = tf.summary.FileWriter(summary_tst_path)

    train_df = pd.read_csv(train_file, index_col=False).astype(np.float32)
    test_df = pd.read_csv(test_file, index_col=False).astype(np.float32)

    tb_process = launch_tensorboard(summary_trn_path, summary_val_path, summary_tst_path)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer_trn.add_graph(session.graph)

        mlp_rms = {
            'training': [],
            'validation': [],
            'test': [],
        }
        train_partition, validation_partition = train_test_split(train_df, test_size=0.1, random_state=0)
        for step in range(TRAIN_STEPS):
            # Compute the decaying learning rate
            lr = MIN_LEARN_RATE + (MAX_LEARN_RATE - MIN_LEARN_RATE) * np.math.exp(-step / DECAY_SPEED)

            # When logging, evaluate also with the validation partition and the test
            if step % LOGS_STEPS == 0:
                print('l - {}'.format(step))
                summary = session.run(merged_summary, feed_dict={
                    x: train_partition[input_cols].values,
                    y: train_partition[[output_col]].values,
                    learning_rate: lr,
                })
                writer_trn.add_summary(summary, step)
                writer_trn.flush()
                mlp_rms['training'].append(session.run(cost, feed_dict={
                    x: train_partition[input_cols].values,
                    y: train_partition[[output_col]].values,
                    learning_rate: lr,
                }))

                summary = session.run(merged_summary, feed_dict={
                    x: validation_partition[input_cols].values,
                    y: validation_partition[[output_col]].values,
                    learning_rate: lr,
                })
                writer_val.add_summary(summary, step)
                writer_val.flush()
                mlp_rms['validation'].append(session.run(cost, feed_dict={
                    x: validation_partition[input_cols].values,
                    y: validation_partition[[output_col]].values,
                    learning_rate: lr,
                }))

                summary = session.run(merged_summary, feed_dict={
                    x: test_df[input_cols].values,
                    y: test_df[[output_col]].values,
                    learning_rate: lr,
                })
                writer_tst.add_summary(summary, step)
                writer_tst.flush()
                mlp_rms['test'].append(session.run(cost, feed_dict={
                    x: test_df[input_cols].values,
                    y: test_df[[output_col]].values,
                    learning_rate: lr,
                }))

            # Train with the training partition
            session.run(train, feed_dict={
                x: train_partition[input_cols].values,
                y: train_partition[[output_col]].values,
                dropout: DROPOUT_RATE,
                learning_rate: lr,
            })
        print()

        # Write results to a file so we can later make graphs
        pd.DataFrame({
            'expected': test_df[[output_col]].values.flatten(),
            'real': session.run(y_hat, feed_dict={
                x: test_df[input_cols].values,
                y: test_df[output_col].values
            }).flatten(),
        }).to_csv('outputs/cf-mlp-outputs-{}-{}.csv'.format(SUBJECT, architecture_str), index=None)
        pd.DataFrame(mlp_rms).to_csv('outputs/cf-mlp-rms-{}-{}.csv'.format(SUBJECT, architecture_str), index=None)
        print('Finished training')
        print('Saving model ...')
        saver = tf.train.Saver()
        saver.save(session, 'models/cf-mlp-{}-{}'.format(SUBJECT, architecture_str))
        saver.export_meta_graph('models/cf-mlp-{}-{}.meta'.format(SUBJECT, architecture_str))
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

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import load_datasets_for_subject, launch_tensorboard, convolutional

LEARNING_RATE = 0.001
LOGS_STEPS = 1
DROPOUT = 0.1
MINIBATCH_SIZE = 10000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains MLP with the set and the layers specified.')
    parser.add_argument('subject', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('steps', type=int)
    parser.add_argument('layers', nargs='*')
    args = parser.parse_args()

    # Load datasets
    datasets = load_datasets_for_subject(args.path, args.subject)
    num_inputs = len(datasets.train.data_columns)
    num_outputs = len(datasets.train.target_columns)

    # Define the network topology
    architecture = args.layers
    architecture_str = '-'.join(args.layers)

    # Create the tensorboard directories associated to this training configuration
    summary_trn_path = 'tensorboard/{}/{}/training'.format(args.subject, architecture_str)
    summary_val_path = 'tensorboard/{}/{}/validation'.format(args.subject, architecture_str)
    summary_tst_path = 'tensorboard/{}/{}/test'.format(args.subject, architecture_str)
    if os.path.exists(summary_trn_path):
        shutil.rmtree(summary_trn_path)
    if os.path.exists(summary_val_path):
        shutil.rmtree(summary_val_path)
    if os.path.exists(summary_tst_path):
        shutil.rmtree(summary_tst_path)

    print('Starting training process.')
    print('\tSubject:\t{}'.format(args.subject))
    print('\tTraining set examples:\t{}'.format(datasets.train.data.shape[0]))
    print('\tValidation set examples:\t{}'.format(datasets.validation.data.shape[0]))
    print('\tTest set examples:\t{}'.format(datasets.test.data.shape[0]))
    print('\tTraining steps:\t{}'.format(args.steps))
    print('\tDropout rate:\t{}'.format(DROPOUT))
    print('\tTopology:\t{}'.format([num_inputs] + args.layers + [num_outputs]))

    # Network
    images_start = images_end = None
    for i, val in enumerate(datasets.train.data_columns):
        if images_start is None and val.startswith('Dm'):
            images_start = i
            break
    x, y_hat, dropout = convolutional(args.layers, num_inputs, num_outputs, images_start, (360, 8, 3))
    tf.add_to_collection('output', y_hat)

    # Expected output
    y = tf.placeholder(tf.int32)

    # Training graph
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_hat)
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    tf.summary.scalar('loss', cost)
    equal_values = tf.equal(
        tf.argmax(y_hat, 1, output_type=np.int32),
        tf.argmax(y, 1, output_type=np.int32)
    )
    accuracy = tf.reduce_mean(tf.cast(equal_values, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    writer_trn = tf.summary.FileWriter(summary_trn_path)
    writer_val = tf.summary.FileWriter(summary_val_path)
    writer_tst = tf.summary.FileWriter(summary_tst_path)

    tb_process = launch_tensorboard(summary_trn_path, summary_val_path, summary_tst_path)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer_trn.add_graph(session.graph)

        mlp_rms = {
            'training': [],
            'validation': [],
            'test': [],
        }
        for step in range(args.steps):
            # Extract the training minibatch
            values_per_class = MINIBATCH_SIZE // num_outputs
            minibatch_train_data = []
            minibatch_train_target = []
            for col in range(datasets.train.target.shape[1]):
                available_idx = np.where(datasets.train.target[:, col] == 1)[0]
                if len(available_idx) > 0:
                    idx = np.random.choice(available_idx, size=values_per_class)
                    minibatch_train_data.append(datasets.train.data[idx, :])
                    minibatch_train_target.append(datasets.train.target[idx, :])
            minibatch_train_data = np.concatenate(minibatch_train_data, axis=0)
            minibatch_train_target = np.concatenate(minibatch_train_target, axis=0)

            # When logging, evaluate also with the validation partition and the test
            if step % LOGS_STEPS == 0:
                print('.', end='', flush=True)
                summary = session.run(merged_summary, feed_dict={
                    x: minibatch_train_data,
                    y: minibatch_train_target,
                })
                writer_trn.add_summary(summary, step)
                writer_trn.flush()
                mlp_rms['training'].append(session.run(cost, feed_dict={
                    x: minibatch_train_data,
                    y: minibatch_train_target,
                }))

                summary = session.run(merged_summary, feed_dict={
                    x: datasets.validation.data,
                    y: datasets.validation.target,
                })
                writer_val.add_summary(summary, step)
                writer_val.flush()
                mlp_rms['validation'].append(session.run(cost, feed_dict={
                    x: datasets.validation.data,
                    y: datasets.validation.target,
                }))

                summary = session.run(merged_summary, feed_dict={
                    x: datasets.test.data,
                    y: datasets.test.target,
                })
                writer_tst.add_summary(summary, step)
                writer_tst.flush()
                mlp_rms['test'].append(session.run(cost, feed_dict={
                    x: datasets.test.data,
                    y: datasets.test.target,
                }))

            # Train with the training partition
            session.run(train, feed_dict={
                x: minibatch_train_data,
                y: minibatch_train_target,
                dropout: DROPOUT
            })
        print()

        # Write results to a file so we can later make graphs
        pd.DataFrame({
            'expected': datasets.test.target.flatten(),
            'real': session.run(y_hat, feed_dict={
                x: datasets.test.data,
                y: datasets.test.target,
            }).flatten(),
        }).to_csv('outputs/lc-cnn-outputs-{}-{}-d{}.csv'.format(args.subject, architecture_str, DROPOUT), index=None)
        pd.DataFrame(mlp_rms).to_csv('outputs/lc-cnn-rms-{}-{}-d{}.csv'.format(args.subject, architecture_str, DROPOUT),
                                     index=None)
        print('Finished training')
        print('Saving model ...')
        saver = tf.train.Saver()
        saver.save(session, 'models/lc-cnn-{}-{}-d{}'.format(args.subject, architecture_str, DROPOUT))
        saver.export_meta_graph('models/lc-cnn-{}-{}-d{}.meta'.format(args.subject, architecture_str, DROPOUT))
        print('Saved')
    tb_process.join()

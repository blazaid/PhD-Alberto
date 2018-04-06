import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers

from utils import load_datasets_for_subject, convolutional, extract_minibatch_data, launch_tensorboard

MAX_LEARN_RATE = 0.1
MIN_LEARN_RATE = 0.001
DECAY_SPEED = 2000
ACTIVATION_FN = tf.nn.relu
OUTPUT_FN = None
DROPOUT = 0.1
EPOCHS = 10000
LOGS_STEPS = EPOCHS / 100
MINIBATCH_SIZE = 3000


if __name__ == '__main__':
    args = type('test', (object,), {})()
    args.subject = 'all'
    args.path = './data'
    args.steps = EPOCHS
    # args.layers = ['c16-4-18-v', 'd128']
    # args.layers = ['c16-4-18-v', 'c32-3-18-v', 'd128']
    args.layers = ['c16-3-18-v', 'c32-3-18-v', 'c64-2-18-v', 'd128']
    #parser = argparse.ArgumentParser(description='Trains MLP with the set and the layers specified.')
    #parser.add_argument('subject', type=str)
    #parser.add_argument('path', type=str)
    #parser.add_argument('steps', type=int)
    #parser.add_argument('layers', nargs='*')
    #args = parser.parse_args()

    # Load datasets
    datasets = load_datasets_for_subject(args.path, args.subject)
    num_inputs = len(datasets.train.data_columns)
    num_outputs = len(datasets.train.target_columns)

    # Define the network topology
    architecture = args.layers
    architecture_str = '-'.join(args.layers)

    # Create the tensorboard directories associated to this training configuration
    summary_trn_path = 'tensorboard/{}/lc-cnn-{}/training'.format(args.subject, architecture_str)
    summary_val_path = 'tensorboard/{}/lc-cnn-{}/validation'.format(args.subject, architecture_str)
    summary_tst_path = 'tensorboard/{}/lc-cnn-{}/test'.format(args.subject, architecture_str)
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
    x, y_hat, dropout = convolutional(args.layers, num_inputs, num_outputs, images_start, (8, 360, 3))
    tf.add_to_collection('output', y_hat)

    # Expected output
    y = tf.placeholder(tf.int32)

    # Training graph
    learning_rate = tf.placeholder_with_default(0.0, shape=())
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_hat)
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    tf.summary.scalar('loss', cost)
    equal_values = tf.equal(
        tf.argmax(y_hat, 1, output_type=np.int32),
        tf.argmax(y, 1, output_type=np.int32)
    )
    # Data for tensorboard
    tf.summary.scalar('learning_rate', learning_rate)
    accuracy = tf.reduce_mean(tf.cast(equal_values, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'convolution'):
        var_name, var_type = var.name.replace(':0', '').split('/')
        if 'bias' == var_type:
            tf.summary.histogram(var_type, var, family=var_name)
        elif 'kernel' == var_type:
            for filter_num in range(var.shape[3]):
                channels = var.shape[2]
                # Grayscale or color
                filter = tf.transpose(var, [3, 0, 1, 2])[filter_num:filter_num+1, :, :, :]
                if channels not in (1, 3):
                    # Whatever it is
                    filter = tf.transpose(var, [3, 0, 1, 2])[filter_num:filter_num+1, :, :, :]
                    # Sum along all the channels
                    filter = tf.reduce_sum(filter, 3, keepdims=True)
                    # Normalize being the highest weight the lowest (i.e. darkest) value
                    filter = 1 - filter / tf.reduce_max(filter)
                    # Display as image
                tf.summary.image('{}_filter_{}'.format(var_name, filter_num), filter, family=var_name)
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense'):
        var_name, var_type = var.name.replace(':0', '').split('/')
        tf.summary.histogram('{}_{}'.format(var_name, var_type), var, family=var_name)

    merged_summary = tf.summary.merge_all()

    writer_trn = tf.summary.FileWriter(summary_trn_path)
    writer_val = tf.summary.FileWriter(summary_val_path)
    writer_tst = tf.summary.FileWriter(summary_tst_path)

    tb_process = launch_tensorboard(summary_trn_path, summary_val_path, summary_tst_path)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer_trn.add_graph(session.graph)

        cnn_accuracy = {
            'training': [],
            'validation': [],
            'test': [],
        }
        for step in range(args.steps):
            # Extract the minibatches
            train_data, train_target = extract_minibatch_data(datasets.train, MINIBATCH_SIZE, num_outputs)
            validation_data, validation_target = extract_minibatch_data(datasets.validation, MINIBATCH_SIZE,
                                                                        num_outputs)
            test_data, test_target = extract_minibatch_data(datasets.validation, MINIBATCH_SIZE, num_outputs)

            # Compute the decaying learning rate
            lr = MIN_LEARN_RATE + (MAX_LEARN_RATE - MIN_LEARN_RATE) * np.math.exp(-step / DECAY_SPEED)

            # When logging, evaluate also with the validation partition and the test
            if step % LOGS_STEPS == 0:
                summary = session.run(merged_summary, feed_dict={
                    x: train_data,
                    y: train_target,
                    learning_rate: lr,
                })
                writer_trn.add_summary(summary, step)
                writer_trn.flush()
                cnn_accuracy['training'].append(session.run(accuracy, feed_dict={
                    x: train_data,
                    y: train_target,
                    learning_rate: lr,
                }))

                summary = session.run(merged_summary, feed_dict={
                    x: validation_data,
                    y: validation_target,
                    learning_rate: lr,
                })
                writer_val.add_summary(summary, step)
                writer_val.flush()
                cnn_accuracy['validation'].append(session.run(accuracy, feed_dict={
                    x: validation_data,
                    y: validation_target,
                    learning_rate: lr,
                }))

                summary = session.run(merged_summary, feed_dict={
                    x: test_data,
                    y: test_target,
                    learning_rate: lr,
                })
                writer_tst.add_summary(summary, step)
                writer_tst.flush()
                cnn_accuracy['test'].append(session.run(accuracy, feed_dict={
                    x: test_data,
                    y: test_target,
                    learning_rate: lr,
                }))

            print('training step ... ', end='', flush=True)
            start = time.process_time()
            # Train with the training partition
            session.run(train, feed_dict={
                x: train_data,
                y: train_target,
                dropout: DROPOUT,
                learning_rate: lr,
            })
            elapsed = time.process_time() - start
            remaining_time = (EPOCHS - step) * elapsed / 3600
            print('{:.2f} s. ({:.2f} hours remaining)'.format(elapsed, remaining_time))

        # Write results to a file so we can later make graphs
        real_classes = np.argmax(datasets.test.target, axis=1)
        predicted_classes = session.run(
            tf.argmax(input=y_hat, axis=1),
            feed_dict={
                x: datasets.test.data,
                y: datasets.test.target,
            })
        pd.DataFrame(
            data=np.column_stack([real_classes, predicted_classes]),
            columns=['Real classes', 'Predicted']
        ).astype(np.int32).to_csv('outputs/lc-cnn-outputs-{}-{}-d{}.csv'.format(args.subject, architecture_str, DROPOUT), index=None)
        pd.DataFrame(cnn_accuracy).to_csv(
            'outputs/lc-cnn-accuracy-{}-{}-d{}.csv'.format(args.subject, architecture_str, DROPOUT),
            index=None)
        print('Finished training')
        print('Saving model ...')
        saver = tf.train.Saver()
        saver.save(session, 'models/lc-cnn-{}-{}-d{}'.format(args.subject, architecture_str, DROPOUT))
        saver.export_meta_graph('models/lc-cnn-{}-{}-d{}.meta'.format(args.subject, architecture_str, DROPOUT))
        print('Saved')
    tb_process.join()

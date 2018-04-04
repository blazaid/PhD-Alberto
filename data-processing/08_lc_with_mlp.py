import argparse
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import load_datasets_for_subject, multilayer_perceptron, launch_tensorboard, extract_minibatch_data

MAX_LEARN_RATE = 0.1
MIN_LEARN_RATE = 0.001
DECAY_SPEED = 20000
ACTIVATION_FN = tf.nn.relu
OUTPUT_FN = None
DROPOUT = 0.1
EPOCHS = 10000
LOGS_STEPS = EPOCHS / 100
MINIBATCH_SIZE = 250000


if __name__ == '__main__':
    args = type('test', (object,), {})()
    args.subject = 'all'
    args.path = './data'
    args.steps = EPOCHS
    args.layers = [128]  # [128], [64, 64], [128, 64, 16]
    #parser = argparse.ArgumentParser(description='Trains MLP with the set and the layers specified.')
    #parser.add_argument('subject', type=str)
    #parser.add_argument('path', type=str)
    #parser.add_argument('steps', type=int)
    #parser.add_argument('layers', type=int, nargs='*')
    #args = parser.parse_args()
    # Load datasets

    datasets = load_datasets_for_subject(args.path, args.subject)
    num_inputs = len(datasets.train.data_columns)
    num_outputs = len(datasets.train.target_columns)

    # Define the network topology
    architecture = [num_inputs] + args.layers + [num_outputs]
    architecture_str = '-'.join(str(x) for x in architecture)

    # Create the tensorboard directories associated to this training configuration
    summary_trn_path = 'tensorboard/{}/lc-mlp-{}/training'.format(args.subject, architecture_str)
    summary_val_path = 'tensorboard/{}/lc-mlp-{}/validation'.format(args.subject, architecture_str)
    summary_tst_path = 'tensorboard/{}/lc-mlp-{}/test'.format(args.subject, architecture_str)
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
    x, y_hat, dropout = multilayer_perceptron(architecture, activation_fn=ACTIVATION_FN, output_fn=OUTPUT_FN)
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
    accuracy = tf.reduce_mean(tf.cast(equal_values, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', learning_rate)
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
            # Extract the minibatches
            train_data, train_target = extract_minibatch_data(datasets.train, MINIBATCH_SIZE, num_outputs)
            validation_data, validation_target = extract_minibatch_data(datasets.validation, MINIBATCH_SIZE,
                                                                        num_outputs)
            test_data, test_target = extract_minibatch_data(datasets.validation, MINIBATCH_SIZE, num_outputs)

            # Compute the decaying learning rate
            lr = MIN_LEARN_RATE + (MAX_LEARN_RATE - MIN_LEARN_RATE) * np.math.exp(-step / DECAY_SPEED)

            # When logging, evaluate also with the validation partition and the test
            if step % LOGS_STEPS == 0:
                print('l', end='', flush=True)
                summary = session.run(merged_summary, feed_dict={
                    x: train_data,
                    y: train_target,
                    learning_rate: lr,
                })
                writer_trn.add_summary(summary, step)
                writer_trn.flush()
                mlp_rms['training'].append(session.run(cost, feed_dict={
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
                mlp_rms['validation'].append(session.run(cost, feed_dict={
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
                mlp_rms['test'].append(session.run(cost, feed_dict={
                    x: test_data,
                    y: test_target,
                    learning_rate: lr,
                }))

            print('t', end='', flush=True)
            # Train with the training partition
            session.run(train, feed_dict={
                x: train_data,
                y: train_target,
                dropout: DROPOUT,
                learning_rate: lr,
            })
        print()

        # Write results to a file so we can later make graphs
        print(test_data.shape, test_target.shape, y_hat.shape)
        pd.DataFrame({
            'expected': test_target.flatten(),
            'real': session.run(y_hat, feed_dict={
                x: test_data,
                y: test_target,
            }).flatten(),
        }).to_csv('outputs/lc-mlp-outputs-{}-{}-d{}.csv'.format(args.subject, architecture_str, DROPOUT), index=None)
        pd.DataFrame(mlp_rms).to_csv('outputs/lc-mlp-rms-{}-{}-d{}.csv'.format(args.subject, architecture_str, DROPOUT),
                                     index=None)
        print('Finished training')
        print('Saving model ...')
        saver = tf.train.Saver()
        saver.save(session, 'models/lc-mlp-{}-{}-d{}'.format(args.subject, architecture_str, DROPOUT))
        saver.export_meta_graph('models/lc-mlp-{}-{}-d{}.meta'.format(args.subject, architecture_str, DROPOUT))
        print('Saved')
    tb_process.join()

import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from tfz import IVar, OVar, fuzzy_controller

SUBJECT = 'miguel'
DATASETS_PATH = '/media/blazaid/Saca/Phd/data/datasets'
MOMENTS = 't-t5-t10-t20'
LEARNING_RATE = 0.01
TRAIN_STEPS = 10000
LOGS_STEPS = 100

tensorboard_path = 'tmp'

if __name__ == '__main__':
    train_file = os.path.join(DATASETS_PATH, 'cf-{}-training-t-t5-t10-t20.csv'.format(SUBJECT))
    test_file = os.path.join(DATASETS_PATH, 'cf-{}-validation-t-t5-t10-t20.csv'.format(SUBJECT))
    train_df = pd.read_csv(train_file, index_col=False).astype(np.float32)
    test_df = pd.read_csv(test_file, index_col=False).astype(np.float32)

    tf.reset_default_graph()

    # Input variables
    leader_distance = IVar(name='LeaderDist', fuzzy_sets=3, domain=(0., 1.))
    next_tls_distance = IVar(name='NextTlsDist', fuzzy_sets=4, domain=(0., 1.))
    next_tls_status = IVar(name='NextTlsStatus', fuzzy_sets=3, domain=(0., 1.))
    relative_speed = IVar(name='RelativeSpeed', fuzzy_sets=3, domain=(0., 2.))
    speed_to_leader = IVar(name='SpeedToLeader', fuzzy_sets=3, domain=(-1., 1.))

    # Output variable
    acceleration = OVar(name='Acceleration', values=(-1, 1))

    # Controller
    x, y_hat = fuzzy_controller(
        i_vars=[leader_distance, next_tls_distance, next_tls_status, relative_speed, speed_to_leader],
        o_var=acceleration
    )

    # Training process
    y = tf.placeholder(tf.float32)
    with tf.name_scope('train'):
        cost = tf.reduce_mean(tf.squared_difference(y, y_hat))
        train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
        tf.summary.scalar('cost', cost)

    merged_summary = tf.summary.merge_all()
    if os.path.exists(tensorboard_path + '/training'):
        shutil.rmtree(tensorboard_path + '/training')
    if os.path.exists(tensorboard_path + '/test'):
        shutil.rmtree(tensorboard_path + '/test')

    writer_trn = tf.summary.FileWriter(tensorboard_path + '/training')
    writer_tst = tf.summary.FileWriter(tensorboard_path + '/test')

    inputs = train_df[
        ['Leader distance', 'Next TLS distance', 'Next TLS status', 'Relative speed', 'Speed to leader']
    ].values
    output = train_df[['Acceleration']].values
    inputs_test = test_df[
        ['Leader distance', 'Next TLS distance', 'Next TLS status', 'Relative speed', 'Speed to leader']
    ].values
    output_test = test_df[['Acceleration']].values

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        writer_trn.add_graph(session.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard: tensorboard --logdir {}".format(datetime.now(), tensorboard_path + '/training'))
        print("{} Open http://0.0.0.0:6006/ into your web browser".format(datetime.now()))

        train_feed_dict = {x: inputs, y: output}
        test_feed_dict = {x: inputs_test, y: output_test}
        for step in range(TRAIN_STEPS):
            session.run(train, feed_dict=train_feed_dict)
            if TRAIN_STEPS % LOGS_STEPS == 0:
                summary = session.run(merged_summary, feed_dict=train_feed_dict)
                writer_trn.add_summary(summary, step)
                writer_trn.flush()
                summary = session.run(merged_summary, feed_dict=test_feed_dict)
                writer_tst.add_summary(summary, step)
                writer_tst.flush()

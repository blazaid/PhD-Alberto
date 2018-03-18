import os
from pprint import pprint

import numpy as np
import pandas as pd
import tensorflow as tf

from tfz import IVar, OVar, fuzzy_controller

SUBJECT = 'miguel'
DATASETS_PATH = '/media/blazaid/Saca/Phd/data/datasets'
MOMENTS = 't-t5-t10-t20'
LEARNING_RATE = 0.01
TRAIN_STEPS = 1000
LOGS_STEPS = 100

if __name__ == '__main__':
    train_file = os.path.join(DATASETS_PATH, 'cf-{}-training-t-t5-t10-t20.csv'.format(SUBJECT))
    test_file = os.path.join(DATASETS_PATH, 'cf-{}-validation-t-t5-t10-t20.csv'.format(SUBJECT))
    train_df = pd.read_csv(train_file, index_col=False).astype(np.float32)

    tf.reset_default_graph()

    # Input variables
    leader_distance = IVar(name='LeaderDist', fuzzy_sets=3, domain=(0., 1.))
    next_tls_distance = IVar(name='NextTlsDist', fuzzy_sets=3, domain=(0., 1.))
    next_tls_status = IVar(name='NextTlsStatus', fuzzy_sets=3, domain=(0., 1.))
    relative_speed = IVar(name='RelativeSpeed', fuzzy_sets=3, domain=(0., 2.))
    speed_to_leader = IVar(name='SpeedToLeader', fuzzy_sets=3, domain=(-50., 50.))

    # Output variable
    acceleration = OVar(name='Acceleration', values=(-1, 1))

    # Controller
    x, y_hat = fuzzy_controller(
        i_vars=[leader_distance, next_tls_distance, next_tls_status, relative_speed, speed_to_leader],
        o_var=acceleration
    )

    # Training process
    y = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.squared_difference(y, y_hat))
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    inputs = train_df[
        ['Leader distance', 'Next TLS distance', 'Next TLS status', 'Relative speed', 'Speed to leader']
    ].values
    output = train_df[['Acceleration']].values

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        feed_dict = {x: inputs, y: output}

        for step in range(TRAIN_STEPS):
            session.run(train, feed_dict=feed_dict)
            if TRAIN_STEPS % LOGS_STEPS == 0:
                pass

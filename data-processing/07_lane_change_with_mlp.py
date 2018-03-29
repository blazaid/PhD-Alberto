import argparse
import os
import shutil
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.training.adam import AdamOptimizer

LEARNING_RATE = 0.01
MODEL_DIR = '/tmp/tf-model/'
OVERWRITE_PREVIOUS_MODEL = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains MLP with the set and the layers specified.')
    parser.add_argument('subject', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('steps', type=int)
    parser.add_argument('layers', type=int, nargs='+')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate for the network')
    args = parser.parse_args()

    print('Starting training process.')
    print('\tTraining set:\t{}'.format(args.training_file))
    print('\tTraining set size:\t{}'.format(len(training_set.data)))
    print('\tValidation set:\t{}'.format(args.validation_file))
    print('\tValidation set size:\t{}'.format(len(validation_set.data)))
    print('\tTraining steps:\t{}'.format(args.steps))
    print('\tDropout rate:\t{}'.format(args.dropout))
    print('\tTopology:\t{}'.format([num_inputs] + args.layers + [num_classes]))

    # Input columns
    feature_columns = [
        tf.feature_column.numeric_column('features', shape=[num_inputs])
    ]

    # We create the classifier
    if os.path.exists(MODEL_DIR) and OVERWRITE_PREVIOUS_MODEL:
        shutil.rmtree(MODEL_DIR)
    classifier = DNNClassifier(
        hidden_units=args.layers,
        feature_columns=feature_columns,
        model_dir=MODEL_DIR,
        n_classes=3,
        dropout=args.dropout or None,
        optimizer=AdamOptimizer(
            learning_rate=0.001
        ),

    )

    # Now we train it
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'features': np.array(training_set.data)},
        y=np.array(training_set.target),
        batch_size=1000,
        num_epochs=None,
        shuffle=True,
    )
    training_time = time.time()
    classifier.train(
        input_fn=train_input_fn,
        steps=args.steps,
    )
    training_time = time.time() - training_time

    # ... and evaluate it in both training set and validation set
    training_for_evaluation_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'features': np.array(training_set.data)},
        y=np.array(training_set.target),
        shuffle=False,
    )
    evaluations = classifier.evaluate(input_fn=training_for_evaluation_input_fn)
    trn_accuracy = evaluations['accuracy']
    validation_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'features': np.array(validation_set.data)},
        y=np.array(validation_set.target),
        shuffle=False,
    )
    evaluations = classifier.evaluate(input_fn=validation_input_fn)
    val_accuracy = evaluations['accuracy']

    # Generate the prediction results
    filename_prefix = os.path.join(
        results_dir,
        'dm_mlp_m{}_i{}_c{}_h{}_s{}_d{}'.format(
            get_moments_before(args.training_file),
            num_inputs,
            num_classes,
            ','.join(str(x) for x in args.layers),
            args.steps,
            args.dropout,
        )
    )

    print('Network results for training set: ' + filename_prefix + '-trn.csv')

    with open(filename_prefix + '-trn.csv', 'w') as f:
        f.write('expected_output,network_output\n')
        o_expected = training_set.target.tolist()
        predicting_time = time.time()
        o_real = classifier.predict(input_fn=training_for_evaluation_input_fn)
        predicting_time = time.time() - predicting_time
        o_real = list(o_real)
        predicting_time = predicting_time / len(o_real)
        for expected, network in zip(o_expected, o_real):
            f.write('{},{}\n'.format(expected, int(network['classes'][0])))

    print('Network results for validation set: ' + filename_prefix + '-val.csv')
    with open(filename_prefix + '-val.csv', 'w') as f:
        f.write('expected_output,network_output\n')
        o_expected = validation_set.target.tolist()
        o_real = list(classifier.predict(input_fn=validation_input_fn))
        for expected, network in zip(o_expected, o_real):
            f.write('{},{}\n'.format(expected, int(network['classes'][0])))

    print('Accuracy results: ' + filename_prefix + '-kpi.csv')
    with open(filename_prefix + '-kpi.csv', 'w') as f:
        f.write('training_accuracy,validation_accuracy, training_time, predicting_time\n')
        f.write('{},{},{},{}\n'.format(trn_accuracy, val_accuracy, training_time, predicting_time))
        print('\tTraining:\t{}'.format(trn_accuracy))
        print('\tValidation:\t{}'.format(val_accuracy))
        print('\tTraining time:\t{}'.format(training_time))
        print('\tPrediction time:\t{}'.format(predicting_time))

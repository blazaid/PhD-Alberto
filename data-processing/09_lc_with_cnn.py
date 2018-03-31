import argparse
import os
import shutil
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import \
    load_csv_without_header
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.training.adam import AdamOptimizer


def multilayer_perceptron(layers, activation_fn=None):
    # The placeholder to activate or deactivate the dropout. Defaults to inactive"
    dropout_active = tf.placeholder_with_default(False, shape=())

    activation_fn = activation_fn or tf.nn.relu
    x = tf.placeholder(tf.float32, name='input', shape=[None, layers[0]])
    output = x
    for layer_id, num_neurons in enumerate(layers[1:], start=1):
        output = tf.layers.dense(inputs=output, units=num_neurons, activation=activation_fn)
        output = tf.layers.dropout(output, training=dropout_active)
    return x, output, dropout_active


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains MLP with the set and the layers specified.')
    parser.add_argument('subject', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('steps', type=int)
    parser.add_argument('layers', type=int, nargs='+')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate for the network')
    args = parser.parse_args()

    x, y_hat = multilayer_perceptron(architecture, activation_fn=ACTIVATION_FUNCTION)
    tf.add_to_collection('output', y_hat)

    # Expected output
    y = tf.placeholder(tf.float32)

    # Training graph
    cost = tf.reduce_mean(tf.squared_difference(y, y_hat))
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    import argparse
    import os
    import shutil
    import time

    import numpy as np
    import tensorflow as tf
    from tensorflow.contrib.learn.python.learn.datasets.base import load_csv_without_header
    from tensorflow.python.training.adam import AdamOptimizer

    from settings import results_dir, deep_map_shape
    from utils import get_moments_before

    MODEL_DIR = '/tmp/tf-model/'
    OVERWRITE_PREVIOUS_MODEL = True

    tf.logging.set_verbosity(tf.logging.INFO)


    def check_architecture(layers):
        """ Unfolds the architecture to feed the model.

        It'll check if the architecture is correct. That is, starts with a convolution layer, follow with any set of
        convolutions, pollings and so and ends with one-to-many dense layers.
        """
        # Checking layers architecture order
        if layers[0][0] != 'c':
            raise argparse.ArgumentTypeError('First layers should be a convolutional one')
        if layers[-1][0] != 'd':
            raise argparse.ArgumentTypeError('Last layer should be a dense one')
        found_dense_layer = False
        for layer in layers:
            layer_type = layer[0]
            if found_dense_layer and layer_type != 'd':
                raise argparse.ArgumentTypeError('Al the dense layers should be at the end'.format(f))
            elif layer_type == 'd':
                found_dense_layer = True

        return layers


    def split_intentions_and_images(dataset, columns_per_image, channels):
        """ Returns a dictionary of two datasets, the one with the intentions
        and the other with the image columns.
        """
        images_columns = channels * columns_per_image
        return {
            'images': np.array([ts[-images_columns:] for ts in dataset]),
            'intentions': np.array([ts[:-images_columns] for ts in dataset]),
        }


    def dm_cnn(features, labels, mode, params):
        """Model function for CNN."""

        def build_convolution_layer(inputs, desc):
            filters, rows, cols = desc[1:].split('-')
            filters, rows, cols = int(filters), int(rows), int(cols)
            return tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[rows, cols],
                padding="same",
                activation=tf.nn.relu
            )

        def build_pooling_layer(inputs, desc):
            rows, cols, stride = desc[1:].split('-')
            rows, cols, stride = int(rows), int(cols), int(stride)

            return tf.layers.max_pooling2d(inputs=inputs, pool_size=[rows, cols], strides=stride)

        def build_dense_layer(inputs, desc):
            units, dropout = desc[1:].split('-')
            units, dropout = int(units), float(dropout)
            outputs = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu)
            if dropout > 0.0:
                outputs = tf.layers.dropout(inputs=outputs, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
            return outputs

        build_layer_functions = {
            'd': build_dense_layer,
            'c': build_convolution_layer,
            'p': build_pooling_layer,
        }

        patterns_layers = [layer for layer in params['architecture'] if layer[0] != 'd']
        dense_layers = [layer for layer in params['architecture'] if layer[0] == 'd']

        # Input layer is the first layer
        layer = tf.reshape(features['images'], params['images_tensor_shape'])

        # Patterns layer
        for layer_description in patterns_layers:
            layer = build_layer_functions[layer_description[0]](layer, layer_description)

        # Flatten
        layer = tf.reshape(layer, [-1, int(np.prod(layer.shape[1:]))])
        layer = tf.concat([layer, features['images']], 1)

        # Dense Layers
        for layer_description in dense_layers:
            layer = build_layer_functions[layer_description[0]](layer, layer_description)

        # Logits Layer
        logits = tf.layers.dense(inputs=layer, units=params['classes'])

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
            # "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params['classes'])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # TODO set the optimizer configurable
            optimizer = params['optimizer']
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    if __name__ == '__main__':
        parser = argparse.ArgumentParser(
            description='Trains CNN for a deepness map with the set and the topology specified.'
        )
        parser.add_argument('training_file', type=str)
        parser.add_argument('validation_file', type=str)
        parser.add_argument('steps', type=int)
        parser.add_argument('arch', type=str, nargs='+')
        args = parser.parse_args()

        # Load datasets
        training_set = load_csv_without_header(
            filename=args.training_file,
            target_dtype=np.int32,
            features_dtype=np.float32
        )
        validation_set = load_csv_without_header(
            filename=args.validation_file,
            target_dtype=np.int32,
            features_dtype=np.float32
        )

        num_inputs = len(training_set.data[0])
        num_classes = len(set(training_set.target))

        print('Starting training process.')
        print('\tTraining set:\t{}'.format(args.training_file))
        print('\tTraining set size:\t{}'.format(len(training_set.data)))
        print('\tValidation set:\t{}'.format(args.validation_file))
        print('\tValidation set size:\t{}'.format(len(validation_set.data)))
        print('\tTraining steps:\t{}'.format(args.steps))
        print('\tArchitecture:\t{}'.format(args.arch))

        channels = len(training_set.data[0]) // np.prod(deep_map_shape)
        columns_per_image = np.prod(deep_map_shape)

        # We create the classifier
        if os.path.exists(MODEL_DIR) and OVERWRITE_PREVIOUS_MODEL:
            shutil.rmtree(MODEL_DIR)
        # Create the Estimator
        classifier = tf.estimator.Estimator(
            model_fn=dm_cnn,
            model_dir=MODEL_DIR,
            params={
                'images_tensor_shape': [-1] + list(deep_map_shape) + [channels],
                'channels': channels,
                'architecture': check_architecture(args.arch),
                'classes': num_classes,
                'optimizer': AdamOptimizer(
                    learning_rate=0.001
                ),
            }
        )

        # Now we train it
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=split_intentions_and_images(training_set.data, columns_per_image, channels),
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
            x=split_intentions_and_images(training_set.data, columns_per_image, channels),
            y=np.array(training_set.target),
            shuffle=False,
        )
        evaluations = classifier.evaluate(input_fn=training_for_evaluation_input_fn)
        trn_accuracy = evaluations['accuracy']
        validation_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=split_intentions_and_images(validation_set.data, columns_per_image, channels),
            y=np.array(validation_set.target),
            shuffle=False,
        )
        evaluations = classifier.evaluate(input_fn=validation_input_fn)
        val_accuracy = evaluations['accuracy']

        # Generate the prediction results
        filename_prefix = os.path.join(
            results_dir,
            'dm_cnn_m{}_i{}_c{}_t{}_s{}'.format(
                get_moments_before(args.training_file),
                num_inputs,
                num_classes,
                ','.join(str(x) for x in args.arch),
                args.steps,
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
                f.write('{},{}\n'.format(expected, int(network['classes'])))

        print('Network results for validation set: ' + filename_prefix + '-val.csv')
        with open(filename_prefix + '-val.csv', 'w') as f:
            f.write('expected_output,network_output\n')
            o_expected = validation_set.target.tolist()
            o_real = list(classifier.predict(input_fn=validation_input_fn))
            for expected, network in zip(o_expected, o_real):
                f.write('{},{}\n'.format(expected, int(network['classes'])))

        print('Accuracy results: ' + filename_prefix + '-kpi.csv')
        with open(filename_prefix + '-kpi.csv', 'w') as f:
            f.write('training_accuracy,validation_accuracy, training_time, predicting_time\n')
            f.write('{},{},{},{}\n'.format(trn_accuracy, val_accuracy, training_time, predicting_time))
            print('\tTraining:\t{}'.format(trn_accuracy))
            print('\tValidation:\t{}'.format(val_accuracy))
            print('\tTraining time:\t{}'.format(training_time))
            print('\tPrediction time:\t{}'.format(predicting_time))

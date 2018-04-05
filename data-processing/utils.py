import os
import random
from multiprocessing import Process

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorboard.default
import tensorboard.program
import tensorflow as tf


def load_master_df(base_path, dataset, kind):
    """
    :param kind: One of "detours", "sections", "tls" or "yields".
    """
    kinds = ["detours", "sections", "tls", "yields"]
    if kind not in kinds:
        raise ValueError('parameter kind should be one of {}'.format(kinds))

    filename = '{}_{}.csv'.format(dataset, kind)
    return pd.read_csv(os.path.join(base_path, 'routes_data', filename))


def load_subject_df(base_path, subject, dataset, kind):
    """
    :param kind: One of "dataset" or "tls"
    """
    kinds = ["dataset", "tls"]
    if kind not in kinds:
        raise ValueError('parameter kind should be one of {}'.format(kinds))

    filename = '{}.csv'.format(kind)
    return pd.read_csv(os.path.join(base_path, subject, dataset, filename))


DATASETS_INFO = {
    'edgar': {
        'training': {
            'starting_frame': 1078,
            'ending_frame': 9609,
            'lane_changes': [
                (1119, 1135, 1), (2240, 2260, -1), (2246, 2264, 1), (2499, 2516, -1),
                (2999, 3020, 1), (6095, 6115, -1), (6395, 6415, 1), (6715, 6734, 1),
                (6764, 6784, -1), (6876, 6897, -1), (8460, 8480, 1), (8526, 8537, -1),
            ],
            'skippable_sequences': None,
            'max_speed': [(0, 50), (1089, 40), (2960, 50), (6536, 40), ],
            'lanes_distances': [
                (0, 0, np.inf, 0),
                (940, np.inf, np.inf, 0),
                (1129, 0, np.inf, np.inf),
                (1290, (40.383074, -3.630246), np.inf, np.inf),
                (1317, 0, np.inf, np.inf),
                (1425, (40.383933, -3.632542), np.inf, np.inf),
                (1770, 0, np.inf, np.inf),
                (1840, 0, (40.384765, -3.633968), np.inf),
                (1883, (40.384765, -3.633968), np.inf, np.inf),
                (1895, 0, (40.384765, -3.633968), np.inf),
                (1943, (40.384765, -3.633968), np.inf, np.inf),
                (1950, 0, np.inf, np.inf),
                (2080, np.inf, np.inf, 0),
                (2090, (40.385600, -3.635954), np.inf, np.inf),
                (2099, 0, (40.385600, -3.635954), np.inf),
                (2140, (40.385600, -3.635954), np.inf, np.inf),
                (2148, 0, (40.387391, -3.640122), np.inf),
                (2255, (40.387391, -3.640122), np.inf, 0),
                (2454, 0, (40.387391, -3.640122), np.inf),
                (2509, (40.387391, -3.640122), np.inf, 0),
                (2914, (40.387391, -3.640122), np.inf, (40.388916, -3.639228)),
                (2933, np.inf, (40.388916, -3.639228), 0),
                (3012, 0, np.inf, (40.388916, -3.639228)),
                (3250, 0, np.inf, 0),
                (3768, (40.389074, -3.644033), np.inf, (40.387686, -3.645143)),
                (4690, 0, np.inf, (40.387686, -3.645143)),
                (5085, 0, np.inf, np.inf),
                (6100, np.inf, 0, np.inf),
                (6406, 0, np.inf, np.inf),
                (6448, np.inf, np.inf, 0),
                (6724, 0, np.inf, np.inf),
                (6776, np.inf, np.inf, 0),
                (6850, (40.385502, -3.635610), np.inf, np.inf),
                (6888, np.inf, np.inf, 0),
                (8477, 0, np.inf, np.inf)
            ],
            'cf_dist': [
                (1440, 1520, 1, 2),
                (1690, 1830, 1, 2),
                (2250, 2375, 1, 2),
                (2470, 2605, 1, 2),
                (3120, 3245, 1, 2),
                (4810, 4895, 1, 2),
                (4990, 5050, 1, 2),
                (6130, 6220, 1, 2),
                (6250, 6350, 1, 2),
                (8220, 8380, 1, 2),
            ],
            'calibration_data': {
                'rot_y': 3.1,
                'rot_z': -3.1,
            },
        },
        'validation': {
            'starting_frame': 61,
            'ending_frame': 2687,
            'lane_changes': [
                (370, 384, 1), (1506, 1523, 1), (1588, 1600, -1), (2510, 2522, 1),
                (2580, 2591, 1), (2623, 2649, -1), (3003, 3020, 1), (3070, 3086, -1),
            ],
            'skippable_sequences': None,
            'max_speed': [(0, 40), (370, 50)],
            'lanes_distances': [
                (0, (40.381525, -3.624439), np.inf, 0),
                (205, (40.380804, -3.625603), np.inf, 0),
                (375, 0, (40.380804, -3.625603), np.inf),
                (518, (40.380804, -3.625603), np.inf, 0),
                (528, 0, (40.380804, -3.625603), np.inf),
                (573, (40.380804, -3.625603), np.inf, 0),
                (582, 0, np.inf, 0),
                (1055, (40.376785, -3.630526), np.inf, 0),
                (1064, 0, (40.376785, -3.630526), np.inf),
                (1103, (40.376785, -3.630526), np.inf, np.inf),
                (1114, (40.373363, -3.637526), np.inf, 0),
                (1515, 0, (40.373363, -3.637526), np.inf),
                (1597, (40.373363, -3.637526), np.inf, np.inf),
                (1635, np.inf, (40.376865, -3.639101), 0),
                (2522, np.inf, np.inf, (40.376865, -3.639101)),
                (2587, 0, np.inf, np.inf),
                (2587, np.inf, np.inf, 0),
            ],
            'cf_dist': [
                (0, 15, 1, 2),
                (150, 250, 1, 2),
                (1470, 1500, 1, 2),
            ],
            'calibration_data': {
                'rot_y': 3.1,
                'rot_z': -3.1,
            },
        },
    },
    'jj': {
        'training': {
            'starting_frame': 150,
            'ending_frame': 10580,
            'lane_changes': [
                (1560, 1575, 1), (2510, 2530, -1), (3115, 3130, 1), (3140, 3151, -1),
                (3173, 3190, 1), (5452, 5475, 1), (6926, 6947, 1), (7360, 7385, 1),
                (7565, 7584, -1), (7726, 7750, 1), (8699, 8714, 1), (8735, 8755, -1),
                (10010, 10033, -1), (10082, 10093, 1), (10112, 10135, -1),
            ],
            'skippable_sequences': [(5600, 6878), ],
            'max_speed': [(0, 50), (1065, 40), (3000, 50), (8562, 40), ],
            'lanes_distances': [
                (0, 0, np.inf, 0),
                (874, np.inf, np.inf, 0),
                (1420, (40.383933, -3.632542), np.inf, np.inf),
                (1695, 0, np.inf, np.inf),
                (1815, (40.384765, -3.633968), np.inf, np.inf),
                (1904, np.inf, np.inf, 0),
                (1911, (40.384765, -3.633968), np.inf, np.inf),
                (1919, 0, (40.384765, -3.633968), np.inf),
                (1975, (40.384765, -3.633968), np.inf, np.inf),
                (1983, 0, np.inf, np.inf),
                (2142, np.inf, np.inf, 0),
                (2155, (40.385600, -3.635954), np.inf, np.inf),
                (2162, 0, (40.385600, -3.635954), np.inf),
                (2214, (40.385600, -3.635954), np.inf, np.inf),
                (2227, 0, (40.387391, -3.640122), np.inf),
                (2377, (40.387391, -3.640122), np.inf, 0),
                (2950, np.inf, (40.388916, -3.639228), 0),
                (2967, 0, np.inf, (40.388916, -3.639228)),
                (2990, np.inf, (40.388916, -3.639228), 0),
                (3037, 0, np.inf, (40.388916, -3.639228)),
                (3565, 0, np.inf, 0),
                (4690, (40.389074, -3.644033), np.inf, (40.387686, -3.645143)),
                (5285, np.inf, (40.387686, -3.645143), 0),
                (5317, 0, np.inf, (40.387686, -3.645143)),
                (6718, 0, np.inf, (40.387686, -3.645143)),
                (7224, np.inf, np.inf, 0),
                (6792, 0, np.inf, np.inf),
                (7150, np.inf, np.inf, 0),
                (7227, 0, np.inf, np.inf),
                (7430, np.inf, np.inf, 0),
                (8560, 0, np.inf, np.inf),
                (8606, np.inf, np.inf, 0),
                (8932, (40.385502, -3.635610), np.inf, np.inf),
                (9008, np.inf, np.inf, 0),
                (9023, (40.385502, -3.635610), np.inf, np.inf),
                (9080, np.inf, np.inf, 0),
                (9092, 0, np.inf, np.inf),
                (9297, np.inf, np.inf, 0),
                (9304, (40.384547, -3.633746), np.inf, np.inf),
                (9315, 0, (40.384547, -3.633746), np.inf),
                (9355, (40.384547, -3.633746), np.inf, np.inf),
                (8370, np.inf, np.inf, 0),
                (9377, 0, np.inf, np.inf),
                (9876, np.inf, np.inf, 0),
                (9937, 0, np.inf, np.inf),
                (9975, np.inf, np.inf, 0),
            ],
            'cf_dist': [
                (1420, 1710, 1, 2),
                (2439, 2600, 1, 2),
                (5000, 5082, 1, 2),
                (5170, 5420, 1, 2),
                (7270, 7375, 1, 2),
                (9350, 9780, 1, 2),
            ],
            'calibration_data': {
                'rot_y': 3.1,
                'rot_z': -3.1,
            },
        },
        'validation': {
            'starting_frame': 512,
            'ending_frame': 4142,
            'lane_changes': [
                (3115, 3142, 1), (3540, 3570, -1),
            ],
            'skippable_sequences': None,
            'max_speed': [(0, 40), (780, 50)],
            'lanes_distances': [
                (0, (40.381525, -3.624439), np.inf, 0),
                (550, (40.380804, -3.625603), np.inf, 0),
                (1008, 0, np.inf, 0),
                (2382, (40.373363, -3.637526), np.inf, 0),
                (2382, (40.373363, -3.637526), np.inf, 0),
                (3020, (40.373363, -3.637526), np.inf, (40.376865, -3.639101)),
                (3040, np.inf, (40.376865, -3.639101), 0),
                (3138, (40.374415, -3.638122), np.inf, (40.376865, -3.639101)),
                (3350, 0, np.inf, (40.376865, -3.639101)),
                (3437, (40.375614, -3.638688), np.inf, (40.376865, -3.639101)),
                (3515, np.inf, np.inf, (40.376865, -3.639101)),
                (3523, 0, np.inf, np.inf),
                (3560, np.inf, np.inf, (40.376865, -3.639101)),
            ],
            'cf_dist': [
                (0, 205, 1, 2),
                (420, 500, 1, 2),
                (1020, 1318, 1, 2),
                (1319, 1430, 1, 2),
                (1705, 2075, 1, 2),
            ],
            'calibration_data': {
                'rot_y': 3.1,
                'rot_z': -3.1,
            },
        },
    },
    'miguel': {
        'training': {
            'starting_frame': 100,
            'ending_frame': 10232,
            'lane_changes': [
                (1532, 1550, 1), (3510, 3530, 1), (3560, 3572, -1), (3580, 3602, 1),
                (3613, 3630, -1), (5239, 5249, 1), (6050, 6068, 1), (6190, 6211, -1),
                (6279, 6302, 1), (6510, 6529, 1), (6550, 6580, -1), (8729, 8755, 1),
                (9499, 9520, -1), (9830, 9851, 1), (10129, 10149, -1),
            ],
            'skippable_sequences': None,
            'max_speed': [(0, 50), (1400, 40), (3000, 50), (7845, 40), ],
            'lanes_distances': [
                (0, 0, np.inf, 0),
                (1147, np.inf, np.inf, 0),
                (1546, 0, np.inf, np.inf),
                (1600, (40.383074, -3.630246), np.inf, np.inf),
                (1619, 0, np.inf, np.inf),
                (1702, (40.383933, -3.632542), np.inf, np.inf),
                (1759, 0, np.inf, np.inf),
                (1767, np.inf, np.inf, 0),
                (1804, (40.384765, -3.633968), np.inf, np.inf),
                (1844, np.inf, np.inf, 0),
                (1851, (40.384765, -3.633968), np.inf, np.inf),
                (1857, 0, (40.384765, -3.633968), np.inf),
                (1885, (40.384765, -3.633968), np.inf, np.inf),
                (1893, np.inf, np.inf, 0),
                (2172, (40.385600, -3.635954), np.inf, np.inf),
                (2230, (40.387391, -3.640122), np.inf, 0),
                (2956, np.inf, (40.388916, -3.639228), 0),
                (3210, 0, np.inf, (40.388916, -3.639228)),
                (3385, 0, np.inf, 0),
                (4535, (40.389074, -3.644033), np.inf, (40.387686, -3.645143)),
                (4535, 0, np.inf, (40.387686, -3.645143)),
                (5700, np.inf, np.inf, 0),
                (6064, 0, np.inf, np.inf),
                (6207, np.inf, np.inf, 0),
                (6295, 0, np.inf, np.inf),
                (6470, np.inf, np.inf, 0),
                (6527, 0, np.inf, np.inf),
                (6568, np.inf, np.inf, 0),
                (8462, (40.385502, -3.635610), np.inf, np.inf),
                (8547, np.inf, np.inf, 0),
                (8561, (40.385502, -3.635610), np.inf, np.inf),
                (8636, np.inf, np.inf, 0),
                (8752, 0, np.inf, np.inf),
                (8912, np.inf, np.inf, 0),
                (8919, (40.384547, -3.633746), np.inf, np.inf),
                (8930, 0, (40.384547, -3.633746), np.inf),
                (8971, (40.384547, -3.633746), np.inf, np.inf),
                (8979, np.inf, np.inf, 0),
                (8983, 0, np.inf, np.inf),
                (9513, np.inf, np.inf, 0),
                (9846, 0, np.inf, np.inf),
                (9513, np.inf, np.inf, 0),
            ],
            'cf_dist': [
                (15, 120, 1, 2),
                (140, 240, 1, 2),
                (1996, 2400, 1, 2),
                (3275, 3351, 1, 2),
                (3812, 3965, 1, 2),
                (4399, 4445, 1, 2),
                (4785, 4880, 1, 2),
                (5090, 5210, 1, 2),
                (5250, 5390, 1, 2),
                (7950, 8400, 1, 2),
                (9530, 9720, 1, 2),
                (9865, 10065, 1, 2),
            ],
            'calibration_data': {
                'rot_y': 3.1,
                'rot_z': -3.1,
            },
        },
        'validation': {
            'starting_frame': 44,
            'ending_frame': 3651,
            'lane_changes': [
                (880, 898, 1), (2820, 2839, 1), (2844, 2867, -1), (3462, 3492, 1),
                (4140, 4158, 1)
            ],
            'skippable_sequences': None,
            'max_speed': [(0, 40), (600, 50)],
            'lanes_distances': [
                (0, (40.381525, -3.624439), np.inf, 0),
                (550, (40.380804, -3.625603), np.inf, 0),
                (920, 0, (40.380804, -3.625603), np.inf),
                (966, (40.380804, -3.625603), np.inf, 0),
                (975, 0, np.inf, 0),
                (2040, np.inf, np.inf, 0),
                (2062, (40.376785, -3.630526), np.inf, np.inf),
                (2114, (40.373363, -3.637526), np.inf, 0),
                (2647, (40.373363, -3.637526), np.inf, np.inf),
                (2690, np.inf, (40.376865, -3.639101), 0),
                (2690, np.inf, (40.376865, -3.639101), 0),
                (2840, 0, np.inf, (40.376865, -3.639101)),
                (2857, np.inf, (40.376865, -3.639101), 0),
                (3471, np.inf, np.inf, (40.376865, -3.639101)),
            ],
            'cf_dist': [
                (1304, 1370, 1, 2),
                (1410, 1485, 1, 2),
                (1692, 1765, 1, 2),
                (1880, 1933, 1, 2),
                (1970, 2050, 1, 2),
                (2950, 3010, 1, 2),
                (3390, 3460, 1, 2),
            ],
            'calibration_data': {
                'rot_y': 3.1,
                'rot_z': -3.1,
            },
        },
    },
}


def plot_mfs(ax, X, Y, stage='first', **kwargs):
    """
    
    :param stage: 'first', 'train' or 'final'.
    """
    stage_styles = {
        'first': {'linewidth': 3, 'alpha': 0.4},
        'train': {'linewidth': 0.2, 'c': '0.6'},
        'final': {'linewidth': 2},
    }
    for i, column in enumerate(Y.T):
        # Select the style of the line according to the parameter stage
        mf_params = stage_styles[stage].copy()
        # Select the color index from the style cmap
        if 'c' not in mf_params:
            mf_params['c'] = 'C{}'.format(i % 10)
        # Get the label for this membership function (if any). If not, set the
        # default value.
        if 'labels' in kwargs and len(kwargs['labels']) > i:
            mf_params['label'] = kwargs['labels'][i]
        # Plot the membership function
        if 'singletons' in kwargs and kwargs['singletons'][i] is not None:
            mf_params['colors'] = mf_params.pop('c')
            ax.vlines(kwargs['singletons'][i], 0, 1, **mf_params);
        else:
            ax.plot(X[:, 0], column, **mf_params);
    ax.plot(X, [0 for _ in X], linewidth=2, c='0.0');


def plot_lvar(lvar, res=1000):
    mf_names = [fs for fs in lvar]
    singletons = [
        lvar[fs].a if isinstance(lvar[fs], fuzzle.mfs.SingletonMF) else None
        for fs in mf_names
    ]

    X = np.linspace(*lvar.domain, res)
    X = np.concatenate(([-X[1]], X, [X[-1] + X[1] - X[0]]))
    Y = []

    for x in X:
        Y.append([lvar[mf_name](x) for mf_name in mf_names])
    X = np.reshape(X, (-1, 1))
    Y = np.array(Y)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(lvar.name)
    plot_mfs(ax, X, Y, stage='final', labels=mf_names, singletons=singletons)


original_plot_params = {'linewidth': 10, 'alpha': 0.4, 'c': 'b'}
original_train_params = {'linewidth': 0.2, 'alpha': 0.9, 'c': 'r'}
first_result_params = {'linewidth': 10, 'alpha': 0.4, 'c': 'r'}
final_result_params = {'linewidth': 2, 'c': 'b'}


def launch_tensorboard(tb_trn_path, tb_val_path, tb_tst_path):
    def process(path_trn, path_val, path_tst):
        tensorboard.program.FLAGS.logdir = 'training:{},validation:{},test:{}'.format(path_trn, path_val, path_tst)
        tensorboard.program.main(tensorboard.default.get_plugins(), tensorboard.default.get_assets_zip_provider())
        print('Tensorboard started on http://localhost:6006/'.format())

    tb_process = Process(target=process, args=(tb_trn_path, tb_val_path, tb_tst_path))
    tb_process.start()
    return tb_process


def multilayer_perceptron(layers, activation_fn=None, output_fn=None):
    """

    :param layers:
    :param activation_fn:
    :param output_fn:
    :return:
    """
    # Defaults for the activation functions
    activation_fn = activation_fn or tf.nn.relu

    # The placeholder to activate or deactivate the dropout. Defaults to inactive"
    dropout_rate = tf.placeholder_with_default(0.0, shape=())

    # The inputs placeholder
    x = tf.placeholder(tf.float32, name='input', shape=[None, layers[0]])

    # All the layers 'till the output one
    output = x
    for layer_id, num_neurons in enumerate(layers[1:], start=1):
        fn = output_fn if layer_id == len(layers) - 1 else activation_fn
        output = tf.layers.dense(inputs=output, units=num_neurons, activation=fn)
        # Dropout in all the layers except the final one
        if layer_id < len(layers) - 1:
            output = tf.layers.dropout(output, rate=dropout_rate, training=(dropout_rate > 0.0))

    return x, output, dropout_rate


Dataset = collections.namedtuple('Dataset', ['data', 'data_columns', 'target', 'target_columns'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

LC_TARGET_COLS = ['Lane change left', 'Lane change none', 'Lane change right']


def extract_fuzzy_controller_data(session, fc_data, input_vars):
    all_variables = [n.name for n in tf.get_default_graph().as_graph_def().node]
    patterns = ['^tfz/var/{}/b$'.format(var) for var in input_vars]
    patterns.extend(['^tfz/var/{}/s\d+$'.format(var) for var in input_vars])
    patterns.extend(['^tfz/var/{}/sf\d+$'.format(var) for var in input_vars])
    patterns.extend(['^tfz/var/{}/sl\d+$'.format(var) for var in input_vars])
    patterns.append('^tfz/rules/weights$')
    for pattern in patterns:
        for var in all_variables:
            if re.match(pattern, var) is not None:
                tensor = tf.get_default_graph().get_tensor_by_name(var + ':0')
                values = session.run(tensor)
                if var == 'weights':
                    values = values.flatten()
                    values = ';'.join([str(x) for x in values])
                fc_data[var].append(values)


def load_datasets_for_subject(datasets_path, subject):
    path = os.path.join(datasets_path, 'lc-{}-{}.csv')
    # Load the training set and split it into training and validation sets
    train_df = pd.read_csv(path.format(subject, 'training'), dtype=np.float32, index_col=None)
    rows = random.sample(list(train_df.index), int(len(train_df.index) / 10))
    validation_df = train_df.iloc[rows]
    train_df = train_df.drop(rows)
    # Load the test set
    test_df = pd.read_csv(path.format(subject, 'validation'), dtype=np.float32, index_col=None)

    datasets = {}
    for dataset, df in (('train', train_df), ('validation', validation_df), ('test', test_df)):
        # Separate between data and target columns
        data_df = df.drop(LC_TARGET_COLS, axis=1)
        data_df = data_df.reindex(sorted(data_df.columns), axis=1)
        target_df = df[LC_TARGET_COLS]
        target_df = target_df.reindex(sorted(target_df.columns), axis=1)
        # Create the dataset
        datasets[dataset] = Dataset(
            data=data_df.values,
            data_columns=data_df.columns,
            target=target_df.values.astype(np.int8),
            target_columns=target_df.columns,
        )

    return Datasets(**datasets)


def convolutional(layers, num_inputs, num_outputs, img_start, image_shape):
    def build_convolution_layer(name, inputs, desc, dropout):
        padding_mapping = {'s': 'same', 'v': 'valid'}
        filters, rows, cols, padding = desc[1:].split('-')
        filters, rows, cols, padding = int(filters), int(rows), int(cols), padding_mapping[padding]
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[rows, cols],
            padding=padding,
            activation=tf.nn.relu,
            name=name
        )

    def build_pooling_layer(name, inputs, desc, dropout):
        rows, cols, stride = desc[1:].split('-')
        rows, cols, stride = int(rows), int(cols), int(stride)

        return tf.layers.max_pooling2d(inputs=inputs, pool_size=[rows, cols], strides=stride, name=name)

    def build_dense_layer(name, inputs, desc, dropout):
        units = int(desc[1:])
        output = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu, name=name)
        output = tf.layers.dropout(output, rate=dropout, training=(dropout > 0.0), name=name + '_dropout')
        return output

    image_shape = list(image_shape)
    build_layer_functions = {
        'd': build_dense_layer,
        'c': build_convolution_layer,
        'p': build_pooling_layer,
    }

    patterns_layers = [layer for layer in layers if layer[0] != 'd']
    dense_layers = [layer for layer in layers if layer[0] == 'd']

    x = tf.placeholder(tf.float32, name='input', shape=[None, num_inputs])
    dropout_rate = tf.placeholder_with_default(0.0, shape=())

    # Extract and resize the images from the inputs. Leave the rest as is.
    img_size = np.prod(image_shape)
    x_values_1, x_images, x_values_2 = tf.split(x, [img_start, img_size, num_inputs - img_start - img_size], axis=1)
    layer = tf.reshape(x_images, [-1] + image_shape)

    # Patterns layer
    layers_num = {
        'c': ['convolution', 0],
        'p': ['pooling', 0],
    }
    for layer_description in patterns_layers:
        layer_name, layer_num = layers_num[layer_description[0]]
        layer = build_layer_functions[layer_description[0]]('{}{}'.format(layer_name, layer_num), layer,
                                                            layer_description, dropout_rate)
        layers_num[layer_description[0]] = [layer_name, layer_num + 1]

    # Flatten
    layer = tf.reshape(layer, [-1, int(np.prod(layer.shape[1:]))])
    layer = tf.concat([layer, x_values_1, x_values_2], 1)

    # Dense Layers
    for i, layer_description in enumerate(dense_layers):
        layer = build_layer_functions[layer_description[0]]('dense{}'.format(i), layer, layer_description, dropout_rate)

    output = tf.layers.dense(inputs=layer, units=num_outputs, name='dense{}'.format(i + 1))

    return x, output, dropout_rate


def extract_minibatch_data(dataset, minibatch_size, num_outputs):
    values_per_class = minibatch_size // num_outputs
    data, target = [], []
    for col in range(dataset.target.shape[1]):
        available_idx = np.where(dataset.target[:, col] == 1)[0]
        if len(available_idx) > 0:
            idx = np.random.choice(available_idx, size=values_per_class)
            data.append(dataset.data[idx, :])
            target.append(dataset.target[idx, :])

    data = np.concatenate(data, axis=0)
    target = np.concatenate(target, axis=0)
    return data, target

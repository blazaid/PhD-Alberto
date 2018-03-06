import os

import numpy as np
import pandas as pd


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
    'edgar':{
        'training':{
            'starting_frame':1078,
            'ending_frame':9609,
            'lane_changes': [
                (1119, 1135, 1),(2240, 2260, -1),(2246, 2264, 1),(2499, 2516, -1),
                (2999, 3020, 1),(6095, 6115, -1),(6395, 6415, 1),(6715, 6734, 1),
                (6764, 6784, -1),(6876, 6897, -1),(8460, 8480, 1),(8526, 8537, -1),
            ],
            'skippable_sequences':None,
            'max_speed':[(0, 50), (1089, 40), (2960, 50), (6536, 40),],
            'lanes_distances':[
                (0, 0, np.inf, 0), (940, np.inf, np.inf, 0),
                (1129, 0, np.inf, np.inf), (1290, (40.383074, -3.630246), np.inf, np.inf),
                (1317, 0, np.inf, np.inf), (1425, (40.383933, -3.632542), np.inf, np.inf),
                (1770, 0, np.inf, np.inf), (1840, 0, (40.384765, -3.633968), np.inf),
                (1883, (40.384765, -3.633968), np.inf, np.inf), (1895, 0, (40.384765, -3.633968), np.inf),
                (1943, (40.384765, -3.633968), np.inf, np.inf), (1950, 0, np.inf, np.inf),
                (2080, np.inf, np.inf, 0), (2090, (40.385600, -3.635954), np.inf, np.inf),
                (2099, 0, (40.385600, -3.635954), np.inf), (2140, (40.385600, -3.635954), np.inf, np.inf),
                (2148, 0, (40.387391, -3.640122), np.inf), (2255, (40.387391, -3.640122), np.inf, 0),
                (2454, 0, (40.387391, -3.640122), np.inf), (2509, (40.387391, -3.640122), np.inf, 0),
                (2914, (40.387391, -3.640122), np.inf, (40.388916, -3.639228)), (2933, np.inf, (40.388916, -3.639228), 0),
                (3012, 0, np.inf, (40.388916, -3.639228)), (3250, 0, np.inf, 0),
                (3768, (40.389074, -3.644033), np.inf, (40.387686, -3.645143)),
                (4690, 0, np.inf, 40.387686, -3.645143), (5085, 0, np.inf, np.inf),
                (6100, np.inf, 0, np.inf), (6406, 0, np.inf, np.inf),
                (6448, np.inf, np.inf, 0), (6724, np.inf, np.inf),
                (6776, np.inf, np.inf, 0), (6850, (40.385502, -3.635610), np.inf, np.inf),
                (6888, np.inf, np.inf, 0), (8477, 0, np.inf, np.inf)
            ],
        },
        'validation':{
            'starting_frame':61,
            'ending_frame':2687,
            'lane_changes': [
                (370, 384, 1),(1506, 1523, 1),(1588, 1600, -1),(2510, 2522, 1),
                (2580, 2591, 1),(2623, 2649, -1),(3003, 3020, 1),(3070, 3086, -1),
            ],
            'skippable_sequences':None,
            'max_speed':[(0, 40), (370, 50)],
        }
    },
    'jj':{
        'training':{
            'starting_frame':150,
            'ending_frame':10580,
            'lane_changes': [
                (1560, 1575, 1),(2510, 2530, -1),(3115, 3130, 1),(3140, 3151, -1),
                (3173, 3190, 1),(5452, 5475, 1),(6926, 6947, 1),(7360, 7385, 1),
                (7565, 7584, -1),(7726, 7750, 1),(8699, 8714, 1),(8735, 8755, -1),
                (10010, 10033, -1),(10082, 10093, 1),(10112, 10135, -1),
            ],
            'skippable_sequences': [(5600, 6878),],
            'max_speed':[(0, 50), (1065, 40), (3000, 50), (8562, 40),],
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
        },
        'validation':{
            'starting_frame':512,
            'ending_frame':4142,
            'lane_changes': [
                (3115, 3142, 1),(3540, 3570, -1),
            ],
            'skippable_sequences':None,
            'max_speed':[(0, 40), (780, 50)],
        }
    },
    'miguel':{
        'training':{
            'starting_frame':100,
            'ending_frame':10232,
            'lane_changes':[
                (1532, 1550, 1),(3510, 3530, 1),(3560, 3572, -1),(3580, 3602, 1),
                (3613, 3630, -1),(5239, 5249, 1),(6050, 6068, 1),(6190, 6211, -1),
                (6279, 6302, 1),(6510, 6529, 1),(6550, 6580, -1),(8729, 8755, 1),
                (9499, 9520, -1),(9830, 9851, 1),(10129, 10149, -1),
            ],
            'skippable_sequences':None,
            'max_speed':[(0, 50), (1400, 40), (3000, 50), (7845, 40),],
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
        },
        'validation':{
            'starting_frame':44,
            'ending_frame':3651,
            'lane_changes': [
                (880, 898, 1),(2820, 2839, 1),(2844, 2867, -1),(3462, 3492, 1),
                (4140, 4158, 1)
            ],
            'skippable_sequences':None,
            'max_speed':[(0, 40), (600, 50)],
        }
    },
}
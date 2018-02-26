import os

import pandas as pd


def load_master_df(path, dataset, kind):
    """
    :param kind: One of "detours", "sections", "tls" or "yields".
    """
    kinds = ["detours", "sections", "tls", "yields"]
    if kind not in kinds:
        raise ValueError('parameter kind should be one of {}'.format(kinds))
    
    filename = '{}_{}.csv'.format(dataset, kind)
    return pd.read_csv(os.path.join(path, filename))

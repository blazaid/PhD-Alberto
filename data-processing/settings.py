import os


class Paths(object):
    """ Utility to manage paths for the data workflow. """
    
    def __init__(self, base_dir, *subdirs):
        if not os.path.isdir(base_dir):
            raise ValueError(base_dir + ' directory doesn\'t exist')
        self.base_dir = base_dir
        for subdir in subdirs:
            full_path = os.path.join(base_dir, subdir)
            setattr(self, subdir, full_path)
            if not os.path.isdir(full_path):
                os.makedirs(full_path)
    
    @property
    def bags_data_dir(self):
        bags_data_dir = os.path.join(self.base_data_dir, 'bags')
        if not os.path.exists(bags_data_dir):
            os.makedirs(bags_data_dir)
        return bags_data_dir
    
    @property
    def raw_csv_data_dir(self):
        raw_csv_data_dir = os.path.join(self.base_data_dir, 'raw_csv')
        if not os.path.exists(raw_csv_data_dir):
            os.makedirs(raw_csv_data_dir)
        return raw_csv_data_dir
    
    def bag_dataset_path(self, subject, dataset):
        filename = subject + '-' + dataset + '.bag'
        return os.path.join(self.bags_data_dir, filename)

    def raw_csv_dataset_path(self, subject, dataset, topic):
        filename = subject + '-' + dataset + topic.replace('/', '-') + '.csv'
        return os.path.join(self.raw_csv_data_dir, filename)

    def raw_csv_dir(self, subject, dataset, dirname):
        directory = os.path.join(
            self.raw_csv_data_dir,
            subject + '-' + dataset + '-' + dirname
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

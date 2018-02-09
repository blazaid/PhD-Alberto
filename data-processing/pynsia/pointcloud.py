import numpy as np

from sensor_msgs.msg import PointField


class PointCloud(object):
    PF_MAPPINGS = {
        PointField.INT8: (np.int8, 1),
        PointField.UINT8: (np.uint8, 1),
        PointField.INT16: (np.int16, 2),
        PointField.UINT16: (np.uint16, 2),
        PointField.INT32: (np.int32, 4),
        PointField.UINT32: (np.uint32, 4),
        PointField.FLOAT32: (np.float32, 4),
        PointField.FLOAT64: (np.float64, 5),
    }

    EXTRA_PAD = 'EXTRA_PADDING'

    def __init__(self):
        self.points = None

    @classmethod
    def from_pc2(cls, pc2):
        """ Constructs a new instance from a PointCloud2 ROS message.

        :param pc2: The PointCloud2 message from ROS.
        :return: An instance of this class.
        """
        offset = 0
        struct = []
        for f in pc2.fields:
            # Mark each extra padding offset as so, so we can remove it later
            while offset < f.offset:
                struct.append((PointCloud.EXTRA_PAD + '_' + str(offset), np.uint8))
                offset += 1
            # We're in a defined field, so create it and advance the offset
            datatype, size = PointCloud.PF_MAPPINGS[f.datatype]
            struct.append((f.name, datatype))
            offset += size

        # If there is still padding space, fill it with bytes
        while offset < pc2.point_step:
            struct.append((PointCloud.EXTRA_PAD + '_' + str(offset), np.uint8))
            offset += 1

        # Parse the pcl data with the structure
        pcl = np.fromstring(pc2.data, struct)

        # Remove the extra padding columns
        valid_columns = [f for f, _ in struct if not f.startswith(PointCloud.EXTRA_PAD)]
        pcl = pcl[valid_columns]

        # Convert into an array of shape (-1, 3) and return it
        instance = cls()
        instance.points = np.array([np.array(list(t)) for t in pcl])
        instance.points = instance.points[:, :3]

        return instance

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.points = np.loadtxt(path, delimiter=',')[:, :3]
        return instance

    def save(self, path):
        if self.points is not None and self.points.size > 0:
            np.savetxt(path, self.points, delimiter=',')
        else:
            raise ValueError('Empty point cloud')

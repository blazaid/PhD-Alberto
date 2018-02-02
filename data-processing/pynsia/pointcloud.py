# Correspondence between pointfields and its numpy type plus size
import numpy as np
from sensor_msgs.msg import PointField

pf_mappings = {
    PointField.INT8: (np.int8, 1),
    PointField.UINT8: (np.uint8, 1),
    PointField.INT16: (np.int16, 2),
    PointField.UINT16: (np.uint16, 2),
    PointField.INT32: (np.int32, 4),
    PointField.UINT32: (np.uint32, 4),
    PointField.FLOAT32: (np.float32, 4),
    PointField.FLOAT64: (np.float64, 5),
}

EXTRA_PADDING = 'EXTRA_PADDING'


def pc2_to_pc(msg):
    """ Transforms a PointCloud2 message to a 2D ndarray.

    The array will have a MxN size, where M is the number of points and N the
    fields for each point. For example, a PointCloud2 with only
    """
    # Extract the structure for each point_step
    offset = 0
    structure = []
    for f in msg.fields:
        # Mark each extra padding offset as so, so we can remove it later
        while offset < f.offset:
            structure.append((EXTRA_PADDING + '_' + str(offset), np.uint8))
            offset += 1
        # We're in a defined field, so create it and advance the offset
        datatype, size = pf_mappings[f.datatype]
        structure.append((f.name, datatype))
        offset += size

    # If there is still padding space, fill it with bytes
    while offset < msg.point_step:
        structure.append((EXTRA_PADDING + '_' + str(offset), np.uint8))
        offset += 1

    # Parse the pcl data with the structure
    pcl = np.fromstring(msg.data, structure)

    # Remove the extra padding columns
    valid_columns = [f for f, _ in structure if not f.startswith(EXTRA_PADDING)]
    pcl = pcl[valid_columns]

    # Convert into an array of shape (-1, 5)
    return np.array([np.array(list(t)) for t in pcl])


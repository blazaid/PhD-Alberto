import numpy as np

from sensor_msgs.msg import PointField


def to_spherical(ps):
    """ Transforms a set of points into its spherical representations.

    The points should be expressed as a Mx3 matrix, where M is the number of
    points and 3 the cartesian coordinates x, y and z.

    :param ps: The matrix of points.
    :return: The matrix of the same points but with their spherical coords where
        each row will be in the form (distance, theta, phi).
    """
    x2, y2, z2 = ps[:, 0] ** 2, ps[:, 1] ** 2, ps[:, 2] ** 2
    xy = np.sqrt(x2 + y2)

    result = np.zeros(ps.shape)
    # Distance to point
    result[:, 0] = np.sqrt(x2 + y2 + z2)
    # Azimuth
    pos_x = ps[:, 0] >= 0
    neg_x = ps[:, 0] < 0
    result[pos_x, 1] = np.arccos(ps[pos_x, 0] / xy[pos_x])
    result[neg_x, 1] = 2 * np.math.pi - np.arccos(ps[neg_x, 0] / xy[neg_x])
    # Elevation angle from XY-plane up
    result[:, 2] = np.arctan2(ps[:, 2], xy)
    return result


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
        """ Initializer for instances of this class.

        It shouldn't be called directly, but from class methods.
        """
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

    def to_deepmap(
            self,
            h_range=(0, 360), v_range=(-15, 15),
            h_res=1, v_res=1,
            max_dist=None,
            normalize=True
    ):
        """ Transforms this instance into a deepness_map. """

        # Establish the default values
        h_range = list(map(float, h_range))
        v_range = list(map(float, v_range))
        h_res, v_res = float(h_res), float(v_res)
        max_dist = float(max_dist) or np.inf

        # Get the steps to travel horizontally and vertically
        h_r = int(min(h_range) / h_res), int(max(h_range) / h_res)
        v_r = int(min(v_range) / v_res), int(max(v_range) / v_res)

        # Create the matrix for the deep map
        dm = np.full((abs(v_r[0] - v_r[1]), abs(h_r[0] - h_r[1])), max_dist)

        # Transform the points to their spherical coordinates
        cartesian_coords = self.points[:, :3]
        spherical_coords = to_spherical(cartesian_coords)
        # Convert radians to degrees
        spherical_coords[:, 1:] = np.degrees(spherical_coords[:, 1:])
        # Convert to the expected resolution
        spherical_coords[:, 1] = (spherical_coords[:, 1] / h_res)
        spherical_coords[:, 2] = (spherical_coords[:, 2] / v_res)
        # Map all the values to each interval
        spherical_coords[:, 1] -= h_r[0]
        spherical_coords[:, 2] -= v_r[0]

        for d, t, r in spherical_coords[spherical_coords[:, 0] <= max_dist, :]:
            # Translate the value
            t = int(max(0, min(dm.shape[1] - 1, t)))
            r = int(max(0, min(dm.shape[0] - 1, r)))
            dm[r][t] = min(dm[r][t], d)
        # Si normalize esta activo, normalizamos todos los valores al intervalo [0, 1] ([min, max] distancia).
        if normalize:
            dm = dm / max_dist
            dm = 1 - dm
        # Ahora recolocamos el array para que sea algo mas realista la imagen
        dm = np.flipud(dm)
        fr = dm[:, :len(dm[0]) // 4]
        br = dm[:, len(dm[0]) // 4:len(dm[0]) // 2]
        bl = dm[:, len(dm[0]) // 2:3*len(dm[0]) // 4]
        fl = dm[:, 3*len(dm[0]) // 4:]
        return np.concatenate((fl, fr, br, bl), axis=1)
import os

import numpy as np

POINTCLOUDS_PATH = 'lidar'


def point_cloud_key(pcl_name):
    return int(pcl_name[4:-4])


DEEPMAP_MODES = 'deep', 'coords'


def to_spherical_old(x, y, z):
    x2, y2, z2 = pow(x, 2), pow(y, 2), pow(z, 2)

    d = np.math.sqrt(x2 + y2 + z2)
    if x >= 0:
        theta = np.math.acos(y / np.math.sqrt(x2 + y2))
    else:
        theta = 2 * np.math.pi - np.math.acos(y / np.math.sqrt(y2 + z2))
    phi = np.math.asin(z / d)

    return d, theta, phi


def to_spherical(ps):
    """ Transforms a set of points into its spherical representations.

    The points should be expressed as a Mx3 matrix, where M is the number of
    points and 3 the cartesian coordinates x, y and z.

    :param ps: The matrix of points.
    :return: The matrix of the same points but with their spherical coords where
        each row will be in the form (distance, theta, phi).
    """
    x2, y2, z2 = ps[:, 0] ** 2, ps[:, 1] ** 2, ps[:, 2] ** 2

    result = np.zeros(ps.shape)
    # Distance to point
    result[:, 0] = np.sqrt(x2 + y2 + z2)
    # Theta
    result[ps[:, 0] >= 0, 1] = np.math.acos(ps[ps[:, 0] >= 0, 1] / np.sqrt(x2 + y2))
    #result[ps[:, 0] < 0, 1] = 2 * np.math.pi - np.math.acos(ps[ps[:, 0] < 0, 1] / np.sqrt(y2 + z2))
    result[not ps[:, 0] >= 0, 1] = np.math.acos(not ps[ps[:, 0] >= 0, 1] / np.sqrt(x2 + y2))
    #result[:, 1] = np.arctan2(np.sqrt(xy), points[:, 2])
    #result[:, 1] = np.arctan2(points[:, 1], points[:, 0])
    # Phi: elevation angle from XY-plane up
    result[:, 2] = np.arctan2(ps[:, 2], np.sqrt(x2 + y2))
    return result


class PointCloud(object):
    """ Class that represents a point cloud. """

    def __init__(self, pc):
        """ Initializer for instances of this class.

        It shouldn't be called directly, but from class methods.
        """
        self.pc = pc

    @classmethod
    def from_file(cls, path):
        """ Creates a new PointCloud given the file with PointCloud information.

        :param path: The location of the file.
        :return: A new PointCloud instance.
        """
        pc = np.loadtxt(path, delimiter=',')
        return cls(pc=pc)

    def save(self, path):
        """ Saves the point cloud into a file.

        :param path: The location of the file.
        """
        np.savetxt(path, self.pc, delimiter=',')

    def to_deepmap(
            self,
            h_range=(0, 360), v_range=(-15, 15),
            h_res=1, v_res=1,
            max_dist=None,
            normalize=True
    ):
        """ Transforms this instance into a deepness_map. """
        # Establish the default values
        max_dist = max_dist or np.inf

        # Get the steps to travel horizontally and vertically
        h_r = int(min(h_range) / h_res), int(max(h_range) / h_res)
        v_r = int(min(v_range) / v_res), int(max(v_range) / v_res)

        # Create the matrix for the deep map
        dm = np.full((abs(v_r[0] - v_r[1]), abs(h_r[0] - h_r[1])), max_dist)

        # Transform the points to their spherical coordinates
        cartesian_coords = self.pc[0, :3]
        print(to_spherical_old(cartesian_coords[0], cartesian_coords[1], cartesian_coords[2]))
        print(to_spherical(np.array([cartesian_coords])))
        return
        for x, y, z in point_cloud:
            d, t, r = to_spherical(x, y, z)
            if d < max_dist:
                # traducimos a fila y columna.
                t, r = int(math.degrees(t) / h_res), int(
                    math.degrees(r) / v_res)
                t = max(0, min(dm.shape[1], t))
                r = max(0, min(dm.shape[0] - 1, r - v_r[0]))
                dm[r][t] = max_dist - d
        # Si normalize esta activo, normalizamos todos los valores al intervalo [0, 1] ([min, max] distancia).
        if normalize:
            dm = dm / max_dist
        # Ahora recolocamos el array para que sea algo mas realista la imagen
        dm = np.flipud(dm)
        deep_map_l = dm[:, :len(dm[0]) // 2]
        deep_map_r = dm[:, len(dm[0]) // 2:]
        dm = np.concatenate((deep_map_r, deep_map_l), axis=1)
        return dm


if __name__ == '__main__':
    for filename in sorted(os.listdir(POINTCLOUDS_PATH), key=point_cloud_key):
        pointcloud_path = os.path.join(POINTCLOUDS_PATH, filename)
        pointcloud = PointCloud.from_file(pointcloud_path)
        deepmap = pointcloud.to_deepmap()
        break

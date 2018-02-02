import abc
import os

import csv
import numpy as np

class MsgParser(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def header(self, message):
        raise NotImplementedError()

    @abc.abstractmethod
    def row(self, message):
        raise NotImplementedError()


class CanParser(MsgParser):
    def __init__(self, allowed_messages=None):
        if allowed_messages is not None:
            self.messages = set(['t' + m for m in allowed_messages])
        else:
            self.messages = None

    def header(self, message):
        return 'secs', 'nsecs', 'frame'

    def row(self, message):
        allowed_message = any(m in message.data for m in self.messages)
        if self.messages is not None and not allowed_message:
            return None
        else:
            return (
                message.header.stamp.secs,
                message.header.stamp.nsecs,
                message.data,
            )


class GpsPositionParser(MsgParser):
    def header(self, message):
        return 'secs', 'nsecs', 'latitude', 'longitude', 'altitude'

    def row(self, message):
        return (
            message.header.stamp.secs,
            message.header.stamp.nsecs,
            message.latitude,
            message.longitude,
            message.altitude,
        )


class GpsSpeedParser(MsgParser):
    def header(self, message):
        return 'secs', 'nsecs', 'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z'

    def row(self, message):
        return (
            message.header.stamp.secs,
            message.header.stamp.nsecs,
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.linear.z,
            message.twist.angular.x,
            message.twist.angular.y,
            message.twist.angular.z,
        )


class KinectImageParser(MsgParser):
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.index = 0
        self.bridge = CvBridge()

    def header(self, message):
        return 'secs', 'nsecs', 'path'

    def row(self, message):
        # Extract the pcl from the PointCloud2 message
        image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        # Rotate it 180 degs. because is upside down
        M = cv2.getRotationMatrix2D((message.width / 2, message.height / 2),
                                    180, 1)
        image = cv2.warpAffine(image, M, (message.width, message.height))
        # Save onto the current image path
        path = os.path.join(self.path, 'img_' + str(self.index) + '.png')
        cv2.imwrite(path, image)
        self.index += 1
        # Return the values
        return (
            message.header.stamp.secs,
            message.header.stamp.nsecs,
            path,
        )


class LidarParser(MsgParser):
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.index = 0

    def header(self, message):
        return 'secs', 'nsecs', 'path'

    def row(self, message):
        # Extract the pcl from the PointCloud2 message
        pcl = pc2_to_pc(message)
        # Save onto the current frame path
        path = os.path.join(self.path, 'pcl_' + str(self.index) + '.csv')
        np.savetxt(path, pcl, delimiter=',')
        self.index += 1
        # Return the values
        return (
            message.header.stamp.secs,
            message.header.stamp.nsecs,
            path,
        )

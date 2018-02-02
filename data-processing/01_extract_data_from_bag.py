import abc
import csv
import os

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

from pynsia.pointcloud import pc2_to_pc
from pynsia.ros.bag import topics

FILE_PREFIX = 'subject1-test'
BAG_FILE = '{}.bag'.format(FILE_PREFIX)

TOPICS_TO_EXTRACT = [
    '/capture/can',
    '/capture/gps/position',
    '/capture/gps/speed',
    '/capture/kinect/image',
    '/capture/lidar'
]


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


def dump_topic_to_csv(topic_name, parser):
    # Process only if topic between extractable topics
    if topic_name not in TOPICS_TO_EXTRACT:
        print('Skipping topic {}'.format(topic_name))
        return
    else:
        print('Dumping ' + topic_name)
    # Create the csv file for this topic
    csv_file = FILE_PREFIX + topic_name.replace('/', '-') + '.csv'
    with open(csv_file, 'w+') as f:
        # Create the object that writes csv
        writer = csv.writer(f)
        # Start writing the topics as rows in the csv
        for i, (_, msg, ts) in enumerate(bag.read_messages(topic_name)):
            # For each topic name, write headers
            if i == 0:
                writer.writerow(parser.header(msg))
            # Write the row if there is any
            row = parser.row(msg)
            if row is not None:
                writer.writerow(row)


if __name__ == '__main__':
    topics_parsers = {
        '/capture/can': CanParser(allowed_messages=[
            '696', '208', '412', '346', '374',
            '236', '2F2', '418', '424', '231'
        ]),
        '/capture/gps/position': GpsPositionParser(),
        '/capture/gps/speed': GpsSpeedParser(),
        '/capture/kinect/image': KinectImageParser('kinect_image'),
        '/capture/lidar': LidarParser('lidar'),
    }

    print('Loading bag file')
    bag = rosbag.Bag(BAG_FILE)
    print('Launching dump processes')
    for topic in topics(bag):
        dump_topic_to_csv(topic, topics_parsers[topic])

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

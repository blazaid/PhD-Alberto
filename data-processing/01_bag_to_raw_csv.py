#!/usr/bin/python2
#

""" bag_to_raw_csv.py

Extracts all the data captured by ros with can+kineck+gps+lidar and extracts it to a csv per sensor.

Usage: python2 01_bag_to_raw_csv.py subject validation data/bags/subject1.bag data/raw_csvs
"""

import argparse
import csv
import os

import rosbag

from bag_parsers import CanParser, GpsPositionParser, GpsSpeedParser, KinectImageParser, LidarParser
from pynsia.ros.bag import topics

CAN_MESSAGES = '412',  # '208', '236', '346', '374', '412',

KINECT_IMAGES_DIR = 'snapshots'
LIDAR_IMAGES_DIR = 'pointclouds'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transforms a the bag files to a set of csvs, one for each sensor.')
    parser.add_argument('subject', help='The subject of the experiment (the bag data belongs to him).')
    parser.add_argument('dataset', choices=('training', 'validation'), help='One of the dataset types')
    parser.add_argument('bag', help='The bag file to extract.')
    parser.add_argument('output', help='The directory where the data is extracted.')
    args = parser.parse_args()

    # Configure the topic parsers given the arguments
    output_dir = os.path.join(args.output, args.subject, args.dataset)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    topics_parsers = {
        '/capture/can': ('canbus', CanParser(allowed_messages=CAN_MESSAGES)),
        '/capture/gps/position': ('gps_positions', GpsPositionParser()),
        '/capture/gps/speed': ('gps_speeds', GpsSpeedParser()),
        '/capture/kinect/image': ('snapshots', KinectImageParser(os.path.join(output_dir, KINECT_IMAGES_DIR))),
        '/capture/lidar': ('pointclouds', LidarParser(os.path.join(output_dir, LIDAR_IMAGES_DIR))),
    }

    # Load the bag to work with it
    print('Loading bag: ' + args.bag)
    bag = rosbag.Bag(args.bag)

    # Identify with which topics we have to deal with
    extractable_topics = [topic for topic in topics(bag) if topic in topics_parsers.keys()]
    skippable_topics = [topic for topic in topics(bag) if topic not in topics_parsers.keys()]
    print('Messages to extract: ' + ', '.join(extractable_topics))
    print('Messages to avoid: ' + ', '.join(skippable_topics))

    # Now traverse over all the extractable topics, saving each one to its own file
    for topic in sorted(extractable_topics):
        print('Extracting ' + topic + ' ...')

        # Get the parser for the topic
        csv_name, parser = topics_parsers[topic]

        # Get the csv file to dump
        csv_path = os.path.join(output_dir, csv_name + '.csv')

        with open(csv_path, 'w+') as f:
            # Create the object that writes csv
            writer = csv.writer(f)

            # Start writing the topics as rows in the csv
            print('Writing raw csv:\t' + csv_path + ' ...')

            for i, (_, msg, ts) in enumerate(bag.read_messages(topic)):
                # For each topic name, write headers
                if i == 0:
                    writer.writerow(parser.header(msg))

                # Write the row if there is any
                row = parser.row(msg)
                if row is not None:
                    writer.writerow(row)

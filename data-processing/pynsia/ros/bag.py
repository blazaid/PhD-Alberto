import rosbag

num_msgs = 100


def crop(bag, path, n):
    """ Creates a new bag file with the first `n` topics.

    The process will create the file referenced by path so, if the file already
    contains information, it'll be erased.

    :param bag: The bag to be cropped.
    :param path: A string with the path where the file with the n topics will be
        written.
    :param n: The (maximum) number of topics to write. If there are less topics
        than `n`, only the existent topics will be written.
    """
    with rosbag.Bag(path, 'w') as f:
        for topic, msg, t in bag.read_messages():
            if n < 1:
                break
            n -= 1
            f.write(topic, msg, t)


def topics(bag):
    """ Returns the topics contained in the bag.

    :param bag: The bag.
    :return: A generator for the topics. To get'em all, it's necessary to
        iterate over them.
    """
    # Get the existent topics from the bag
    topics_found = set()
    for topic, msg, t in bag.read_messages():
        if topic not in topics_found:
            topics_found.add(topic)
            yield topic

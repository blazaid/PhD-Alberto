import rosbag

from pynsia.ros.bag import crop, topics

#bag = rosbag.Bag('/media/blazaid/Saca/Phd/data/bags/miguel-validation.bag')

#crop(bag, '/home/blazaid/1000msgs.bag', 1000)

bag = rosbag.Bag('/home/blazaid/1000msgs.bag')
print(list(topics(bag)))

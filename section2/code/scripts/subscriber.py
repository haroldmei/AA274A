#!/usr/bin/env python

import rospy
from aa274_s2.msg import MyMessage

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def subscriber():
    rospy.init_node('my_subscriber', anonymous=True)
    rospy.Subscriber("my_topic", MyMessage, callback)
    rospy.spin()

if __name__ == '__main__':
    subscriber()

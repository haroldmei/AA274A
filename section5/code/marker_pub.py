import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D


# use global variables to update the navigation goal
x_g = 0.0
y_g = 0.0
theta_g = 0.0
def cmd_nav_callback(data):
    """
    loads in goal if different from current goal, and replans
    """
    rospy.loginfo(rospy.get_caller_id() + "I heard %f, %f, %f", data.x,data.y,data.theta)
    global x_g
    global y_g
    global theta_g

    x_g = data.x
    y_g = data.y
    theta_g = data.theta

def publisher():
    vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
    rospy.init_node('marker_node', anonymous=True)

    rospy.loginfo("Subscriber created")
    rospy.Subscriber('/cmd_nav', Pose2D, cmd_nav_callback)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = 0

        marker.type = 1 # sphere

        marker.pose.position.x = x_g
        marker.pose.position.y = y_g
        marker.pose.position.z = 1

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        vis_pub.publish(marker)
        print('Published marker!', x_g, y_g)
        
        rate.sleep()


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass

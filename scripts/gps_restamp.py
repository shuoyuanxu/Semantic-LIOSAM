#!/usr/bin/env python3
"""
GPS timestamp corrector.

The antobot_gps receiver clock is frozen (all messages share the same
header.stamp). We restamp each fix to the current ROS clock time so that
navsat_transform_node and LIO-SAM's GPS factor see advancing, time-aligned
fixes. With use_sim_time=true, rospy.Time.now() returns the current bag
playback time, keeping GPS in sync with LiDAR/IMU.
"""
import rospy
from sensor_msgs.msg import NavSatFix

_pub = None


def callback(msg):
    out = NavSatFix()
    out.header = msg.header
    out.header.stamp = rospy.Time.now()   # bag clock time, not frozen GPS clock
    out.status = msg.status
    out.latitude = msg.latitude
    out.longitude = msg.longitude
    out.altitude = msg.altitude
    out.position_covariance = msg.position_covariance
    out.position_covariance_type = msg.position_covariance_type
    _pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("gps_restamp")
    _pub = rospy.Publisher("/antobot_gps_restamped", NavSatFix, queue_size=10)
    rospy.Subscriber("/antobot_gps", NavSatFix, callback)
    rospy.loginfo("[gps_restamp] Restamping /antobot_gps -> /antobot_gps_restamped (frozen GPS clock workaround)")
    rospy.spin()

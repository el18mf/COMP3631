#Changes the / to mean True division throughout the code, for use in working out the centroid of the detected Shapes
from __future__ import division
#import OpenCV
import cv2
#facilitates mathematical operations on arrays and for the log conversion
import math
import numpy as np
#Allows interfacing with ROS topics, serices and parameters. Essentially the ROS library
import rospy 
#Allows system operations to be performed via code
import sys
#allows control of the robots movement through a publisher
from geometry_msgs.msg import Twist, Vector3
#allows the import and use of the bumpers data, incase 
from kobuki_msgs.msg import BumperEvent
#Allows the import and manipulation of the Image data from the robot
from sensor_msgs.msg import Image
#Used in error reporting
from cv_bridge import CvBridge, CvBridgeError
#Use not decided as of yet
from os import system, name

#Reading the Image file for comparison, and importing it as a greyscale version
Circle_template = cv2.imread("/home/csunix/el18mf/catkin_ws/src/group_project/scripts/Mat/Circle.jpg", cv2.IMREAD_GRAYSCALE)
#binarisation/creation of templates threshold image
_, template_thresh = cv2.threshold(Circle_template, 127, 255, 0)
#finding the moments of the templates threshold image
Circle_moments = cv2.moments(Circle_template)
#finding of the Hu moments of the templates threshold image
Circle_Hu_Moments = cv2.HuMoments(Circle_moments)

print("The Hu Moments are: \n")
for i in range(0,7):
    print(str(i+1) + ": ", Circle_Hu_Moments[i])

c = max()
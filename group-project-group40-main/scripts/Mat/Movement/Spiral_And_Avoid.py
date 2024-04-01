from __future__ import division
import cv2
import numpy as np
import rospy
import sys

#Autonomous movement to x,y coords
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *

#import tf to use for the obtaining of the current x,y co-ordinates of the turtlebot3
import tf

#for use in the reading of the input co-ordinates
import yaml

import cv2

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
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3

#allows the import and use of the bumpers data, incase 
from kobuki_msgs.msg import BumperEvent

#Allows the import and manipulation of the Image data from the robot
from sensor_msgs.msg import Image

#Used in error reporting
from cv_bridge import CvBridge, CvBridgeError

import os

#To enable use of the timer
import time

#Timer to make sure we print the best guess before 5 minutes has elapsed
start = time.time()
seconds = 300

# ** Don't change this path **
PROJECT_DIR = os.path.expanduser('~/catkin_ws/src/group_project')


def read_input_points():
    path = f'{PROJECT_DIR}/world/input_points.yaml'
    with open(path) as opened_file:
        return yaml.safe_load(opened_file)


def write_character_id(character_id):
    path = f'{PROJECT_DIR}/output/cluedo_character.txt'
    with open(path, 'w') as opened_file:
        opened_file.write(character_id)


def save_character_screenshot(cv_image):
    path = f'{PROJECT_DIR}/output/cluedo_character.png'
    cv2.imwrite(path, cv_image)

print("\n",read_input_points)
coords = read_input_points()
print("\n",coords)
input_name = list(coords.keys())
input_coords = list(coords.values())
print("\n",input_name)
print("\n",input_coords)
for i in range(0,3):
    print("For the location:", input_name[i], "The Co-ordinates are -  x =", input_coords[i][0], ", y =", input_coords[i][1])


class GoToPose():
    def __init__(self):
        
        self.goal_sent = False
        
        self.listener = tf.TransformListener()
        
        self.room_found = False
        self.room1_correct = False
        self.room2_correct = False
        
        self.green_circle_found = False
        
        # What to do if shut down (e.g. Ctrl-C or failure)
        rospy.on_shutdown(self.shutdown)
	
        # Tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('Wait for the action server to come up')

        self.move_base.wait_for_server()

    def gotoNearestEntrance(self, pos, quat):
        # Send a goal
        self.goal_sent = True
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000), Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))

	    # Start moving
        self.move_base.send_goal(goal)

        # Allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(60)) 
        
        # if dest_xy == turtle_xy:
            
        

        state = self.move_base.get_state()
        result = False

        if success and state == GoalStatus.SUCCEEDED:
            result = True
        else:
            self.move_base.cancel_goal()

        self.goal_sent = False

        return result

    def gotoCorrectCentre(self, pos, quat):
        # Send a goal
        self.goal_sent = True
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000), Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))

	    # Start moving
        self.move_base.send_goal(goal)

        # Allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(60)) 
        
        # if dest_xy == turtle_xy:
            
        

        state = self.move_base.get_state()
        result = False

        if success and state == GoalStatus.SUCCEEDED:
            result = True
        else:
            self.move_base.cancel_goal()

        self.goal_sent = False

        return result

    def shutdown(self):
        if self.goal_sent:
            self.move_base.cancel_goal()
        rospy.loginfo('Stop')
        rospy.sleep(1)

    
if __name__ == '__main__':
    green_room_found = False
    red_room_found = False
    correct_room_entered = False
    
    try:
        nearest_entrance_circle = ''
        
        rospy.init_node('nav_test', anonymous=True)
        navigator = GoToPose()
        
        listener = tf.TransformListener()
        #find the closest point between the different locations
        # Wait for transformation to become available
        listener.waitForTransform("/map", "/base_footprint", rospy.Time(), rospy.Duration(4.0))

        # Get transformation between map and robot frames
        (trans, rot) = listener.lookupTransform("/map", "/base_footprint", rospy.Time(0))
        
        #Extract the x,y coords
        turtle_x = trans[0]
        turtle_y = trans[1]
        
        turtle_xy = [turtle_x, turtle_y]
        
        #
        if correct_room_entered != True:
            #Calc the distance between the robots current position and the coordinates of room 1's entrance
            room1_diff_x = abs(turtle_x - input_coords[0][0])
            room1_diff_y = abs(turtle_y - input_coords[0][1])
            room1_distance = math.sqrt(room1_diff_x**2 + room1_diff_y**2)
            #Calc the distance between the robots current position and the coordinates of room 2's entrance
            room2_diff_x = abs(turtle_x - input_coords[1][0])
            room2_diff_y = abs(turtle_y - input_coords[1][1])
            room2_distance = math.sqrt(room2_diff_x**2 + room2_diff_y**2)
        
            if room1_distance < room2_distance:
                current_destination = 'room1'
                x = input_coords[0][0]
                y = input_coords[0][1]
                dest_xy = [x,y]
            else:
                current_destination = 'room2'
                x = input_coords[1][0]
                y = input_coords[1][1]
                dest_xy = [x,y]
            # Customize the following values so they are appropriate for your location
            # SPECIFY X COORDINATE HERE
            # SPECIFY Y COORDINATE HERE
            # y = 
            # SPECIFY THETA (ROTATION) HERE
            theta = 0
            position = {'x': x, 'y' : y}
            quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}

            rospy.loginfo('Go to (%s, %s) pose', position['x'], position['y'])
            success = navigator.gotoNearestEntrance(position, quaternion)

            if success:
                rospy.loginfo('Hooray, reached the desired pose')
            else:
                rospy.loginfo('The base failed to reach the desired pose')

        # Sleep to give the last log messages time to be sent
        rospy.sleep(1)
    except rospy.ROSInterruptException:
        rospy.loginfo('Ctrl-C caught. Quitting')



# class spiral_avoid():
#     def __init__(self):
#         #initialise a publisher to publish messages to teh robot base
#         self.pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=0)
        
#         #initialise Flags
        
#         #initialise some standard movement messages
#         self.current = 0.1308
#         self.forw = "Moving Forward!"
#         self.backw = "Moving Backward"
#         self.stop = "Stopping"
#         self.spiral = "Searching in Spiral Pattern"
        
#         # Initialise a CvBridge() and set up a sub to the image topic
#         self.bridge = CvBridge()
#         self.sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)
        
#     def callback(self, data):
#         #convert the received image into an OpenCV usable image
#         #Remember to always use a exception handler
#         try:
#             cv_i = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            
            
            
            
# def main(args):
#     rospy.init_node("Detector", anonymous = True)
#     cI = spiral_avoid
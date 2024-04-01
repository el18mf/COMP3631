# Mathew's Contribution towards Group 40's Cluedo Detective Robot

# Object Recognition - Main Aims: 
#   Detecting the Door Identifying Circles
#   Identifying the confirmed circle's colour
#   Assigning the corresponding door with it's circle (and whether to enter or avoid said door)
#   Identifying the cluedo image shape

# Possible additional functions:
#   Add circle location and colour onto map
#   logging the circles location and colour on map/adding the location to the map

#automated running commands
# 1 - roscd group_project
# 2 - cd world_manager
# 3 - ./main.sh
# 4 - type: delete catkin_ws/src/world
# Functions to run:
#   1 - export TURTLEBOT_GAZEBO_WORLD_FILE=$HOME/catkin_ws/src/group_project/world/project_circle.world - tells the simulator which file to use, in this case it uses the group projects world file
#   2 - roslaunch turtlebot_gazebo turtlebot_world.launch - Launches the simulator
#   If localisation and autonomous movement has not been implemented, you can manually operate robot through the use of the Tele-Operator
#   3 - roslaunch turtlebot_teleop keyboard_teleop.launch
###note: you can paste into the terminator windows by using shift+ctrl+p
# to launch the map and input the points to test program in World 1 while not using the main.sh option
# 1 - export TURTLEBOT_GAZEBO_WORLD_FILE=$HOME/catkin_ws/src/group_project/world_manager/worlds/world1
# 2 - roslaunch turtlebot_gazebo turtlebot_world.launch - Launches the simulator
# 3 - cd $HOME/catkin_ws/src/group_project/launch/
# 4 - roslaunch simulated_localisation.launch map_file:=$HOME/catkin_ws/src/group_project/world/map/project_map.yaml 2> >(grep -v TF_REPEATED_DATA buffer_core)
# 5 - roslaunch turtlebot_rviz_launchers view_navigation.launch
# Libraries required/to use: 
#   import cv2 - Stands for Computer Vision 2, and is to Process images and make decisions based on them, and the library is called OpenCV2
#   import numpy as np
#   from cv_bridge import CvBridge, CvBridgeError - Due to handling ROS, this library is needed as it can translate ROS images into Images readable by OpenCV
#   from sensor_msgs.msg import Image

# Receiving & Processing Camera Image Data 
#   Must subscribe to the topic the camera outputs too
#   RGB iamge from teh 3D sensor is on the topic : camera/rgb/image_raw

# Converting Between OpenCV2 and ROS Images
# To be able to manipulate the image data, you must convert between the arrived ROS image to the OpenCV type, using the built in OpenCV2 function:
#   imgmsg_to_cv2(data, "bgr8") - do this on a CvBridge object you have created in your class
#   Always a wise idea to wrap calls to this method in an exception handler incase anything goes wrong with the camera feed
#   Output the camera feed to the screen

# Displaying Image on New Window
# To Do this, you can use the inbuilt cv2 Functions as follows:
#   cv2.namedWindow('window_name')
#   cv2.imshow('window_name',<image name>)
#   cv2.waitKey(3)
# ---------------------------------------------------------------
# Simulated localisation/automated script
# ---------------------------------------------------------------





# Dont forget to add the libraries to package.xml and the scripts used to CMakeLists.txt
#-------------------------------------------------------------------------
#To do:
# Code way to confirm that the detected colour is indeed a circle - Done
# Code way to associate the detected circle with it's corresponding door, and how to confirm this is the correct door (most likely using the shortest distance between the circle and doors co-ordinate of the door and the circle on the map)
# Code way to detect Rectangular shape for cluedo image, and verify this shape
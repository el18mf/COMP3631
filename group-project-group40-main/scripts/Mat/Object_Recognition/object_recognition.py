#Object Recognition Script for Cluedo Robot - Group 40 - Mathew Fuller


from __future__ import division
import cv2
import numpy as np
import rospy
import sys


from geometry_msgs.msg import Twist, Vector3
#added bumper from lab 3 just incase I wanted to make things really fancy
from kobuki_msgs.msg import BumperEvent
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from os import system, name

class colourIdentifier():
    print('fuck')
    print('what the hell')
    def __init__(self):
        print('fuck')
        #Initialise flags
        #self.door_found = False
        self.red_found = False
        self.green_found = False
        #self.Cluedo = False
        
        #initialise colour detection sensitivity with default placeholder of 10 for now
        self.sensitivity = 10 
        
        #Initialises CvBridge
        self.bridge = CvBridge()
        #Subscriber set to the image topic whose image data we wish to receive
        self.sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)
    print('test')  
    def callback(self, data):
        print('callback')
        try:
            #set the image data received to a variable with the type BGR 8 bit color
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            #set upper and lower bounds for the 2 colours we want to identify
            hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
            hsv_red_lower = np.array([0 - self.sensitivity, 100, 100])
            hsv_red_upper = np.array([0 + self.sensitivity, 255, 255])
            
            #convert the cv_image from BGR to HSV using the inbuilt cv2 function
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            #create the colour masks that filter out all but the wanted colours
            g_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
            r_mask = cv2.inRange(hsv_image, hsv_red_lower, hsv_red_upper)
            rg_mask = cv2.bitwise_or(g_mask, r_mask)
            
            #attach the mask to the original image using the cv2.bitrwise_and() method
            #Best way to do this is to bitwise an image with itself and pass the mask to the mask parameter
            g_image = cv2.bitwise_and(hsv_image, hsv_image, mask=g_mask)
            r_image = cv2.bitwise_and(hsv_image, hsv_image, mask=r_mask)
            rg_image = cv2.bitwise_and(hsv_image, hsv_image, mask=rg_mask)
            
            #identify the contours that appear within the colour masks usig cv2.findContours()
            #Red
            r_contours, r_hierarchy = cv2.findContours(r_mask, mode = cv2.RETR_LIST,method = cv2.CHAIN_APPROX_SIMPLE)
            #Green
            g_contours, g_hierarchy = cv2.findContours(g_mask, mode = cv2.RETR_LIST,method = cv2.CHAIN_APPROX_SIMPLE)
        
        except CvBridgeError as e:
            print(e)

        #Green detected
        if len(g_contours) != 0:
            #set c to the detected contour area of green
            c = max(g_contours, key=cv2.contourArea)

            #now check if the coloured object is close enough to effectively be confirmed as the green circle we wish to identify. 
            #a value of 500 was chosen as 200 and below is the cut off for the Turtlebot 3's sensor, and so a minimum of 500 seemed reasonable to give it some wiggle room to avoid mistaking other objects coloured similarly to the green circle, in case they're present
            if cv2.contourArea(c) > 1000:
                
                #Draw a circle on the contour being identified
                #minEnclosingCircle can find the centre and radius of the largest contour (result from max())
                (x,y), radius = cv2.minEnclosingCircle(c)
                
                #needed to change the values to integers or it'd cause an error
                centre = (int(x),int(y))
                radius = int(radius)
                
                # cv2.circle(<image>,(<center x>,<center y>),<radius>,<colour (bgr tuple)>,<thickness (defaults to 1)>)
                cv2.circle(img = rg_image, center = centre, radius = radius, color = (0,255,0), thickness = 3)
                
                #Alter the value of the corresponding flag, i.e green
                self.green_found = True
        else:
                self.green_found = False    
            
        # Red Detected
        if len(r_contours) != 0:
            #set c to the detected contour area of green
            c = max(r_contours, key=cv2.contourArea)

            #now check if the coloured object is close enough to effectively be confirmed as the green circle we wish to identify. 
            #a value of 500 was chosen as 200 and below is the cut off for the Turtlebot 3's sensor, and so a minimum of 500 seemed reasonable to give it some wiggle room to avoid mistaking other objects coloured similarly to the green circle, in case they're present
            if cv2.contourArea(c) > 1000:
                
                #Draw a circle on the contour being identified
                #minEnclosingCircle can find the centre and radius of the largest contour (result from max())
                (x,y), radius = cv2.minEnclosingCircle(c)
                
                #needed to change the values to integers or it'd cause an error
                centre = (int(x),int(y))
                radius = int(radius)
                
                # cv2.circle(<image>,(<center x>,<center y>),<radius>,<colour (bgr tuple)>,<thickness (defaults to 1)>)
                cv2.circle(img = rg_image, center = centre, radius = radius, color = (0,0,255), thickness = 3)
                
                #Alter the value of the corresponding flag, i.e green
                self.red_found = True
        else:
                self.red_found = False        
        
        #if statements checking if the flags are True, and if so, figuring out the map co-ordinates of the circles and/or the nearest door to the located circle and setting that door as enter (if green is nearest) or avoid (if red is nearest)
        print("\n==================================================\n")
        if self.green_found == True:
            print('Green detected')   
        else:
            print("Green Not Detected")
        if self.red_found == True:
            print('Red Detected')
        else:
            print('Red Not Detected')
        # # standard View
        # cv2.namedWindow('Normal_Vision')
        # cv2.imshow('Normal_Vision', cv_image)
        # cv2.waitKey(3)
        # # Detective View
        # cv2.namedWindow('Detective_Vision')
        # cv2.imshow('Detective_Vision', rg_image)
        # cv2.waitKey(3)
        
        cv2.namedWindow('camera_Feed')
        cv2.imshow('camera_Feed', cv_image)
        cv2.waitKey(3)
        #red mask view
        cv2.namedWindow('camera_Feed_RGB')# modify your thresholds
        cv2.imshow('camera_Feed_RGB', rg_image)
        cv2.waitKey(3)  
        
def main(args):
    # instantiate your class
    # And rospy.init the entire node
    #INCLUDE cI
    cI = colourIdentifier()
    rospy.init_node('colourIdentifier', anonymous=True)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Clocking Out")
    #remember to destroy all image windows before closing the node
    cv2.destroyAllWindows()
    #clear teh screen
    _ = system('clear')
    
# check if the node is executing in the main path    
if __name__ == '__main__':
    main(sys.argv)

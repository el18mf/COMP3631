# Exercise 3 - If green object is detected, and above a certain size, then send a message (print or use lab2)

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
    def __init__(self):
        # Initialise any flags that signal a colour has been detected (default to false)
        self.red_found = False
        self.green_found = False
        self.blue_found = False
        # Initialise the value you wish to use for sensitivity in the colour detection (10 should be enough)
        self.sensitivity = 10
        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        self.bridge = CvBridge()
        # We covered which topic to subscribe to should you wish to receive image data
        self.sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)
    
    def callback(self, data):
        # Convert the received image into a opencv image
        # But remember that you should always wrap a call to this conversion method in an exception handler
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Convert the rgb image into a hsv imageprint
            Hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            # Set the upper and lower bounds for the colour you wish to identify - green
            #Green Limits
            hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
            
            #Red limits
            hsv_red_lower = np.array([0  - self.sensitivity, 100, 100])
            hsv_red_upper = np.array([0 + self.sensitivity, 255, 255])
            
            #Blue limits
            hsv_blue_lower = np.array([120  - self.sensitivity, 100, 100])
            hsv_blue_upper = np.array([120 + self.sensitivity, 255, 155])
          
            # Filter out everything but particular colours using the cv2.inRange() method
            # Do this for each colour
            #Green
            g_mask = cv2.inRange(Hsv_image, hsv_green_lower, hsv_green_upper) 
            # inv_mask = cv2.bitwise_not(mask)
            green_image = cv2.bitwise_and(Hsv_image, Hsv_image, mask= g_mask)
            
            #Red 
            r_mask = cv2.inRange(Hsv_image, hsv_red_lower, hsv_red_upper) 
            # inv_mask = cv2.bitwise_not(mask)
            red_image = cv2.bitwise_and(Hsv_image, Hsv_image, mask= r_mask)
            
            #Blue
            b_mask = cv2.inRange(Hsv_image, hsv_blue_lower, hsv_blue_upper) 
            # inv_mask = cv2.bitwise_not(mask)
            blue_image = cv2.bitwise_and(Hsv_image, Hsv_image, mask= b_mask)

        except CvBridgeError as e:
            print(e)
            
            
        rg_mask = cv2.bitwise_or(r_mask, g_mask)
        rgb_mask = cv2.bitwise_or(rg_mask, b_mask)
        rgb_image = cv2.bitwise_and(Hsv_image,Hsv_image, mask= rgb_mask)
        # Find the contours that appear within the certain colour mask using the cv2.findContours() method
        # For <mode> use cv2.RETR_LIST for <method> use cv2.CHAIN_APPROX_SIMPLE
        #Red
        r_contours, r_hierarchy = cv2.findContours(r_mask, mode = cv2.RETR_LIST,method = cv2.CHAIN_APPROX_SIMPLE)
        #Green
        g_contours, g_hierarchy = cv2.findContours(g_mask, mode = cv2.RETR_LIST,method = cv2.CHAIN_APPROX_SIMPLE)
        #Blue
        b_contours, b_hierarchy = cv2.findContours(b_mask, mode = cv2.RETR_LIST,method = cv2.CHAIN_APPROX_SIMPLE)
        #forgot to include the hierarchy above, which caused it not to work for ages :( just adding a comment here to remind me not to make the same silly mistake again.
        
        # if len(contours) > 0:
            # Loop over the contours
            # There are a few different methods for identifying which contour is the biggest:
            # Loop through the list and keep track of which contour is biggest or
            # Use the max() method to find the largest contour
            #c = max(<contours>, key=cv2.contourArea)
        #RED
        if len(r_contours) != 0:
            #print(contours)
            c = max(r_contours, key=cv2.contourArea)
            #print("This is the max contour: ", c, " Damn") 
            
                
            #Moments can calculate the center of the contour
            # M = cv2.moments(c)
            M = cv2.moments(c)
            # cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            ##M = cv2.moments(c)
            M = cv2.moments(c)
            #Check if the area of the shape you want is big enough to be considered, with the area I chose as 3.142 (pi)
            # If it is then change the flag for that colour to be True(1)
            #chose 500 as at that distance, if the colour is not singular on the object, it could be quite difficult to discern the actual colour. Note: below 200 the object disappears completely from the mask
            if cv2.contourArea(c) > 500: #<What do you think is a suitable area?>: 
    
                # draw a circle on the contour you're identifying
                #minEnclosingCircle can find the centre and radius of the largest contour(result from max())
                #(x, y), radius = cv2.minEnclosingCircle(c)
                (x, y), radius = cv2.minEnclosingCircle(c)
                #print(x,y,radius)
                #received the following error which just showed that before drawing the circle, the co-ordinates and radius needed to be converted to an integer using int()
                centre = (int(x),int(y))
                radius = int(radius)
                # cv2.circle(<image>,(<center x>,<center y>),<radius>,<colour (rgb tuple)>,<thickness (defaults to 1)>)
                cv2.circle(img = rgb_image,center=centre,radius=radius,color=(0,0,255),thickness=5)
                # Then alter the values of any flags
                self.red_found = True
                area_C = cv2.contourArea(c)
                print("Red Area is: ", area_C)
        else:
            self.red_found = False
        
        #green    
        if len(g_contours) != 0:
                    #print(contours)
                    c = max(g_contours, key=cv2.contourArea)
                    #print("This is the max contour: ", c, " Damn") 
                    
                        
                    #Moments can calculate the center of the contour
                    # M = cv2.moments(c)
                    M = cv2.moments(c)
                    # cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    ##M = cv2.moments(c)
                    M = cv2.moments(c)
                    #Check if the area of the shape you want is big enough to be considered, with the area I chose as 3.142 (pi)
                    # If it is then change the flag for that colour to be True(1)
                    if cv2.contourArea(c) > 31.42: #<What do you think is a suitable area?>: 
            
                        # draw a circle on the contour you're identifying
                        #minEnclosingCircle can find the centre and radius of the largest contour(result from max())
                        #(x, y), radius = cv2.minEnclosingCircle(c)
                        (x, y), radius = cv2.minEnclosingCircle(c)
                        #print(x,y,radius)
                        #received the following error which just showed that before drawing the circle, the co-ordinates and radius needed to be converted to an integer using int()
                        centre = (int(x),int(y))
                        radius = int(radius)
                        # cv2.circle(<image>,(<center x>,<center y>),<radius>,<colour (rgb tuple)>,<thickness (defaults to 1)>)
                        cv2.circle(img = rgb_image,center=centre,radius=radius,color=(0,255,0),thickness=5)
                        # Then alter the values of any flags
                        self.green_found = True
        else:
                    self.green_found = False
        #Blue    
        if len(b_contours) != 0:
                    #print(contours)
                    c = max(b_contours, key=cv2.contourArea)
                    #print("This is the max contour: ", c, " Damn") 
                    
                        
                    #Moments can calculate the center of the contour
                    # M = cv2.moments(c)
                    M = cv2.moments(c)
                    # cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    ##M = cv2.moments(c)
                    M = cv2.moments(c)
                    #Check if the area of the shape you want is big enough to be considered, with the area I chose as 3.142 (pi)
                    # If it is then change the flag for that colour to be True(1)
                    if cv2.contourArea(c) > 31.42: #<What do you think is a suitable area?>: 
            
                        # draw a circle on the contour you're identifying
                        #minEnclosingCircle can find the centre and radius of the largest contour(result from max())
                        #(x, y), radius = cv2.minEnclosingCircle(c)
                        (x, y), radius = cv2.minEnclosingCircle(c)
                        #print(x,y,radius)
                        #received the following error which just showed that before drawing the circle, the co-ordinates and radius needed to be converted to an integer using int()
                        centre = (int(x),int(y))
                        radius = int(radius)
                        # cv2.circle(<image>,(<center x>,<center y>),<radius>,<colour (rgb tuple)>,<thickness (defaults to 1)>)
                        cv2.circle(img = rgb_image,center=centre,radius=radius,color=(255,0,0),thickness=5)
                        # Then alter the values of any flags
                        self.blue_found = True
        else:
                    self.blue_found = False
        #if the flag is true (colour has been detected)
            #print the flag or colour to test that it has been detected
            #alternatively you could publish to the lab1 talker/listener        
        [print("\n-----------------------------------------------------------\n")]
        if self.red_found == True:
            print("\nRed Detected!")
        else:
            print("\nRed Not Detected.")
        if self.blue_found == True:
            print("\nBlue Detected!")
        else:
            print("\nBlue Not Detected.")
        if self.green_found == True:
            print("\nGreen Detected!")
        else:
            print("\nGreen Not Detected.")
        if self.red_found != True and self.blue_found != True and self.green_found != True:
            print("\nNo Colour Detected.")
            
        
            
        #Show the resultant images you have created. You can show all of them or just the end result if you wish to.
        #standard view
        cv2.namedWindow('camera_Feed')
        cv2.imshow('camera_Feed', cv_image)
        cv2.waitKey(3)
        #red mask view
        cv2.namedWindow('camera_Feed_RGB')# modify your thresholds
        cv2.imshow('camera_Feed_RGB', rgb_image)
        cv2.waitKey(3)        
# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main(args):
    # Instantiate your class
    # And rospy.init the entire node
    cI = colourIdentifier()
    # Ensure that the node continues running with rospy.spin()
    # You may need to wrap it in an exception handler in case of KeyboardInterrupts
    rospy.init_node("colourIdentifier", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")    
    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()
    #clear the screen
    _ = system('clear')

# Check if the node is executing in the main path
if __name__ == '__main__':
    main(sys.argv)

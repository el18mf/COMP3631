#Resources
    #https://notebook.community/rkburnside/python_development/MasteringComputerVision/Lecture%204.4%20-%20Matching%20Contours%20Shape
    #https://learnopencv.com/shape-matching-using-hu-moments-c-python/
    #https://stackoverflow.com/questions/21132963/how-to-process-output-of-the-matchshapes-function
#Testing out the use of HuMoments 

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

#create the Class
class Detector():
    #create an initialisation section in case variables, flags etc. are needed to be initialised for this class
    def __init__(self):
                
        #flags
        #Red Colour Flags
        self.r_found = False
        #Green Colour Flags
        self.g_found = False
        #Flag for matchShapes Confirmation check
        self.circ_found = False

        #Sensitivity for colours and other things - Default of 10
        self.sensitivity = 10
        
        #Initliasation of CvBridge
        self.bridge = CvBridge()
        
        #Initialisation of the subscriber which fetches the Image data
        self.sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)
        
    #creation of Callback in which the Image data is sent to be analysed and manipulated
    def callback(self, data):
    
    #Remember to convert the raw image data into a format usuable by OpenCV through the use of bridge.imgmsg_to_cv2(data, type) and in our case we will change it to the bgr8 image type
        #Note: always perform this within an exception handler so that you can wrap a call to this conversion method in case there are errors by using try:
        try:
        #image conversion
            cv_i = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
        #create a version of the image in the hsv image type
            hsv_i = cv2.cvtColor(cv_i, cv2.COLOR_BGR2HSV)
            
    #Upper and Lower boundary arrays for colours that are wanted to be identified
            #Red Upper and Lower limits - Based on the provided red image which is #FE0000 or RGB(255, 0, 0) - The sensitivity at 10 was picking up the orange cones, which have a blue value of between 7-9, thus I change it down to 5 just to be safe, as while the size checker and also the Hu Moment matchShapes would be abloe to prevent these being mistaken for the circle, I thought it best to not take chances
            R_upper = np.array([0 - 5, 100, 100])
            R_lower = np.array([0 + 5, 255, 255])
            #Green Upper and Lower Limits - Based on the provided green image which is the colour #00af50 or RGB(0, 175, 80) - Needed a larger sensitivity for green as 10 wouldn't pick up the shade of green that the example word uses for it's green circle
            G_upper = np.array([60 - 20, 100, 100])
            G_lower = np.array([60 + 20, 255, 255])
        
    #Reading Image files for comparison, and importing them as greyscale versions
            #Coloured Circles
            Circle_i = cv2.imread("/home/csunix/el18mf/catkin_ws/src/group_project/scripts/Mat/Circle.jpg", cv2.IMREAD_GRAYSCALE)
            #Cluedo Characters
            Mustard = cv2.imread("/home/csunix/el18mf/catkin_ws/src/group_project/cluedo_images/mustard.png", cv2.IMREAD_GRAYSCALE)
            Peacock = cv2.imread("/home/csunix/el18mf/catkin_ws/src/group_project/cluedo_images/peacock.png", cv2.IMREAD_GRAYSCALE)
            Plum = cv2.imread("/home/csunix/el18mf/catkin_ws/src/group_project/cluedo_images/plum.png", cv2.IMREAD_GRAYSCALE)
            Scarlet = cv2.imread("/home/csunix/el18mf/catkin_ws/src/group_project/cluedo_images/scarlet.png", cv2.IMREAD_GRAYSCALE)
            
    #create colour masks that filter out all but colours in the specified boundaries
            #Red Mask
            R_mask = cv2.inRange(hsv_i, R_upper, R_lower)
            #Green Mask
            G_mask = cv2.inRange(hsv_i, G_upper, G_lower)
            
            #Red & Green Mask by combining the Red and Green masks, by using the bitwise_or operator
            RG_mask = cv2.bitwise_or(R_mask, G_mask)
                      
    #create colour masked images
            #Red masked Image
            R_i = cv2.bitwise_and(hsv_i, hsv_i, mask = R_mask)
            #Green masked Image
            G_i = cv2.bitwise_and(hsv_i, hsv_i, mask = G_mask)

            
            #Red & Green masked Image
            RG_i = cv2.bitwise_and(hsv_i, hsv_i, mask = RG_mask)     
           
        #creating the excpetion handler which prints any errors         
        except CvBridgeError as e:
            print(e)
  
#----------------------------------------------------------------------------------------------------------------------------------------------------------
          
#Setting up if statements that detect if any of the colours earmarked for detection are found
    #Calculation of the Moments & Hu moments of each colour. Note: remember to include the int() so that the values are calculated as integers, otherwise errors shall occur
        #Identify the contours within the image and colour masked images. To note: the _hierarchy is required or the code shan't function, however we have no use for it
        #Red Contours
        R_contours, R_hierarchy = cv2.findContours(R_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
        #Green Contours
        G_contours, G_hierarchy = cv2.findContours(G_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)

        Green_Area = 0
        Red_Area = 0

        if len(R_contours) != 0 or len(G_contours) != 0:
            if len(R_contours) != 0:
                self.r_found = True
                r_c = max(R_contours, key = cv2.contourArea, default = 0.1)
                Red_Area = cv2.contourArea(r_c)
            #    print('Red Detected, anaylsing shape')
            if len(G_contours) != 0:
                g_c = max(G_contours, key = cv2.contourArea, default = 0)
                Green_Area = cv2.contourArea(g_c)
                self.g_found = True   
            # print('Green Detected, anaylsing shape')
    
    #Checks if Red or Green are detected
        if self.r_found == True or self.g_found == True:

        #using max(<contours>, key) to find the largest contour

            print("__________________________________________________________________________________________________\n")
            print("\nThe Contour Area of Red is: ", Red_Area)
            print("\nThe Contour Area of Green is: ", Green_Area)
            #check if the detected colour has a suitable enough contour area size to be analysed - The value of 1500 was chosen to give leeway to the robot as it rotates to detect the circle in case it is not in the correct orientation as it reaches the doorway co-ordinates, but also big enough for it to not detect the circle at the other door
            if Red_Area > 1500 or Green_Area > 1500:
            #Calculation of Hu Moments of the Grayscale Comparison Circle Image
                #Binarisation of Grayscale image - Creation of the Threshold Image - can't tell if I should use 127 or 128, as different examples I found used them interchangeably 
                _, Template_t = cv2.threshold(Circle_i, 127, 255, 0)
            
                #Calculate the moments of the binary image
                Circ_M = cv2.moments(Template_t)
                
                #Calculate the Hu Moments
                Circ_Hu = cv2.HuMoments(Circ_M)

            #Gray Scaling and Binarisation of the Red & Green Images
                #Red
                R_i_grey = cv2.cvtColor(R_i, cv2.COLOR_BGR2GRAY)
                _, R_t = cv2.threshold(R_i_grey, 127, 255, 0)
                #Green
                G_i_grey = cv2.cvtColor(G_i, cv2.COLOR_BGR2GRAY)
                _, G_t = cv2.threshold(G_i_grey, 127, 255, 0)
            
            #Calculating how similar the detected image is in comparison to the Circle Image, using cv2.matchShapes
                #Note: There are 3 distance equations, CONTOURS_MATCH_I1/2/3, and I have gone with equation 2 (for ∑ and H, before^∑ = on top/upper | ∑^below/lower: D(A, B) = ∑^i=0 |B^H^i - A^H^i|
                #Binarised Red Mask Image and Binarised Template
                R_match = cv2.matchShapes(R_t,Template_t, cv2.CONTOURS_MATCH_I2, 0)
                #Binarised Green masked Image and Binarised Template
                G_match = cv2.matchShapes(G_t, Template_t, cv2.CONTOURS_MATCH_I2, 0)
                
            #If statement to check if the matchShapes value - using sources, I landed on a value of less than 0.15 as a good matchShapes output
            #A "perfect" match seems to have the value of 0.4735538897196503  which is roughly the same for both Red & green
                if R_match < 0.4735538897196503: R_percent_match = round((R_match/0.4735538897196503)*100,2)
                if G_match < 0.4735538897196503: G_percent_match = round((G_match/0.4735538897196503)*100, 2)
                if R_match > 0.4735538897196503: R_percent_match = round((0.4735538897196503/R_match)*100, 2)
                if G_match > 0.4735538897196503: G_percent_match = round((0.4735538897196503/G_match)*100, 2)

                print("__________________________________________________________________________________________________\n")
                #75% either way of 0.4735538897196503 is 0.35516541729 and 0.59194236215, thus I made the upper and lower boundaries 0.355 and 0.592 to be as close to a 75% accuracy as possible
                if R_match > 0.355 and R_match < 0.592 and self.r_found == True:
                    self.r_found = True
                    print("\n The Red Match Value is: ", R_match)
                    print("\n Which is a ", str(R_percent_match)+"% Match!")
                    print ("\nRed Circle Found! Navigating to Alternative Doorway Co-ordinates.")
                elif G_match > 0.35 and G_match < 0.55 and self.g_found == True:
                    self.g_found = True
                    print("\n The Green Match Value is: ", G_match) 
                    print("\n Which is a ", str(G_percent_match)+"% Match!")
                    print ("\nGreen Circle Found! Navigating to corresponding doorway and entering.") 
                else:
                    self.r_found, self.r_found, self.circ_found = False, False, False
                    print('\nFalse Alarm - Does not Match Circle Template')
                    print("\n The Red Match Value is: ", R_match)
                    print("\n Which is a ", str(R_percent_match)+"% Match!")
                    print("\n The Green Match Value is: ", G_match)
                    print("\n Which is a ", str(G_percent_match)+"% Match!")
                print("__________________________________________________________________________________________________\n\n")  
  
        
        
        #displaying the images
        cv2.namedWindow('Colour Vision')
        cv2.imshow("Colour Vision", RG_i)
        cv2.waitKey(5)
        
        cv2.namedWindow('Detective View')
        cv2.imshow("Detective View", cv_i)
        cv2.waitKey(5)



def main(args):
    
    #Instantiate your class and rospy.init the entire node
    cI = Detector()
    # c2I = CluedoHu()
    
    #ensure that the ndoe continues running with rospy.spin()
    #You may need to wrap it in an exception handler in case of KeyboardInterrupts
    rospy.init_node("Detector", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down!")
    
    #Remember to clean up with destroy all windows
    cv2.destroyAllWindows()
    
#check if the node is executing in the main path
if __name__ == '__main__':
    main(sys.argv)
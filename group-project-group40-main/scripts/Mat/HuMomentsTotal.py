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
        self.r_circ_found = False
        self.g_circ_found = False
        #Cluedo Character Flags
        self.must_sus = False
        self.pea_sus = False
        self.plum_sus = False
        self.scar_sus = False
        self.killer_found = False

        #Variable for suspected killers
        self.suspect = ""
        self.potential_killer = ""
        self.confirmed_killer = ""

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
            R_lower = np.array([0 - 5, 100, 100])
            R_upper = np.array([0 + 5, 255, 255])
            #Green Upper and Lower Limits - Based on the provided green image which is the colour #00af50 or RGB(0, 175, 80) - Needed a larger sensitivity for green as 10 wouldn't pick up the shade of green that the example word uses for it's green circle
            G_lower = np.array([60 - 20, 100, 100])
            G_upper = np.array([60 + 20, 255, 255])
        #The Cluedo Character Colours Upper and lower bounds, worked out in a separate excel sheet by using a palette of 5 for each image, with a range of the lightest to darkest shades of their corresponding in each picture
            #Mustard Upper and Lower
            M_lower = np.array([30 - 11, 75, 50])
            M_upper = np.array([30 + 15, 255, 255])
            #Peacock Upper and Lower
            P_lower = np.array([116 - 18, 30, 0])
            P_upper = np.array([116 + 18, 255, 255])
            #Plum Upper and Lower
            Plum_lower = np.array([144 - 17, 31, 18])
            Plum_upper = np.array([144 + 17, 133, 100])
            # Scarlett Upper and Lower - from tweaking the values, this set up is as close to the full detection of scarlets colour while also minimising the picking up of the cones
            S_lower = np.array([3 - 4, 85, 50])
            S_upper = np.array([3 + 4, 190, 235])
        ######### note: I think the + and - are the wrong way round, test this ########################### - I was correct
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
            R_mask = cv2.inRange(hsv_i, R_lower, R_upper)
            #Green Mask
            G_mask = cv2.inRange(hsv_i, G_lower, G_upper)
        #Cluedo Masks
            #Mustard
            M_mask = cv2.inRange(hsv_i, M_lower, M_upper)
            #Peacock
            P_mask = cv2.inRange(hsv_i, P_lower, P_upper)
            #Plum
            Plum_mask = cv2.inRange(hsv_i, Plum_lower, Plum_upper)
            #Scarlet
            S_mask = cv2.inRange(hsv_i, S_lower, S_upper)    
            
            #Red & Green Mask by combining the Red and Green masks, by using the bitwise_or operator
            RG_mask = cv2.bitwise_or(R_mask, G_mask)
            
        #Mask for all Cluedo Images
            MP_mask = cv2.bitwise_or(M_mask, P_mask)
            PMPlum_mask = cv2.bitwise_or(MP_mask, Plum_mask)
            Cluedo_mask = cv2.bitwise_or(PMPlum_mask, S_mask)
        #combination of all masks
            All_mask = cv2.bitwise_or(Cluedo_mask, RG_mask)
                      
    #create colour masked images
            #Red masked Image
            R_i = cv2.bitwise_and(hsv_i, hsv_i, mask = R_mask)
            #Green masked Image
            G_i = cv2.bitwise_and(hsv_i, hsv_i, mask = G_mask)
      
        #Cluedo Images
            #Mustard Image
            M_i = cv2.bitwise_and(hsv_i, hsv_i, mask = M_mask)
            #Peacock Image
            P_i = cv2.bitwise_and(hsv_i, hsv_i, mask = P_mask)
            #Plum Image
            Plum_i = cv2.bitwise_and(hsv_i, hsv_i, mask = Plum_mask)
            #Scarlet Image
            S_i = cv2.bitwise_and(hsv_i, hsv_i, mask = S_mask)
            #Cluedo Image
            Cluedo_i = cv2.bitwise_and(hsv_i, hsv_i, mask = Cluedo_mask)

            #Red & Green masked Image
            RG_i = cv2.bitwise_and(hsv_i, hsv_i, mask = RG_mask)     
            
            #combined image of both coloured circles and cluedo characters
            All_i = cv2.bitwise_and(hsv_i, hsv_i, mask = All_mask)
  
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
        if self.g_circ_found != True:
            if self.r_found == True or self.g_found == True:

            #using max(<contours>, key) to find the largest contour

                print("__________________________________________________________________________________________________\n")
                print("\nThe Contour Area of Red is: ", Red_Area)
                print("\nThe Contour Area of Green is: ", Green_Area)
                #check if the detected colour has a suitable enough contour area size to be analysed - The value of 1500 was chosen to give leeway to the robot as it rotates to detect the circle in case it is not in the correct orientation as it reaches the doorway co-ordinates, but also big enough for it to not detect the circle at the other door
                if Red_Area > 750 or Green_Area > 750:
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
                #A "perfect" match seems to have the value of 0.4735538897196503 0.47340489351 which I'll round to 6 s.f. to 0.473404 for red, and 0.3767295827903103 -> 0.376729 for green
                    if R_match < 0.473404: R_percent_match = round((R_match/0.473404)*100,2)
                    if G_match < 0.376729: G_percent_match = round((G_match/0.376729)*100, 2)
                    if R_match > 0.473404: R_percent_match = round((0.473404/R_match)*100, 2)
                    if G_match > 0.376729: G_percent_match = round((0.376729/G_match)*100, 2)

                    print("__________________________________________________________________________________________________\n")
                    #75% either way of 0.4735538897196503 is 0.35516541729 and 0.59194236215, thus I made the upper and lower boundaries 0.355 and 0.592 to be as close to a 75% accuracy as possible
                    if R_match > 0.355 and R_match < 0.592 and self.r_found == True:
                        self.r_circ_found = True
                        print("\n The Red Match Value is: ", R_match)
                        print("\n Which is a ", str(R_percent_match)+"% Match!")
                        print ("\nRed Circle Found! Navigating to Alternative Doorway Co-ordinates.")
                    elif G_match > 0.35 and G_match < 0.55 and self.g_found == True:
                        self.g_circ_found = True
                        print("\n The Green Match Value is: ", G_match) 
                        print("\n Which is a ", str(G_percent_match)+"% Match!")
                        print ("\nGreen Circle Found! Navigating to corresponding doorway and entering.")
                        print("Room Found, the murderer is here somewhere, we shall find them!")
                        self.g_circ_found = True 
                    else:
                        self.r_found, self.r_found = False, False
                        print('\nFalse Alarm - Does not Match Circle Template')
                        print("\n The Red Match Value is: ", R_match)
                        print("\n Which is a ", str(R_percent_match)+"% Match!")
                        print("\n The Green Match Value is: ", G_match)
                        print("\n Which is a ", str(G_percent_match)+"% Match!")
                    print("__________________________________________________________________________________________________\n\n")
                    print(self.g_circ_found) 
        #Checks to see if the green circle has been located already, and if so, it switches from searching for the coloured circles to searching for the cluedo character portraits
    
        if self.g_circ_found == True:
            #announce that you are now searching for the suspect
            #find the contours for the cluedo images
            #Mustard
            M_contours, M_hierarchy = cv2.findContours(M_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
            #Peacock
            P_contours, P_hierarchy = cv2.findContours(P_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
            #Plum
            Plum_contours, Plum_hierarchy = cv2.findContours(Plum_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
            #Scarlet
            S_contours, S_hierarchy = cv2.findContours(S_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
            
            #create threshold/binarisation images of the cluedo images
            _, M_t = cv2.threshold(Mustard, 127, 255,0)
            _, P_t = cv2.threshold(Peacock, 127, 255, 0)
            _, Plum_t = cv2.threshold(Plum, 127, 255, 0)
            _, S_t = cv2.threshold(Scarlet, 127, 255, 0)
            
            #create lists of the contours and flags of the cluedo characters for use in a for loop                     
            Cluedo_Contours = [M_contours, P_contours, Plum_contours, S_contours]
            Cluedo_Flags = [self.must_sus, self.pea_sus, self.plum_sus, self.scar_sus]
            Cluedo_images = [M_i, P_i, Plum_i, S_i]
            Cluedo_Names = ["Colonel Mustard", "Mrs Peacock", "Professor Plum", "Miss Scarlett"]
            Cluedo_templates = [M_t, P_t, Plum_t, S_t]
            Cluedo_HuValues = [0.28850991041393925,0.09909818671542636,0.32761774487025885,0.14735167396976978]
            
            #print(S_contours)
            for i in range(0,4):
                # print(i)
                if len(Cluedo_Contours[i]) != 0:
                    # print(Cluedo_Names[i])
                    #find the max contour
                    c = max(Cluedo_Contours[i], key = cv2.contourArea, default = 0)
                    #Find the area of this maximum contour for use in size limitation
                    area = cv2.contourArea(c)                 
                    #sets the corresponding characters flag to true
                    Cluedo_Flags[i] = True
                    #setting image to the gray scaled version of the detected image
                    image = cv2.cvtColor(Cluedo_images[i], cv2.COLOR_BGR2GRAY)
                    #set the percentage match value 
                    percent = Cluedo_HuValues[i]
                    suspect = Cluedo_Names[i]
                    if area > 100:
                        print("__________________________________________________________________________________________________\n")
                        print(Cluedo_Names[i], " has possibly been spotted at the Murder Scene!\n")
                        print("==================================================================================================\n")
                        #set the template character image that the detected 
                        template = Cluedo_templates[i]
                        match = cv2.matchShapes(image, template, cv2.CONTOURS_MATCH_I2, 0) 
                        print("\nThe shapeMatch value of the detected image is: ", match)
                        print(match)
                        print(percent)
                        if match < percent: percent_match = round((match/percent)*100,2)
                        if match > percent: percent_match = round((percent/match)*100, 2)
                        print("I am ", str(percent_match) + "%", "sure that ", suspect, "is the Murder!" ) 
                        print("__________________________________________________________________________________________________\n\n")
                        if percent_match > 85:
                            print("\nJ'accuse! We have found them, the dasterdly Murderer. We have found the villain and thy name is ", str(suspect) + "!!!")
                            print("\nMy Work here is done. However if there is ever the need for justice, you know who to call...Inspector Cv2louseau!")
                        elif percent_match > 50 and percent_match < 85:
                            print("\nIt is very likely they are the killer, however there is still a shadow of a doubt and we must try find more evidence or a more likely suspect before judgement!")
                            self.potential_killer = suspect
                        elif percent_match < 50:
                            print("\nHowever I am not certain yet, there is only a 50% chance they are indeed the suspect. I shall continue searching for further suspects!")
                            self.suspect = suspect                                
                    # else:
                    #     print("Searching for Suspect!")
            #From Testing, Character Related Info:
                #Area detected -
                    #Mustard - Similar to scarlet, the top of one of the cones is picked up but nothing else, 
                    #Plum
                    #Peacock
                    #Scarlett = from half the room away, it's still 2500, and the largest other object picked up is one of the cones, which has a max (straight next to camera) area of 350. I think it's safe to set it at 750 to minimise false ID's 
                #Hu shapeMatch Values
                    #Mustard = 0.28850991041393925
                    #Plum = 0.09909818671542636
                    #Peacock = 0.32761774487025885
                    #Scarlett = 0.14735167396976978
            # if True in Cluedo_Flags:
            #     #can put in area to help guide the robot closer to the image for a better match
            #     if area > 500:
            #         match = cv2.matchShapes(image, template, cv2.CONTOURS_MATCH_I2, 0) 
            #         print("The shapeMatch value of the detected image is: ", match)
            #         if match < 0.473404: percent_match = round((match/0.473404)*100,2)
            #         if match > 0.473404: percent_match = round((0.473404/match)*100, 2)      
                    
               
        cv2.namedWindow('Detective View')
        cv2.imshow("Detective View", cv_i)
        cv2.waitKey(5)
        
        #displaying the images
        cv2.namedWindow('All Seeing Eye')
        cv2.imshow("All Seeing Eye", All_i)
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
       
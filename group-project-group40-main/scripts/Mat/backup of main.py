#Changes the / to mean True division throughout the code, for use in working out the centroid of the detected Shapes
from __future__ import division
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
from geometry_msgs.msg import Twist, Vector3

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


if __name__ == '__main__':
    # rospy.init_node('group_project')
    # Your code should go here. You can break your code into several files and
    # include them in this file. You just need to make sure that your solution 
    # can run by just running rosrun group_project main.py

    # Please note how to properly read/write to files: use similar approach for
    # the rest of your I/O in your project. You might need to use these functions
    # elsewhere this is just a demonstration of how to use them.

    # points = read_input_points()
    # write_text_file('mustard')
    # save_character_screenshot(cv_image)



    #create the Class
    class Detector():
        #create an initialisation section in case variables, flags etc. are needed to be initialised for this class
        def __init__(self):
                    
        #flags
        #Flag for matchShapes Confirmation check
            #Red Circle Found
            self.R_circ_found = False
            #Green Circle Found
            self.G_circ_found = False
            #Cluedo Character Flags
            self.killer_found = False

            #flag for timer
            self.timeout = False
            
            #Variable for suspected killers and their percent match
            self.suspect = ""
            self.suspect_match = 0
            self.suspect_time = 0
            
            self.potential_killer = ""
            self.potential_match = 0
            self.potential_time = 0
            
            self.confirmed_killer = ""
            self.confirmed_match = 0
            self.Cluedo_time = 0
            
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
                Circle_i = cv2.imread(f'{PROJECT_DIR}/scripts/Mat/Circle.jpg', cv2.IMREAD_GRAYSCALE)
                #Cluedo Characters
                Mustard = cv2.imread(f"{PROJECT_DIR}/cluedo_images/mustard.png", cv2.IMREAD_GRAYSCALE)
                Peacock = cv2.imread(f"{PROJECT_DIR}/cluedo_images/peacock.png", cv2.IMREAD_GRAYSCALE)
                Plum = cv2.imread(f"{PROJECT_DIR}/cluedo_images/plum.png", cv2.IMREAD_GRAYSCALE)
                Scarlet = cv2.imread(f"{PROJECT_DIR}/cluedo_images/scarlet.png", cv2.IMREAD_GRAYSCALE)
                
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
                
                #combined image of both coloured circles and cluedo characters
                All_i = cv2.bitwise_and(hsv_i, hsv_i, mask = All_mask)

            #creating the excpetion handler which prints any errors         
            except CvBridgeError as e:
                print(e)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------
            if time.time() - start < 300:
                #checks if the killer has been found yet, if it has then the robot stops working
                if self.killer_found != True:
                    #Checks to see if the green circle has been located already, and if so, it switches from searching for the coloured circles to searching for the cluedo character portraits           
                    if self.G_circ_found == False:
                        #find the contours for the Coloured Circles
                        #Red
                        R_contours, R_hierarchy = cv2.findContours(R_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
                        #Green
                        G_contours, G_hierarchy = cv2.findContours(G_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
                        
                    #Gray Scaling and Binarisation of the Red & Green Images
                        #Red
                        R_grey = cv2.cvtColor(R_i, cv2.COLOR_BGR2GRAY)
                        _, R_t = cv2.threshold(R_grey, 127, 255, 0)
                        #Green
                        G_grey = cv2.cvtColor(G_i, cv2.COLOR_BGR2GRAY)
                        _, G_t = cv2.threshold(G_grey, 127, 255, 0)
                        
                        #create lists of the contours and flags of the cluedo characters for use in a for loop                     
                        #contains the contours of each circle
                        Circle_Contours = [R_contours, G_contours]
                        Circle_images = [R_i, G_i]
                        Circle_Names = ["Red Circle", "Green Circle"]
                        Circle_HuValues = [0.10776837615821666,0.1255277836778994]
                        
                        #print(S_contours)
                        for i in range(0,2):
                            if len(Circle_Contours[i]) != 0:
                                #find the max contour
                                c = max(Circle_Contours[i], key = cv2.contourArea, default = 0)
                                #Find the area of this maximum contour for use in size limitation
                                area = cv2.contourArea(c)                 
                                #sets the corresponding characters flag to true
                                #setting image to the gray scaled version of the detected image
                                image = cv2.cvtColor(Circle_images[i], cv2.COLOR_BGR2GRAY)
                                #set the percentage match value 
                                percent = Circle_HuValues[i]
                                suspect = Circle_Names[i]
                                if area > 100:
                                    print("__________________________________________________________________________________________________\n")
                                    print(Circle_Names[i], "has possibly been spotted!\n")
                                    print("==================================================================================================")
                                    #set the template character image that the detected 
                                    #Compare the two images using matchShapes
                                    match = cv2.matchShapes(image, Circle_i, cv2.CONTOURS_MATCH_I2, 0) 
                                    #Set current colour flag to true as it is suspected
                                    print("The shapeMatch value of the detected image is: ", match)
                                    if match < percent: percent_match = round((match/percent)*100,2)
                                    if match > percent: percent_match = round((percent/match)*100, 2)
                                    print("The area is:", area)
                                    print("I am ", str(percent_match) + "%", "sure that", suspect, "has been found!" ) 
                                    print("__________________________________________________________________________________________________\n\n")
                                    if area > 1200:
                                        if percent_match > 75:
                                            if suspect == "Green Circle":
                                                print("\n The Green Match Value is: ", match) 
                                                print("\n Which is a ", str(percent_match)+"% Match!")
                                                print ("\nGreen Circle Found! Navigating to corresponding doorway and entering.")
                                                print("Room Found, the murderer is here somewhere, we shall find them!")
                                                self.Green_time = round(time.time() - start,2)
                                                print("Green Circle found in ", time.time() - start)
                                                self.G_circ_found = True
                                                #mark the nearest doors x,y co-ordinates as the correct door, and enter said door/autonomously move towards the centre of it's room  
                                            elif suspect == "Red Circle":
                                                print("\n The Red Match Value is: ", match)
                                                print("\n Which is a ", str(percent_match)+"% Match!")
                                                print ("\nRed Circle Found! Navigating to Alternative Doorway Co-ordinates.")
                                                self.Red_time = round(time.time() - start,2)
                                                self.R_circ_found = True
                                                #set nearest doors x,y co-ordinates equal the wrong room and head towards the other, therefore correct/green circle, room
                                        elif percent_match < 75:
                                            print("\nThis could be the circle, however it is not enough of a match to be certain!")
                    
                    if self.G_circ_found == True or self.R_circ_found == True:
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
                        Cluedo_images = [M_i, P_i, Plum_i, S_i]
                        Cluedo_Names = ["Colonel Mustard", "Mrs Peacock", "Professor Plum", "Miss Scarlett"]
                        Cluedo_templates = [M_t, P_t, Plum_t, S_t]
                        Cluedo_HuValues = [0.28850991041393925,0.32820465007282307,0.09692887987761445,0.14735167396976978]
                        #possible issue, peacock was 0.09909818671542636 but now is 0.32820465007282307
                        #print(S_contours)

                        for i in range(0,4):
                            # print(i)
                            if len(Cluedo_Contours[i]) != 0:
                                # print(Cluedo_Names[i])
                                #find the max contour
                                c = max(Cluedo_Contours[i], key = cv2.contourArea, default = 0)
                                #Find the area of this maximum contour for use in size limitation
                                area = cv2.contourArea(c)                 
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
                                    # print("\nThe shapeMatch value of the detected image is: ", match)
                                    # print(match)
                                    # print(percent)
                                    if match < percent: percent_match = round((match/percent)*100,2)
                                    if match > percent: percent_match = round((percent/match)*100, 2)
                                    print("I am ", str(percent_match) + "%", "sure that", suspect, "is the Murder!" ) 
                                    print("__________________________________________________________________________________________________\n\n")
                                    if area > 1500:
                                        if percent_match > 85:
                                            print("\nJ'accuse! We have found them, the dasterdly Murderer. We have found the villain and thy name is ", str(suspect) + "!!!")
                                            print("\nMy Work here is done. However if there is ever the need for justice, you know who to call...Inspector Cv2louseau!")
                                            self.Cluedo_time = round(time.time() - start,2)
                                            if self.confirmed_match < percent_match:
                                                self.confirmed_match = percent_match
                                            self.confirmed_killer = suspect
                                            self.confirmed_i = cv_i
                                            self.killer_found = True
                                            
                                        elif percent_match > 55 and percent_match < 85:
                                            print("\nIt is very likely they are the killer, however there is still a shadow of a doubt and we must try find more evidence or a more likely suspect before judgement!")
                                            if self.potential_time == 0 or round(time.time() - start < self.potential_time):
                                                self.potential_time = round(time.time() - start,2)
                                            self.potential_i = cv_i
                                            if self.potential_match < percent_match:
                                                self.potential_match = percent_match
                                            self.potential_killer = suspect
                                            
                                        elif percent_match < 50 and percent_match > 35:
                                            print("\nHowever I am not certain yet, there is only a 50% chance they are indeed the suspect. I shall continue searching for further suspects!")
                                            if self.suspect_time == 0 or round(time.time() - start < self.suspect_time):
                                                self.suspect_time = round(time.time() - start,2)
                                            self.suspect_i = cv_i
                                            if self.suspect_match < percent_match:
                                                self.suspect_match = percent_match
                                            self.suspect = suspect       
                                                                    
                elif self.killer_found == True:
                    #writing the killers details to the file
                    killer = "The Killer is " + self.confirmed_killer + ". They are a " + str(self.confirmed_match) + "% match to the description. They were found in " + str(self.Cluedo_time) + " seconds."
                    write_character_id(killer)
                    save_character_screenshot(self.confirmed_i)
                    #Ends the program
                    sys.exit()
            else:
                print("5 minute time limit reached.")
                #check if a suspect or potential killer were found, and if they were, write the more likely one to the file and save their image
                if self.potential_killer != "":
                    #Stringing together the potential killers name, the 
                    pkiller = "The searching time limit of 5 minutes was reached. \nA potential killer, " + self.potential_killer + ", was located. \nWhile we cannot confirm without a doubt that they are the killer, as they are only a " + str(self.potential_match) + "% match of the description, but it is our best guess in the time given. \nThey were found in " + str(self.potential_time) + " seconds."
                    write_character_id(pkiller)
                    save_character_screenshot(self.potential_i)
                    exit()
                elif self.suspect != "" and self.potential_killer == "":
                    skiller = "The searching time limit of 5 minutes was reached. \nA Suspect, " + self.suspect + ", was found at the scene. \nAs they were only a " + str(self.potential_match) + "%, we cannot say for certain they are the murderer, but it was the best guess in the time given. \nThey were found at " + str(self.suspect_time) + " seconds."
                    write_character_id(str(skiller))
                    save_character_screenshot(self.suspect_i)
                    #ends the program
                    sys.exit()
                
                    
            #displaying the compounded masked view
            cv2.namedWindow('Detective View')
            cv2.imshow("Detective View", cv_i)
            cv2.waitKey(5)
            
            #displaying what the robot sees
            cv2.namedWindow('All Seeing Eye')
            cv2.imshow("All Seeing Eye", All_i)
            cv2.waitKey(5)


    def main(args):
        
        #Instantiate your class and rospy.init the entire node
        cI = Detector()
        
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
        

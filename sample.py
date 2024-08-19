#find the contours for the cluedo images - Extracted from group-project-group40-main/scripts/main.py - lines 596 to 630
#Mustard
M_contours, M_hierarchy = cv2.findContours(M_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
#Peacock
P_contours, P_hierarchy = cv2.findContours(P_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
#Plum
Plum_contours, Plum_hierarchy = cv2.findContours(Plum_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
#Scarlet
S_contours, S_hierarchy = cv2.findContours(S_mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)     
#calc of camera's centroid
gray_image = cv2.cvtColor(cv_i, cv2.COLOR_BGR2GRAY)
M = cv2.moments(gray_image)
cv_X = int(M["m10"] / M["m00"])
cv_Y = int(M["m01"] / M["m00"])
#create threshold/binarisation images of the cluedo images
_, M_t = cv2.threshold(Mustard, 127, 255,0)
_, P_t = cv2.threshold(Peacock, 127, 255, 0)
_, Plum_t = cv2.threshold(Plum, 127, 255, 0)
_, S_t = cv2.threshold(Scarlet, 127, 255, 0)
#Moments, HuMoments and Centroid
M_moments = cv2.moments(M_t)
P_moments = cv2.moments(P_t)
Plum_moments = cv2.moments(Plum_t)
S_moments = cv2.moments(S_t)
#create lists of the contours and flags of the cluedo characters for use in a for loop                     
Cluedo_Contours = [M_contours, P_contours, Plum_contours, S_contours]
Cluedo_images = [M_i, P_i, Plum_i, S_i]
Cluedo_Names = ["Colonel Mustard", "Mrs Peacock", "Professor Plum", "Miss Scarlett"]
Cluedo_templates = [M_t, P_t, Plum_t, S_t]
Cluedo_HuValues = [0.28850991041393925,0.32820465007282307,0.09692887987761445,0.14735167396976978]

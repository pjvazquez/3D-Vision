import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import time 

# wrapper function for image capturing from camera
def camera_go():
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened(): 
        rval, frame = vc.read()
        oldgray = None
        print('Camera is opened \n')
    else:
        rval = False
        print('Camera is closed \n')
    
    # Keep the video stream open
    while rval:
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = frame
        
        # gray = cv2.resize(gray, (240,320) )

        # Plot the image from camera with all the face and eye detections marked
        cv2.imshow("captured image", gray)
        

        if oldgray is not None:
            substracted_image =  cv2.subtract(gray, oldgray)
            cv2.imshow("Frame diff", 255- substracted_image)
            transf_substracted_image = cv2.GaussianBlur(substracted_image, (5,5),0 )
            cv2.imshow("OPEN filter transf image", transf_substracted_image)
            

        # Press 'q' key to exit laptop video
        if cv2.waitKey(1) & 0xFF == ord('q'): # Exit by pressing q key
            # Destroy windows
            cv2.destroyAllWindows()
            
            for i in range (1,5):
                cv2.waitKey(1)
            return
            break
        
        # Read next frame
        # control framerate for computation - default 20 frames per sec
        time.sleep(0.05)
        oldgray = gray
        rval, frame = vc.read()





if __name__ == "__main__":
    # Call the laptop camera face/eye detector function above
    camera_go()  
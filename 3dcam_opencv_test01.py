'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
'''

import numpy as np
import cv2

cap = cv2.VideoCapture(1)
block_size = 15
num_disp = 32
while True:
    ret_val, frame = cap.read()
    imgL = frame[:,:1280,:]
    imgR = frame[:,1280:,:]
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    disparity = stereo.compute(imgL_gray,imgR_gray)

    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))


    cv2.imshow("left", imgL)
    cv2.imshow("right", imgR)
    cv2.imshow("disparity", disparity)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
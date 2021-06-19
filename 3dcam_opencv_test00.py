import numpy as np
import cv2 
print(cv2.__version__)

cap = cv2.VideoCapture(1)
window_size = 4
min_disp = 16
num_disp = 64-min_disp
while True:
    ret_val, frame = cap.read()
    imgL = frame[:,:1280,:]
    imgR = frame[:,1280:,:]
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    # stereo = cv2.StereoBM_create()

    # stereo = cv2.StereoBM()
    disparity = stereo.compute(imgL_gray,imgR_gray).astype(np.float32) / 16.0
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]


    cv2.imshow("left", imgL)
    cv2.imshow("right", imgR)
    cv2.imshow("disparity", (disparity-min_disp)/num_disp)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
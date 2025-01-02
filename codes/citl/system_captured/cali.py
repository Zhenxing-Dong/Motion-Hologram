import cv2
import numpy as np
from skimage import io, img_as_ubyte
import skimage.transform as transform


def circle_detect(captured_img, num_circles):
    """
    Detects the circle of a circle board pattern

    :param captured_img: captured image
    :param num_circles: a tuple of integers, (num_circle_x, num_circle_y)
    :return: a tuple, (found_dots, H)
             found_dots: boolean, indicating success of calibration
             H: a 3x3 homography matrix (numpy)
    """

    np.set_printoptions(threshold=np.inf)
    img = captured_img


    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img[...,0].mean())
    # print(img[...,1].mean())
    # print(img[...,2].mean())
    #img = cv2.medianBlur(img, 31)
    img = cv2.medianBlur(img, 55)  # Red 71
    #img = cv2.medianBlur(img, 5)  #210104
    img_gray = img.copy()
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 121, 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 0)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 117, 0)  # Red 127
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = 255 - img


    # Blob detection
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 50
    params.maxThreshold = 200
    # params.minThreshold = 121

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 3000 # 210104

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.7

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.2

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    # Detecting keypoints
    # this is redundant for what comes next, but gives us access to the detected dots for debug
    keypoints = detector.detect(img)
    found_dots, centers = cv2.findCirclesGrid(img, (22, 13),
                                              blobDetector=detector, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

    print(found_dots.shape)
    
    # Drawing the keypoints
    cv2.drawChessboardCorners(captured_img, num_circles, centers, found_dots)
    img_gray = cv2.drawKeypoints(img_gray, keypoints, np.array([]), (0, 0, 255),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Find transformation
    H = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]], dtype=np.float32)

    if found_dots:
        ref_pts = np.array([[ 120.  , 60.],[ 200. ,  60.],[ 280. ,  60.],[ 360.  , 60.],[ 440.   ,60.],[ 520. ,  60.],[ 600. ,  60.],[ 680.  , 60.],[ 760. ,  60.],[ 840. ,  60.],[ 920. ,  60.],[1000. ,  60.],[1080. ,  60.],[1160. ,  60.],[1240.  , 60.],[1320.  , 60.],[1400.  , 60.],[1480. ,  60.],[1560.  , 60.],[1640. ,  60.],[1720. ,  60.],[1800. ,  60.],[ 120. , 140.],[ 200. , 140.],[ 280. , 140.],[ 360. , 140.],[ 440. , 140.],[ 520. , 140.],[ 600. , 140.],[ 680. , 140.],[ 760. , 140.],[ 840. , 140.],[ 920. , 140.],[1000. , 140.],[1080. , 140.],[1160. , 140.],[1240. , 140.],[1320. , 140.],[1400. , 140.],[1480. , 140.],[1560. , 140.],[1640. , 140.],[1720. , 140.],[1800. , 140.],[ 120. , 220.],[ 200. , 220.],[ 280. , 220.],[ 360. , 220.],[ 440. , 220.],[ 520. , 220.],[ 600. , 220.],[ 680. , 220.],[ 760. , 220.],[ 840. , 220.],[ 920. , 220.],[1000. , 220.],[1080. , 220.],[1160. , 220.],[1240. , 220.],[1320. , 220.],[1400. , 220.],[1480. , 220.],[1560. , 220.],[1640. , 220.],[1720. , 220.],[1800. , 220.],[ 120. , 300.],[ 200. , 300.],[ 280. , 300.],[ 360. , 300.],[ 440. , 300.],[ 520. , 300.],[ 600. , 300.],[ 680. , 300.],[ 760. , 300.],[ 840. , 300.],[ 920. , 300.],[1000. , 300.],[1080. , 300.],[1160. , 300.],[1240. , 300.],[1320. , 300.],[1400. , 300.],[1480. , 300.],[1560. , 300.],[1640. , 300.],[1720. , 300.],[1800. , 300.],[ 120. , 380.],[ 200. , 380.],[ 280. , 380.],[ 360. , 380.],[ 440. , 380.],[ 520. , 380.],[ 600. , 380.],[ 680. , 380.],[ 760. , 380.],[ 840. , 380.],[ 920. , 380.],[1000. , 380.],[1080. , 380.],[1160. , 380.],[1240. , 380.],[1320. , 380.],[1400. , 380.],[1480. , 380.],[1560. , 380.],[1640. , 380.],[1720. , 380.],[1800. , 380.],[ 120.  ,460.],[ 200. , 460.],[ 280.,  460.],[ 360. , 460.],[ 440. , 460.],[ 520. , 460.],[ 600. , 460.],[ 680. , 460.],[ 760. , 460.],[ 840. , 460.],[ 920. , 460.],[1000. , 460.],[1080. , 460.],[1160. , 460.],[1240. , 460.],[1320. , 460.],[1400. , 460.],[1480.  ,460.],[1560. , 460.],[1640. , 460.],[1720. , 460.],[1800.  ,460.],[ 120.  ,540.],[ 200.  ,540.],[ 280.  ,540.],[ 360.  ,540.],[ 440.  ,540.],[ 520.  ,540.],[ 600.  ,540.],[ 680.  ,540.],[ 760.  ,540.],[ 840.  ,540.],[ 920.  ,540.],[1000.  ,540.],[1080.  ,540.],[1160.  ,540.],[1240.  ,540.],[1320.  ,540.],[1400.  ,540.],[1480.  ,540.],[1560. , 540.],[1640.  ,540.],[1720. , 540.],[1800. , 540.],[ 120. , 620.],[ 200. , 620.],[ 280. , 620.],[ 360. , 620.],[ 440. , 620.],[ 520. , 620.],[ 600. , 620.],[ 680. , 620.],[ 760. , 620.],[ 840. , 620.],[ 920. , 620.],[1000. , 620.],[1080. , 620.],[1160. , 620.],[1240. , 620.],[1320. , 620.],[1400. , 620.],[1480.  ,620.],[1560. , 620.],[1640. , 620.],[1720. , 620.],[1800. , 620.],[ 120. , 700.],[ 200. , 700.],[ 280. , 700.],[ 360. , 700.],[ 440. , 700.],[ 520. , 700.],[ 600. , 700.],[ 680. , 700.],[ 760. , 700.],[ 840. , 700.],[ 920. , 700.],[1000. , 700.],[1080. , 700.],[1160. , 700.],[1240. , 700.],[1320. , 700.],[1400. , 700.],[1480.  ,700.],[1560.  ,700.],[1640.  ,700.],[1720.  ,700.],[1800.  ,700.],[ 120.  ,780.],[ 200.  ,780.],[ 280.  ,780.],[ 360.  ,780.],[ 440. , 780.],[ 520. , 780.],[ 600. , 780.],[ 680. , 780.],[ 760. , 780.],[ 840. , 780.],[ 920. , 780.],[1000. , 780.],[1080. , 780.],[1160. , 780.],[1240. , 780.],[1320. , 780.],[1400. , 780.],[1480. , 780.],[1560. , 780.],[1640. , 780.],[1720. , 780.],[1800. , 780.],[ 120. , 860.],[ 200. , 860.],[ 280.  ,860.],[ 360. , 860.],[ 440. , 860.],[ 520. , 860.], [ 600. , 860.],[ 680. , 860.],[ 760. , 860.],[ 840. , 860.],[ 920. , 860.],[1000. , 860.],[1080. , 860.],[1160. , 860.],[1240. , 860.],[1320. , 860.],[1400. , 860.],[1480. , 860.],[1560. , 860.],[1640. , 860.],[1720. , 860.],[1800. , 860.],[ 120. , 940.],[ 200. , 940.],[ 280. , 940.],[ 360. , 940.],[ 440. , 940.],[ 520. , 940.],[ 600. , 940.],[ 680. , 940.],[ 760. , 940.],[ 840. , 940.],[ 920. , 940.],[1000. , 940.],[1080. , 940.],[1160. , 940.],[1240. , 940.],[1320.  ,940.],[1400. , 940.],[1480. , 940.],[1560.  ,940.],[1640. , 940.],[1720. , 940.],[1800. , 940.],[ 120., 1020.],[ 200., 1020.],[ 280. ,1020.],[ 360., 1020.],[ 440. ,1020.],[ 520. ,1020.],[ 600. ,1020.],[ 680. ,1020.],[ 760. ,1020.],[ 840. ,1020.],[ 920. ,1020.],[1000. ,1020.],[1080. ,1020.],[1160. ,1020.],[1240. ,1020.],[1320. ,1020.],[1400. ,1020.],[1480. ,1020.],[1560. ,1020.],[1640. ,1020.],[1720. ,1020.],[1800. ,1020.]], dtype=np.float32)

        centers = centers.reshape(num_circles[0] * num_circles[1], 2)

        print(centers.shape)
        print(ref_pts.shape)
        H = transform.estimate_transform('polynomial', ref_pts, centers)       
        
    return found_dots, H


range_y = slice(200, 1436)
range_x = slice(100, 2048)
print('  -- Calibrating ...')
captured_img = cv2.imread('')  
img_masked = np.zeros_like(captured_img)
img_masked[range_y, range_x, ...] = captured_img[range_y, range_x, ...]
found_corners, H,  = circle_detect(img_masked, num_circles=(13,22))

if found_corners:
    print('  -- Calibration succeeded!...')
else:
    raise ValueError('  -- Calibration failed')


# for i in range(1, 801):
#     input_filename = f'{i:04d}.png'
#     output_filename = f'{i:04d}.png'
#     input_dir = '/home/users/user4/Ruichen/citl/captured/'
#     output_dir = '/home/users/user4/Ruichen/citl/cap_c/'
    
#     input_path = os.path.join(input_dir, input_filename)
#     output_path = os.path.join(output_dir, output_filename)
    
#     image = io.imread(input_path)
    
#     if image is None:
#         raise IOError(f"Image not found or cannot be read at {input_path}")
    
#     height, width = image.shape[:2]
    
#     corrected_image = transform.warp(image, H, output_shape=(1080, 1920))
    
#     corrected_image = img_as_ubyte(corrected_image)

#     io.imsave(output_path, corrected_image)
    
#     print(f"Processed and saved: {output_filename}")

# print("All images have been processed and saved.")


image = io.imread('')
corrected_image = transform.warp(image, H, output_shape=(1080, 1920))
corrected_image = img_as_ubyte(corrected_image)
io.imsave('',corrected_image)
print("saved")
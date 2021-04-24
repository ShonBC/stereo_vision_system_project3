"""
Shon Cortes
ENPM 673 - Perception for Autonomous Robots:
Project 3 Stereo Vision System 
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def data1(): # Initializes images from data set 1

    im0 = cv2.imread('data_1/im0.png')
    im0 = cv2.resize(im0, (720,480))
    im1 = cv2.imread('data_1/im1.png')
    im1 = cv2.resize(im1, (720,480))

    cam0 = np.array([[5299.313, 0, 1263.818],
    [0, 5299.313, 977.763],
    [0, 0, 1]])
    cam1 = np.array([[5299.313, 0, 1438.004],
    [0, 5299.313, 977.763],
    [0, 0, 1]])

    return im0, im1, cam0, cam1

def data2(): # Initializes images from data set 2

    im0 = cv2.imread('data_2/im0.png')
    im0 = cv2.resize(im0, (720,480))
    im1 = cv2.imread('data_2/im1.png')
    im1 = cv2.resize(im1, (720,480))

    cam0 = np.array([[4396.869, 0, 1353.072],
    [0, 4396.869, 989.702],
    [0, 0, 1]])
    cam1 = np.array([[4396.869, 0, 1538.86],
    [0, 4396.869, 989.702],
    [0, 0, 1]])

    return im0, im1, cam0, cam1

def data3(): # Initializes images from data set 3

    im0 = cv2.imread('data_3/im0.png')
    im0 = cv2.resize(im0, (720,480))
    im1 = cv2.imread('data_3/im1.png')
    im1 = cv2.resize(im1, (720,480))

    cam0 = np.array([[5806.559, 0, 1429.219],
    [0, 5806.559, 993.403],
    [0, 0, 1]])
    cam1 = np.array([[5806.559, 0, 1543.51],
    [0, 5806.559, 993.403],
    [0, 0, 1]])

    return im0, im1, cam0, cam1

def sift(image): # Use SIFT to find keypoints and descriptors for image and draw key points

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(gray, None)
    image = cv2.drawKeypoints(gray, key_points, image) # Draw Keypoints found   

    return key_points, descriptors

def features(image_1, image_2): # Use keypoints and descriptors form SIFT to match features between two images

    key_points_1, descriptors_1 = sift(image_1)
    key_points_2, descriptors_2 = sift(image_2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    """
    Compare the distance between the discriptors. 
    If the first is less than 75% of the second distance, 
    Add it the matched pair to the "good_match" list.
    """

    good_match = []
    pts_1 = []
    pts_2 = []
    for m, n in matches: # Apply ratio test to filter for good matches

        if m.distance < 0.75 * n.distance:

            good_match.append([m])
            # pts_1.append(key_points_1[m.trainIdx].pt)
            # pts_2.append(key_points_2[n.trainIdx].pt)

    x1, y1 = [], []
    x2, y2 = [], []

    for i in range(len(good_match)): # Collect the good_match coordinates from each image

        x1.append(key_points_1[good_match[i][0].queryIdx].pt[0])
        y1.append(key_points_1[good_match[i][0].queryIdx].pt[1])
        pts_1.append(key_points_1[good_match[i][0].queryIdx].pt)
        
        x2.append(key_points_2[good_match[i][0].trainIdx].pt[0])
        y2.append(key_points_2[good_match[i][0].trainIdx].pt[1])
        pts_2.append(key_points_2[good_match[i][0].trainIdx].pt)

    f = fun_mtx(x1, y1, x2, y2) # Fundamental Matrix
    
    pts_1 = np.int32(pts_1)
    pts_2 = np.int32(pts_2)

    F, mask = cv2.findFundamentalMat(pts_1, pts_2, cv2.FM_RANSAC)
    print("Fundamental Matrix: ", '\n', F)

    # Filter for inlier points
    pts_1 = pts_1[mask.ravel() == 1]
    pts_2 = pts_2[mask.ravel() == 1]
    print(F)

    # Compute homography matrices for image rectification
    # _, h1, h2 = cv2.stereoRectifyUncalibrated(np.int32(pts_1),np.int32(pts_2), f, (720,480))
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts_1, pts_2, F, (720,480))

    print("Homography for first image: ", '\n', H1)
    print("Homography for second image:", '\n', H2)

    # image_3 = cv2.drawMatchesKnn(image_1, key_points_1, image_2, key_points_2, good_match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    E = ess_mtx(F) # Essential Matrix

    return pts_1, pts_2, H1, H2, F

def fun_mtx(x1, y1, x2, y2): # Compute the Fundamental Matrix using matched key point coordinates

    A = []
    
    for i in range(len(x1)):
        
        eq = [x1[i] * x2[i], x1[i] * y1[i], x1[i], y1[i] * x2[i], y1[i] * y2[i], y1[i], x2[i], y2[i], 1]

        A.append(eq)

    u, s, v = np.linalg.svd(A)

    f = np.reshape(v[-1], (3, 3))

    return f

def ess_mtx(fundamental_matrix): # Compute the Essential Matrix from the Fundamental Matrix

    w = np.array([[0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]])

    E = cam1.T * fundamental_matrix * cam0

    # Decompose essential matrix into rotation and translation
    u, s, v = np.linalg.svd(E)
    t1 = u[-1]
    t2 = -u[-1]
    r1 = u * w * v.T
    r2 = u * w.T * v.T

    print("Essential Matrix: ", '\n', E)
    print("Rotation: ", '\n',  r1, '\n', '\n', r2)
    print("Translation: ", '\n', t1, '\n', '\n', t2)

    return E

def rectify(image_1, image_2):

    pts_1, pts_2, H1, H2, F = features(image_1, image_2)
    
    pts_1 = np.int32(pts_1)
    pts_2 = np.int32(pts_2)

    rec1 = cv2.warpPerspective(image_1, H1, (720, 480))
    rec2 = cv2.warpPerspective(image_2, H2, (720, 480))

    F_rec = np.dot(np.dot(np.linalg.inv(H2).T, F), np.linalg.inv(H1))

    lines_1 = cv2.computeCorrespondEpilines(pts_1.reshape(-1, 1, 2), 1, F_rec).reshape(-1, 3)
    lines_2 = cv2.computeCorrespondEpilines(pts_2.reshape(-1, 1, 2), 2, F_rec).reshape(-1, 3)

    draw_epiline(rec1, lines_2)
    draw_epiline(rec2, lines_1)

    h_stack = np.hstack((rec1, rec2))
    cv2.imshow('rectify', h_stack)

    return rec1, rec2

def draw_epiline(image, lines):

    height, width, _ = image.shape

    for i in lines:
        
        x1, y1 = map(int, [0, -i[2] / i[1]])
        x2, y2 = map(int, [width, -(i[2] + i[0] * width) / i[1]])
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

def correspondence(rec1, rec2):

    k = 5

    disp = np.ones_like(rec1)

    for i in range(len(rec1)):
        for j in range(len(rec1[i])):
            
            a = rec1[i - k: i + k, j - k: j + k]

            for c in range(len(rec2[i])):
                
                b = rec2[i - k: i + k, c - k: c + k]

                if a.all() == b.all():
                    
                    disp[i][j] = disparity(c, j)
                
                else:

                    pass

    return disp

def disparity(x1, x2):

    disp = x1 - x2

    return disp
   

def stereo_depth(image_1, image_2): # Compute depth map using OpenCv inbuilt functions
    
    gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 11)
    disp = stereo.compute(gray_1, gray_2)
    heatmap = cv2.applyColorMap(np.uint8(disp), cv2.COLORMAP_HOT)
    plt.imshow(disp,'gray')
    cv2.imshow('Heatmap', heatmap)
    # plt.imshow(heatmap, 'Heatmap')
    plt.show()

if __name__ == "__main__":

    im0, im1, cam0, cam1 = data3() # Choose which data set to apply stereo vision to (Each data set consists of two images)

    rec1, rec2 = rectify(im0, im1)

    stereo_depth(im0, im1)

    # disp = correspondence(rec1, rec2)

    cv2.imshow('img0', im0)
    cv2.imshow('img1', im1)
    # cv2.imshow('disp', disp)
    cv2.waitKey(0)
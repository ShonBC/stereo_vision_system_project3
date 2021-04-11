"""
Shon Cortes
ENPM 673 - Perception for Autonomous Robots:
Project 3 Stereo Vision System 
"""
import cv2
import numpy as np

def data1(): # Initializes images from data set 1

    im0 = cv2.imread('data_1/im0.png')
    im0 = cv2.resize(im0, (720,480))
    im1 = cv2.imread('data_1/im1.png')
    im1 = cv2.resize(im1, (720,480))
    return im0, im1

def data2(): # Initializes images from data set 2

    im0 = cv2.imread('data_2/im0.png')
    im0 = cv2.resize(im0, (720,480))
    im1 = cv2.imread('data_2/im1.png')
    im1 = cv2.resize(im1, (720,480))

    return im0, im1

def data3(): # Initializes images from data set 3

    im0 = cv2.imread('data_3/im0.png')
    im0 = cv2.resize(im0, (720,480))
    im1 = cv2.imread('data_3/im1.png')
    im1 = cv2.resize(im1, (720,480))
    return im0, im1

def sift(image): # Use SIFT to find keypoints and descriptors for image and draw key points

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(gray, None)
    image = cv2.drawKeypoints(gray, key_points, image)    

    return key_points, descriptors

def features(image_1, image_2): # Use keypoints and descriptors form SIFT to match features between two images

    key_points_1, descriptors_1 = sift(image_1)
    key_points_2, descriptors_2 = sift(image_2)


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    good_match = []
    for m, n in matches: # Apply ratio test to filter for good matches

        if m.distance < 0.75 * n.distance:

            good_match.append([m])

    image_3 = cv2.drawMatchesKnn(image_1, key_points_1, image_2, key_points_2, good_match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return image_3

def fun_mtx(key_points_1, key_points_2):

    x1 = []
    y1 = []

    for i in key_points_1:

        x1.append(key_points_1[i].pt[0])
        y1.append(key_points_1[i].pt[1])
            

if __name__ == "__main__":

    im0, im1 = data3()

    img3 = features(im0, im1)

    cv2.imshow('img0', im0)
    cv2.imshow('img1', im1)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)
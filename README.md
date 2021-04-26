# stereo_vision_system_project3
ENPM 673 - Perception For Autonomous Robots: In this project, we are going to implement the concept of Stereo Vision. We will be given 3 different datasets, each of them contains 2 images of the same scenario but taken from two different camera angles. By comparing the information about a scene from 2 vantage points, we can obtain the 3D information by examining the relative positions of objects.


The program has the data sets imported in functions at the top of the program. 
In the main:
    
    Choose which data set to run by changing the line:

        im0, im1, cam0, cam1, baseline, focal_length = data3() # Choose which data set to apply stereo vision to (Each data set consists of two images)

    The user can choose between data1(), data2(), and data3().

    The output will print all the required matrices to the consol and show the Matched points, Rectified images with parallel epipolar lines, 
    Disparity and Depth Maps with their corresponding Heat Maps, and the Disparity heat map found using the inbuilt methods for comparison.  

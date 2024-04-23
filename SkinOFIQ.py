# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:12:45 2024

@author: 1136696
"""
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import math

def imageFiftyPercent(image):
    height, width, _ = image.shape

    # compute axis from init and end to obtain the center of 50%
    init_x = int(width * 0.25)  # 25% of width
    end_x = int(width * 0.75)  # 75% of width
    init_y = int(height * 0.25)  # 25% of height
    end_y = int(height * 0.75)  # 75% 25% of height

    # Resize the image 32 %
    imageFifty = image[init_y:end_y, init_x:end_x]

    return imageFifty


def color_convert(x):
    if x > 0.04045:
        return ((x + 0.055) / 1.055) ** 2.4
    else:
        return x / 12.92

def cubic(x, k, eps):
    if x <= eps:
        return ((k * x) + 16) / 116
    else:
        return math.pow(x, 1/3)

def convert_BGR_to_CIELAB(rgb_image):
    k = 24289 / 27.0
    eps = 216 / 24389.0

    # Split the RGB image into channels
    channels = cv2.split(rgb_image)

    # Calculate mean values for each channel
    R = np.mean(channels[2]) / 255.0
    G = np.mean(channels[1]) / 255.0
    B = np.mean(channels[0]) / 255.0

    # Convert colors to linear RGB
    R_L = color_convert(R)
    G_L = color_convert(G)
    B_L = color_convert(B)

    # Convert from linear RGB to XYZ
    X = R_L * 0.43605 + G_L * 0.38508 + B_L * 0.14309
    Y = R_L * 0.22249 + G_L * 0.71689 + B_L * 0.06062
    Z = R_L * 0.01393 + G_L * 0.09710 + B_L * 0.71419

    # Reference values for D65 illuminant
    X_R = X / 0.964221
    Y_R = Y
    Z_R = Z / 0.825211

    # Apply cubic function to normalize values
    F_X = cubic(X_R, k, eps)
    F_Y = cubic(Y_R, k, eps)
    F_Z = cubic(Z_R, k, eps)

    # Calculate CIELAB values
    a = 500.0 * (F_X - F_Y)
    b = 200.0 * (F_Y - F_Z)

    return a, b

def BGR2CIELAB(image):
    # Convert the input image from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Split the LAB image into its individual channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

# Calculate the mean value of the a* and b* channels
    mean_a = cv2.mean(a_channel)[0]
    mean_b = cv2.mean(b_channel)[0]
    return mean_a,mean_b

def calculateScore(meanChannelA, meanChannelB):
        rawScore = np.sqrt(
            np.power(max(max(0.0, 5 - meanChannelA), max(0.0, meanChannelA - 25)), 2)
            + np.power(max(max(0.0, 5 - meanChannelB), max(0.0, meanChannelB - 35)), 2)
        ) if meanChannelA >= 0 and meanChannelB >= 0 else 100
        return rawScore



def assess_skin(image):
    # Resize the input image to a 32x32-pixel square
        #Define the size of redimention
    width32 = 32
    height32 = 32
        # resized image
    imageResized32 = cv2.resize(imagefifty, (width32, height32))
        #############################################
        # Methodology using OFIQ-Project from NIST    
        #############################################
            
    meanChannelA = 0
    meanChannelB = 0
        # Colocar el rostro recortado
        # meanChannelA, meanChannelB = convert_BGR_to_CIELAB(new_img)
    meanChannelA, meanChannelB = BGR2CIELAB(imageResized32)
    print("meanChannelA",meanChannelA)
    print("meanChannelB",meanChannelB)
    rawScore = calculateScore(meanChannelA, meanChannelB)
    print(rawScore) 
    skin=0
    nonSkin = 0
    for index_i in range(width32):
        for index_j in range(height32):
            if Cb_channel[index_i,index_j]>132 and Cb_channel[index_i,index_j]<174 and Cr_channel[index_i,index_j]>79 and Cr_channel[index_i,index_j]<121:
                skin+=1
            else:
                nonSkin+=1
                        
    print(skin)    
    print(nonSkin)    
        # Compute the histogram of the Y channel values
    hist, _ = np.histogram(y_channel, bins=8, range=(0, 256))
        
        
        # Compute the counts for the brightest and darkest bins
    ends = (2 * hist[0]) + hist[1] + hist[-1]
    last = len(hist) - 1

        # Compute the metric values
    dark_count = max(0.0, 1.0 - float(hist[0]) / 320.0)
    even_count = max(0.0, 1.0 - float(4 * ends) / 1024.0)
    skin_count = min(1.0, float(skin) / 1024.0) #Aquí va el skin
    wash_count = max(0.0, 1.0 - float(hist[last]) / 320.0)

    return {
        'dark': {'count': dark_count},
        'even': {'count': even_count},
        'skin': {'count': skin_count},
        'wash': {'count': wash_count}
    }

##Read the image and use MTCNN to detect the face (Use Crop&Align instead use MTCNN)
pathFileImage = "img_933.jpg"
img = cv2.imread(pathFileImage)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detector = MTCNN()        
faces = detector.detect_faces(img)

##Theorically there will be one Face.
for face in faces:
    x, y, width, height = face['box']
    new_img = img[y:y+height, x:x+width]

    #Calcule the 50% of center image
    imagefifty = imageFiftyPercent(new_img)
    #Define the size of redimention
    width32 = 32
    height32 = 32
    # resized image
    imageResized32 = cv2.resize(imagefifty, (width32, height32))

    metrics = assess_skin(imageResized32)
    print("Metrics:", metrics)
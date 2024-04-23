# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:28:28 2024

@author: 1136696
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


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


#############################################
# Methodology using instructions from Gifford
#############################################

def assess_skin(image):
    width32 = 32
    # Resize the input image to a 32x32-pixel square
    height32 = 32
        # resized image
        # Convert the resized image from RGB to YUV color space using OPENCV
        #yuv_image = cv2.cvtColor(imageResized32, cv2.COLOR_BGR2YUV)
        # Extract the Y channel (luminance) from the YUV image
        #y_channel = yuv_image[:,:,0]
        #u_channel = yuv_image[:,:,1]
        #v_channel = yuv_image[:,:,2]
        #Convert RGB to YUV
    yuv_image = np.zeros((height32, width32, 3), dtype=np.uint8)
    for i in range(height32):
        for j in range(width32):
            R, G, B = image[i, j]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            U = -0.147 * R - 0.289 * G + 0.436 * B
            V = 0.615 * R - 0.515 * G - 0.100 * B
            yuv_image[i, j] = [Y, U, V]
        #Convert YUV to YCbCr
    Y, U, V = cv2.split(yuv_image)
    YCbCr_image = np.zeros_like(yuv_image, dtype=np.uint8)
    YCbCr_image[:,:,0] = Y
    YCbCr_image[:,:,1] = U - 128
    YCbCr_image[:,:,2] = V - 128
        
    y_channel = YCbCr_image[:,:,0]
    Cb_channel = YCbCr_image[:,:,1]
    Cr_channel = YCbCr_image[:,:,2]
        #print("Cb_channel",Cb_channel)
        #print("Cr_channel",Cr_channel)
        #blue_channel, green_channel, red_channel = cv2.split(imageResized32)
    skin = 0
    nonSkin = 0
        #We use https://jips-k.org/q.jips?cp=pp&pn=314 to detect and countskin-color pixels:
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space
    #  132<=Cr<=174 and 79<=Cb<=121 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 79, 132), (255, 121,174)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    pixels = np.array(YCrCb_mask)
    plt.imshow(pixels)
    ax = plt.gca()    
    plt.axis('off')
    plt.show()
    #cv2.imshow("YCrCb_mask",YCrCb_mask)        
    for index_i in range(width32):
        for index_j in range(height32):
            if YCrCb_mask[index_i,index_j]==0:
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
    
    
##Read the image from the Crop&Align of Gifford
pathFileImage = "Lenna_Sq.jpg"
img = cv2.imread(pathFileImage)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##Theorically there will be one Face.
#Calcule the 50% of center image
imagefifty = imageFiftyPercent(img)
#Plot
#pixelsFifty = np.array(imagefifty)
#plt.imshow(pixelsFifty)
#ax = plt.gca()    
#plt.axis('off')
#plt.show()

#Define the size of redimention
width32 = 32
height32 = 32
# resized image
imageResized32 = cv2.resize(imagefifty, (width32, height32))
#Plot
#pixelsResized32 = np.array(imageResized32)
#plt.imshow(pixelsResized32)
#ax = plt.gca()    
#plt.axis('off')
#plt.show()
metrics = assess_skin(imageResized32)
print("Metrics:", metrics) 
    

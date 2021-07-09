# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:14:08 2019

@author: jacqueline.cortez
"""

# Import numpy and matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

print("===================== B E G I N =====================")
# Load the image into an array: img
name_file = '192px-Lunar_surface.jpg'
img = plt.imread(name_file)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")
plt.show()

# Print the shape of the image
print("Shape of imgage '",name_file,"': ", img.shape)

# Flatten 2D array into 1D array
pixels = img.flatten()
minval, maxval = pixels.min(), pixels.max()
print("Shape of pixels: ", pixels.shape)
print("Min pixels: ", minval)
print("Max pixels: ", maxval)

# Showing the evolution of the graph
plt.clf() # Clear the graph space
plt.hist(pixels) 
plt.title("plt.hist(pixels)")
plt.show()

plt.clf() # Clear the graph space
plt.hist(pixels, bins=256, range=(0,256)) 
plt.title("plt.hist(pixels, bins=256, range=(0,255))")
plt.show()

plt.clf() # Clear the graph space
#plt.hist(pixels, bins=256, range=(0,255), normed=True) #normed is deprecated 
plt.hist(pixels, bins=256, range=(0,256), density=True) 
plt.title("plt.hist(pixels, bins=256, range=(0,255), normed=True)")
plt.show()

# Get the histogram of the original images
plt.clf() # Clear the graph space
#plt.hist(pixels, bins=256, range=(0,256), normed=True, color="blue", alpha=0.3)
plt.hist(pixels, bins=256, range=(0,256), density=True, color="blue", alpha=0.3) #normed is deprecated
plt.title("plt.hist(pixels, bins=256, range=(0,255), normed=True, color='blue', alpha=0.3)", color="red")  
plt.show()

# Rescaling the image
rescaled = ((255/(maxval-minval))*(pixels-minval)).astype(int)
print("Min rescaled: ", rescaled.min())
print("Max rescaled: ", rescaled.max())

# Restore into originals dimensions
resc_img = rescaled.reshape(img.shape)
print("Shape of resc_img: ", resc_img.shape)

# Showing the rescale image
plt.clf() # Clear the graph space
plt.imshow(resc_img)
plt.title("Rescale Image")
plt.axis("off")
plt.show()

# Define equality between img and resc_img
#print("img == resc_img -->", (img == resc_img).all())
print("img == resc_img -->", np.array_equal(img, resc_img))

# Getting the histogram of the original and rescale image
plt.clf() # Clear the graph space
plt.hist(pixels, bins=256, range=(0,256), normed=True, color="blue", alpha=0.3)
plt.hist(rescaled, bins=256, range=(0,256), normed=True, color="green", alpha=0.3)
plt.legend(["original","rescaled"])
plt.title("Histogram of the pixel intensitive", color="red")
plt.show()
print("pixels == rescaled -->", np.array_equal(pixels, rescaled))

# Getting the histogram of the original and CDF
plt.clf() # Clear the graph space
plt.hist(pixels, bins=256, range=(0,256), normed=True, color="blue", alpha=0.3)
plt.twinx()
orig_cdf, bins, patches = plt.hist(pixels, cumulative=True, bins=256, range=(0,256), normed=True, color="red", alpha=0.3)
plt.title("Image histogram and CDF", color="red")
plt.show()

# Equalizing intensity values
new_pixels = np.interp(pixels, bins[:-1], orig_cdf*255)
new = new_pixels.reshape(img.shape)
plt.clf() # Clear the graph space
plt.imshow(new)
plt.title("Equalized Image")
plt.axis("off")
plt.show()

# Getting the histogram of the equalized image and CDF
plt.clf() # Clear the graph space
plt.hist(new_pixels, bins=256, range=(0,256), normed=True, color="blue", alpha=0.3)
plt.twinx()
plt.hist(new_pixels, cumulative=True, bins=256, range=(0,256), normed=True, color="red", alpha=0.3)
plt.title("Equalized Image histogram and CDF", color="red")
plt.show()

# Showing the three images
plt.clf() # Clear the graph space
plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(resc_img)
plt.title("Rescale Image")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(new)
plt.title("Equalized Image")
plt.axis("off")
plt.show()
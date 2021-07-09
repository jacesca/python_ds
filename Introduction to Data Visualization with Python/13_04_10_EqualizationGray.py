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
name_file = '640px-Unequalized_Hawkes_Bay_NZ.jpg'
img = plt.imread(name_file)

# Flatten 2D array into 1D array
pixels = img.flatten()
minval, maxval = pixels.min(), pixels.max()

# Rescaling the image
rescaled = ((255/(maxval-minval))*(pixels-minval)).astype(int)

# Restore into originals dimensions
resc_img = rescaled.reshape(img.shape)

# Equalizing intensity values
orig_cdf, bins, patches = plt.hist(pixels, cumulative=True, bins=256, range=(0,255), normed=True, color="red", alpha=0.3)
new_pixels = np.interp(pixels, bins[:-1], orig_cdf*255)
new = new_pixels.reshape(img.shape)

plt.clf() # Clear the graph space
plt.gray() # Run image as gray color for default
# Get the histogram of the original images
plt.subplot(4,1,1)
plt.hist(pixels, bins=256, range=(0,256), normed=True, color="blue", alpha=0.3)
plt.xlim((0,256))
plt.title("Histogram of the original image", color="red")
# Getting the histogram of the original and rescale image
plt.subplot(4,1,2)
plt.hist(pixels, bins=256, range=(0,256), density=True, color="blue", alpha=0.3) #normed is deprecated
plt.hist(rescaled, bins=256, range=(0,256), normed=True, color="green", alpha=0.3)
plt.legend(["original","rescaled"])
plt.xlim((0,256))
plt.title("Histogram of the pixel intensitive", color="red")
# Getting the histogram of the original and CDF
plt.subplot(4,1,3)
plt.hist(pixels, bins=256, range=(0,256), normed=False, color="blue", alpha=0.3)
plt.twinx()
orig_cdf, bins, patches = plt.hist(pixels, cumulative=True, bins=256, range=(0,256), normed=True, color="red", alpha=0.3)
plt.xlim((0,256))
plt.title("Image histogram and CDF", color="red")
# Getting the histogram of the equalized image and CDF
plt.subplot(4,1,4)
plt.hist(new_pixels, bins=256, range=(0,256), normed=False, color="blue", alpha=0.3)
plt.twinx()
plt.xlim((0,256))
plt.hist(new_pixels, cumulative=True, bins=256, range=(0,256), normed=True, color="red", alpha=0.3)
plt.title("Equalized Image histogram and CDF", color="red")
# Improve spacing and display the plot
plt.tight_layout()
plt.show()

# Showing the three images
plt.clf() # Clear the graph space
plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(resc_img)
plt.title("Rescaled Image")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(new)
plt.title("Equalized Image")
plt.axis("off")
plt.show()

# Print the shape
print("Shape of imgage: ", img.shape)
print("Shape of rescaled imgage: ", resc_img.shape)
print("Shape of equalized imgage: ", new.shape)

# Showing images using contour
plt.clf() # Clear the graph space
plt.subplot(1,3,1)
plt.contour(img)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.contour(resc_img)
plt.title("Rescaled Image")
plt.axis("off")
plt.subplot(1,3,3)
plt.contour(new)
plt.title("Equalized Image")
plt.axis("off")
plt.show()

# Showing images using pcolor
plt.clf() # Clear the graph space
plt.subplot(1,3,1)
plt.pcolor(img)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.pcolor(resc_img)
plt.title("Rescaled Image")
plt.axis("off")
plt.subplot(1,3,3)
plt.pcolor(new)
plt.title("Equalized Image")
plt.axis("off")
plt.show()


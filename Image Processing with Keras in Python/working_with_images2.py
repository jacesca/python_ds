# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:05:29 2019

@author: jacqueline.cortez
Source: 
https://iphton.github.io/iphton.github.io/Image-Processing-in-Python-Part-2/
https://github.com/iphton/DIP-In-Python
http://hongtuyet.blogspot.com/2013/06/con-vet-xanh.html #parrot image
https://en.wikipedia.org/wiki/Kernel_(image_processing)#Details
http://setosa.io/ev/image-kernels/
"""
import matplotlib.pyplot as plt                                #For creating charts
import numpy             as np                                 #For making operations in lists
from scipy.ndimage                   import gaussian_filter
from scipy.ndimage                   import median_filter
from scipy.signal                    import convolve2d         #For learning machine - deep learning
from skimage                         import exposure
from skimage                         import measure
from skimage.filters.thresholding    import threshold_otsu
from skimage.filters.thresholding    import threshold_local 
from sklearn.cluster                 import KMeans



print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
tema = "1. Intensity Transformation"; print("** %s\n" % tema)

file='img_parrot.jpg'
pic = plt.imread(file)

#Negative transformation
negative = lambda rgb : 255-rgb
neg_pic = negative(pic)

#Log transformation
grayscale = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
log_transform = lambda gray : (255/np.log(1+np.max(gray))) * np.log(1+gray)

log_pic = log_transform(grayscale(pic))
'''
log transform
-> s = c*log(1+r)

So, we calculate constant c to estimate s
-> c = (L-1)/log(1+|I_max|)

'''
#Gamma transformation - Power Law Transform
"""
A gamma value, G < 1 is sometimes called an encoding gamma, and the process of encoding with this compressive power-law nonlinearity 
is called gamma compression; Gamma values < 1 will shift the image towards the darker end of the spectrum.
Conversely, a gamma value G > 1 is called a decoding gamma and the application of the expansive power-law nonlinearity is called gamma 
expansion. Gamma values > 1 will make the image appear lighter. A gamma value of G = 1 will have no effect on the input image.
"""
GAMMA_VALUE = 2.2 # Gamma < 1 ~ Dark  ;  Gamma > 1 ~ Bright
gamma_correction = lambda rgb: ((rgb/255) ** (1/GAMMA_VALUE)) 
gamma_pic = gamma_correction(pic)


#Plotting the transformations
plt.figure(figsize = (6,4))
plt.subplot(2,2,1)
plt.imshow(pic);
plt.axis('off');
plt.title('Original\nimage')

plt.subplot(2,2,2)
plt.imshow(neg_pic);
plt.axis('off');
plt.title('Negative\ntransformation')

plt.subplot(2,2,3)
plt.imshow(log_pic, cmap = plt.get_cmap(name = 'gray'));
plt.axis('off');
plt.title('Log\ntransformation')

plt.subplot(2,2,4)
plt.imshow(gamma_pic)
plt.axis('off');
plt.title('Gamma\ntransformation')

plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0, right=None, top=0.8, wspace=0.4, hspace=0.4)
plt.show()


print("****************************************************")
tema = "2. Convolution"; print("** %s\n" % tema)

def Convolution(image, kernel):
    """
    Return a convoluted image in 3D
    """
    conv_bucket = []
    for d in range(image.ndim):
        conv_channel = convolve2d(image[:,:,d], kernel,  mode="same", boundary="symm")
        conv_bucket.append(conv_channel)
    return np.stack(conv_bucket, axis=2).astype("uint8")

def plot_transformation(pictures, gray_1st=True):
    """
    Display a multiplot figure
    """
    plt.figure(figsize=(12,5))
    rows=len(pictures); cols=len(pictures[0]);
    for row in range(rows):
        for elem in range(cols):
            #print(rows, cols, (cols*row)+elem+1)
            plt.subplot(rows, cols, (cols*row)+elem+1)
            if (row+elem==0 | gray_1st==False):
                plt.imshow(pictures[row][elem]['picture'])
            else:
                plt.imshow(pictures[row][elem]['picture'], cmap = plt.get_cmap(name = 'gray'))
            plt.title(pictures[row][elem]['title'], fontdict={'fontsize':8})
            plt.axis('off')
    plt.suptitle(tema)
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=0.2, hspace=None)
    plt.show()


def sobel_transformation(pic): 
    """
    The Sobel kernels are used to show only the differences in adjacent pixel values in a particular direction. 
    It tries to approximate the gradients of the image along one direction using kernel functions.
    """
    sobel_pic = []
    for i in range(3):
        sx = convolve2d(pic[:,:,i], rightsobel_kernel  , mode="same", boundary="symm")
        sy = convolve2d(pic[:,:,i], leftsobel_kernel  , mode="same", boundary="symm")
        sobel_pic.append(np.sqrt(sx*sx + sy*sy))
    sobel_pic = np.stack(sobel_pic, axis=2).astype("uint8") 
    return sobel_pic



def scipy_ndimage_filter_3D(img, myfilter='gaussian', mask=np.nan):
    """
    Applies a median filer to all channels
    """
    ims = []
    for d in range(3):
        if myfilter=='gaussian':
            img_conv_d = gaussian_filter(img[:,:,d], sigma = 4)
        elif myfilter=='median':
            img_conv_d = median_filter(img[:,:,d], size=(mask,mask))
        ims.append(img_conv_d)
    return np.stack(ims, axis=2).astype("uint8")


#Working with differente kernel sizes
kernel_sizes = [9,15,30,60]

fig, axs = plt.subplots(nrows = 1, ncols = len(kernel_sizes), figsize=(12,5))
for k, ax in zip(kernel_sizes, axs):
    kernel = np.ones((k,k))
    kernel /= np.sum(kernel) #boxblur_kernel
    ax.imshow(Convolution(pic, kernel));
    ax.set_title("Convolved By Kernel: {}".format(k));
    ax.set_axis_off();

plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0, right=None, top=0.8, wspace=0.4, hspace=None)
plt.show()


#Defining kernels to use
identity_kernel    = [[0,0,0],[0,1,0],[0,0,0]]
edge1_kernel       = [[1,0,-1],[0,0,0],[-1,0,1]]
edge2_kernel       = [[0,1,0],[1,-4,1],[0,1,0]]
edge3_kernel       = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
outline_kernel     = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
sharpen_kernel     = [[0,-1,0],[-1,5,-1],[0,-1,0]]
boxblur_kernel     = np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0
gaussian_kernel    = np.array([[1,2,1],[2,4,2],[1,2,1]])/16

blur_kernel        = [[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]]
emboss_kernel      = [[-2,-1,0],[-1,1,1],[0,1,2]]
bottomsobel_kernel = [[-1,-2,-1],[0,0,0],[1,2,1]]
leftsobel_kernel   = [[1,0,-1],[2,0,-2],[1,0,-1]]
rightsobel_kernel  = [[-1,0,1],[-2,0,2],[-1,0,1]]
topsobel_kernel    = [[1,2,1],[0,0,0],[-1,-2,-1]]

file='img_crazycat.jpg'
pic = plt.imread(file)
plt.figure()
plt.imshow(pic)
plt.title('Original Image: img_crazycat.jpg')
plt.suptitle(tema)
#plt.subplots_adjust(left=None, bottom=0, right=None, top=0.8, wspace=0.4, hspace=0.4)
plt.show()


pictures =[[{'title':'Identity kernel', 'picture': Convolution(pic, identity_kernel)},
            {'title':'Edge1 kernel'   , 'picture': Convolution(pic, edge1_kernel)},
            {'title':'Edge2 kernel'   , 'picture': Convolution(pic, edge2_kernel)},
            {'title':'Edge3 kernel'   , 'picture': Convolution(pic, edge3_kernel)}],[
            {'title':'Outline kernel' , 'picture': Convolution(pic, outline_kernel)},
            {'title':'Sharpen kernel' , 'picture': Convolution(pic, sharpen_kernel)},
            {'title':'Box Blur kernel', 'picture': Convolution(pic, boxblur_kernel)},
            {'title':'Gaussian kernel', 'picture': Convolution(pic, gaussian_kernel)}]]
plot_transformation(pictures, gray_1st=False)


pictures =[[{'title':'Blur kernel'        , 'picture': Convolution(pic, blur_kernel)},
            {'title':'Emboss kernel'      , 'picture': Convolution(pic, emboss_kernel)},
            {'title':'Bottom Sobel kernel', 'picture': Convolution(pic, bottomsobel_kernel)}],[
            {'title':'Left Sobel  kernel' , 'picture': Convolution(pic, outline_kernel)},
            {'title':'Right Sobel  kernel', 'picture': Convolution(pic, sharpen_kernel)},
            {'title':'Top Sobel  kernel'  , 'picture': Convolution(pic, gaussian_kernel)}]]
plot_transformation(pictures, gray_1st=False)


#Applying oultine kernel
gray_pic = grayscale(pic)
edges_pic = convolve2d(gray_pic, outline_kernel, mode = 'valid') # we use 'valid' which means we do not add zero padding to our image
equalized_pic = exposure.equalize_adapthist(edges_pic/np.max(np.abs(edges_pic)), clip_limit = 0.03) # Adjust the contrast of the filtered image by applying Histogram Equalization

pictures =[[{'title':'Image: img_crazycat.jpg', 'picture': pic},
            {'title':'Grayscale'              , 'picture': gray_pic},
            {'title':'Edges'                  , 'picture': edges_pic},
            {'title':'Equalized'              , 'picture': equalized_pic}]]
plot_transformation(pictures)


#The Sharpen Kernel emphasizes differences in adjacent pixel values. This makes the image look more vivid.
sharpen_pic = convolve2d(gray_pic, sharpen_kernel, mode = 'valid') # apply sharpen filter to the original image
edges_pic = convolve2d(sharpen_pic, outline_kernel, mode = 'valid') # apply edge kernel to the output of the sharpen kernel
boxblur_pic = convolve2d(edges_pic, boxblur_kernel, mode = 'valid') # apply normalize box blur filter to the edge detection filtered image
equalized_pic = exposure.equalize_adapthist(boxblur_pic/np.max(np.abs(boxblur_pic)), clip_limit=0.03) # Adjust the contrast of the filtered image by applying Histogram Equalization

pictures =[[{'title':'Image: img_crazycat.jpg', 'picture': pic},
            {'title':'Grayscale'              , 'picture': gray_pic},
            {'title':'Sharpen'                , 'picture': sharpen_pic}],[
            {'title':'Edges'                  , 'picture': edges_pic},
            {'title':'Box Blur'               , 'picture': boxblur_pic},
            {'title':'Equalized'              , 'picture': equalized_pic}]]
plot_transformation(pictures)


#The Gaussian window 
file='img_parrot.jpg'
pic = plt.imread(file)
gray_pic = grayscale(pic)
gaussian_pic = convolve2d(gray_pic, gaussian_kernel, mode = 'valid') # we use 'valid' which means we do not add zero padding to our image
equalized_pic = exposure.equalize_adapthist(gaussian_pic/np.max(np.abs(gaussian_pic)), clip_limit = 0.03) # Adjust the contrast of the filtered image by applying Histogram Equalization

pictures =[[{'title':'Image: img_parrot.jpg', 'picture': pic},
            {'title':'Grayscale'            , 'picture': gray_pic},
            {'title':'Gaussian'             , 'picture': gaussian_pic},
            {'title':'Equalized'            , 'picture': equalized_pic}]]
plot_transformation(pictures)


#The Sobel kernels are used to show only the differences in adjacent pixel values in a particular direction. 
#It tries to approximate the gradients of the image along one direction using kernel functions.
pictures =[[{'title':'Image: img_parrot.jpg', 'picture': pic},
            {'title':'Sobel kernels'        , 'picture': sobel_transformation(pic)}]]
plot_transformation(pictures)


#To reduce noise. we generally use a filter like the Gaussian Filter, which is a digital filtering technique that is often 
#used to remove noise from an image. By combining Gaussian filtering and gradient finding operations together, we can generate 
#some strange patterns that resemble the original image and being distorted in interesting ways.
gaussian_pic = scipy_ndimage_filter_3D(pic)
pictures =[[{'title':'Image: img_parrot.jpg', 'picture': pic},
            {'title':'Gaussain filter'      , 'picture': gaussian_pic},
            {'title':'Sobel kernels'        , 'picture': sobel_transformation(gaussian_pic)}]]
plot_transformation(pictures)


#Now, let’s see using a Median filter to see what sort of effect it can make on the image.
median_pic = scipy_ndimage_filter_3D(pic, myfilter='median', mask=14)
pictures =[[{'title':'Image: img_parrot.jpg', 'picture': pic},
            {'title':'Median filter'      , 'picture': median_pic},
            {'title':'Sobel kernels'        , 'picture': sobel_transformation(median_pic)}]]
plot_transformation(pictures)

print("****************************************************")
tema = "3. Thresholding"; print("** %s\n" % tema)
#Thresholding is a very basic operation in image processing. 
#Converting a greyscale image to monochrome is a common image processing task.
#The algorithm assumes that the image is composed of two basic classes: Foreground and Background. 
#It then computes an optimal threshold value that minimizes the weighted within class variances of these two classes.

def otsu_threshold(img):
    # Compute histogram and probabilities of each intensity level
    pixel_counts = [np.sum(img == i) for i in range(256)]
    # Initialization
    s_max = np.array([0.0,0.0])
    for threshold in range(256):
        # update
        w_0 = sum(pixel_counts[:threshold])
        w_1 = sum(pixel_counts[threshold:])
        mu_0 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / w_0 if w_0 > 0 else 0       
        mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0
        # calculate - inter class variance
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2
        if s > s_max[1]:
            s_max = [threshold, s]
    return s_max[0]


file='img_potatoe.jpg'; 
pic = plt.imread(file)
gray_pic = grayscale(pic)

global_thresh = threshold_otsu(gray_pic)
binary_global_pic = gray_pic < global_thresh

histo_param = otsu_threshold(gray_pic)
block_size = 35
binary_adaptive_pic = threshold_local(gray_pic, block_size, offset=10, param=histo_param)

pictures =[[{'title':'Image: img_potatoe.jpeg'     , 'picture': pic},
            {'title':'Grayscale'                   , 'picture': gray_pic},
            {'title':'Thresholding Binary Global'  , 'picture': binary_global_pic},
            {'title':'Thresholding Binary Adaptive', 'picture': binary_adaptive_pic}]]
plot_transformation(pictures)




print("****************************************************")
tema = "4. Vectorization"; print("** %s\n" % tema)

file='img_parrot.jpg'
pic = plt.imread(file)
new_pic = pic.copy()

h, w = new_pic.shape[:2]
im_small_long = new_pic.reshape((h * w, 3))
im_small_wide = im_small_long.reshape((h, w, 3))
km = KMeans(n_clusters=2)
km.fit(im_small_long)
seg = np.asarray([(1 if i == 1 else 0) for i in km.labels_]).reshape((h,w))
contours = measure.find_contours(seg, 0.5, fully_connected="high")
simplified_contours = [measure.approximate_polygon(c, tolerance=5) for c in contours]

plt.figure(figsize=(6,4))
plt.subplot(1,2,1)
plt.imshow(pic);
#plt.axis('off');
plt.title('Original image')

plt.subplot(1,2,2)
for n, contour in enumerate(simplified_contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Contour Tracking')

plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.8, wspace=0.4, hspace=0.4)
plt.show()


print("****************************************************")
print("** END                                            **")
print("****************************************************")
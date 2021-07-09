# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:50:26 2019

@author: jacqueline.cortez
Source: 
https://iphton.github.io/iphton.github.io/Image-Processing-in-Python-Part-1/#1-bullet
https://github.com/iphton/DIP-In-Python
"""
import matplotlib.pyplot as plt                                               #For creating charts
import numpy             as np                                                #For making operations in lists
import random                                                                 #For generating random numbers

print("****************************************************")
print("** BEGIN                                          **")
print("****************************************************")
tema = "1. Importing images and observe it’s properties"; print("** %s\n" % tema)

file='img_squirel.jpg'
pic = plt.imread(file)

title = 'Name of the file: "{}"'.format(file) + \
        '\nShape of the image: {}'.format(pic.shape) + \
        '\nImage high: {}, Image width: {}, Image dimension: {}'.format(pic.shape[0], pic.shape[1], pic.ndim) + \
        '\nImage size: {0:,.2f} MB'.format(pic.size/1024/1024)+ \
        '\nMaximum RGB value: {}, Minimum RGB value: {}'.format(pic.max(), pic.min())

plt.figure(figsize=(6,5))
plt.imshow(pic)
#plt.gca().set_aspect('equal', adjustable='box')
plt.title(title, loc='left')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=None, hspace=None)
plt.show()


# A specific pixel located at Row : 100 ; Column : 50 
# Each channel's value of it, gradually R , G , B
pixel = pic[100, 50]
print("One pixel at position (100, 50): ", pixel)
print('Value of only R channel {}'.format(pic[ 100, 50, 0]))
print('Value of only G channel {}'.format(pic[ 100, 50, 1]))
print('Value of only B channel {}'.format(pic[ 100, 50, 2]))


#####################################################################
print("****************************************************")
tema = "2. Printing each channel alone"; print("** %s\n" % tema)
#####################################################################

plt.figure()
plt.subplot(2,2,1)
plt.imshow(pic[:,:,0])
#plt.ylabel('Height {}'.format(pic.shape[0]))
#plt.xlabel('Width {}'.format(pic.shape[1]))
#plt.gca().set_aspect('equal', adjustable='box')
plt.title('R channel')

plt.subplot(2,2,2)
plt.imshow(pic[:,:,1])
#plt.ylabel('Height {}'.format(pic.shape[0]))
#plt.xlabel('Width {}'.format(pic.shape[1]))
#plt.gca().set_aspect('equal', adjustable='box')
plt.title('G channel')

plt.subplot(2,2,3)
plt.imshow(pic[:,:,2])
#plt.ylabel('Height {}'.format(pic.shape[0]))
#plt.xlabel('Width {}'.format(pic.shape[1]))
#plt.gca().set_aspect('equal', adjustable='box')
plt.title('B channel')

## Intensity on images
intense = pic.copy()
intense[50:100 , : , 0] = 255 # full intensity to those pixel's R channel
intense[200:250 , : , 1] = 255 # full intensity to those pixel's G channel
intense[350:400 , : , 2] = 255 # full intensity to those pixel's B channel
intense[: , 100:150 , 0] = 0 # set value 200 of R channel to those pixels 
intense[: , 400:450 , 1] = 0 # set value 200 of G channel to those pixels 
intense[: , 700:750 , 2] = 0 # set value 200 of B channel to those pixels 

plt.subplot(2,2,4)
plt.imshow(intense)
#plt.gca().set_aspect('equal', adjustable='box')
plt.title('Intense Image')

plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=0.25, hspace=0.25)
plt.show()


#####################################################################
print("****************************************************")
tema = "3. Splitting Layers"; print("** %s\n" % tema)
#####################################################################

fig, ax = plt.subplots(nrows = 1, ncols=3, figsize=(9,5))

for c, ax in zip(range(3), ax): 
    split_img = np.zeros(pic.shape, dtype="int") # create zero matrix
    split_img[ :, :, c] = pic[ :, :, c] # assing each channel 
    ax.imshow(split_img) # display each channel
    ax.set_title(['R','G','B'][c]+' channel')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
plt.show()
    


#####################################################################
print("****************************************************")
tema = "4. Greyscale"; print("** %s\n" % tema)
#####################################################################

grayscale = lambda rgb : np.dot(rgb[... , :3], [0.299 , 0.587, 0.114])
luminosity = lambda rgb : np.dot(rgb[... , :3], [0.21 , 0.72, 0.07])

fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(8,5))
capa = [('Grayscale', grayscale), ('Luminosity', luminosity)]

for c, ax in zip(capa, ax): 
    new_pic = c[1](pic)
    ax.imshow(new_pic, cmap = plt.get_cmap(name = 'gray'))
    ax.set_title(c[0])
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
plt.show()

low_pixel = pic < 20


#####################################################################
print("****************************************************")
tema = "5. Use logical Operator To Process Pixel Values"; print("** %s\n" % tema)
#####################################################################

file='img_red_violet.jpg'
pic = plt.imread(file)

## Filtering an image
low_pixel = pic < 20
new_pic = pic.copy()
new_pic[low_pixel] = random.randint(25,225) # set value randomly range from 25 to 225 - these value also randomly choosen


title = 'Name of the file: "{0}"\nShape of the image: {1}\nImage size: {2:,.2f} MB'.format(file, pic.shape, pic.size/1024/1024)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(pic)
#plt.gca().set_aspect('equal', adjustable='box')
plt.title(title)

plt.subplot(1, 2, 2)
plt.imshow(new_pic)
#plt.gca().set_aspect('equal', adjustable='box')
plt.title('Filtered image')

plt.suptitle("Use logical Operator To Process Pixel Values")
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=None, hspace=None)
plt.show()


#####################################################################
print("****************************************************")
tema = "6. Masking"; print("** %s\n" % tema)
#####################################################################
def masking_img(img):
    """
    Image masking is an image processing technique that is used to remove the background from which photographs those have fuzzy edges, 
    transparent or hair portions.
    """
    # seperate the row and column values
    total_row , total_col , layers = img.shape
    
    '''
    Create vector. Ogrid is a compact method of creating a multidimensional-ndarray operations in single lines. for ex:
    >>> np.ogrid[0:5,0:5]
    output: [array([[0],
                    [1],
                    [2],
                    [3],
                    [4]]), 
            array([[0, 1, 2, 3, 4]])]     
    '''
    x , y = np.ogrid[:total_row , :total_col]

    # get the center values of the image
    cen_x , cen_y = total_row/2 , total_col/2
        
    '''
    Measure distance value from center to each border pixel. To make it easy, we can think it's like, we draw a line from center-
    to each edge pixel value --> s**2 = (Y-y)**2 + (X-x)**2 
    '''
    distance_from_the_center = np.sqrt((x - cen_x)**2 + (y - cen_y)**2)

    # Select convenient radius value
    radius = (total_row/2)

    # Using logical operator '>' 
    '''
    logical operator to do this task which will return as a value 
    of True for all the index according to the given condition
    '''
    circular_pic = distance_from_the_center > radius

    '''
    let assign value zero for all pixel value that outside the cirular disc.
    All the pixel value outside the circular disc, will be black now.
    '''
    new_img = img.copy()
    new_img[circular_pic] = 0
    return new_img


title = 'Name of the file: "{0}"\nShape of the image: {1}\nImage size: {2:,.2f} MB'.format(file, pic.shape, pic.size/1024/1024)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(pic)
#plt.gca().set_aspect('equal', adjustable='box')
plt.title(title)

new_pic = masking_img(pic)

plt.subplot(1, 2, 2)
plt.imshow(new_pic)
#plt.gca().set_aspect('equal', adjustable='box')
plt.title('Masking image')

plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=None, hspace=None)
plt.show()


#####################################################################
print("****************************************************")
tema = "7. Satellite Image Processing"; print("** %s\n" % tema)## 
#####################################################################

file='img_satelite.jpg'
pic = plt.imread(file)

title = 'Name of the file: "{0}"\nShape of the image: {1}\nImage size: {2:,.2f} MB'.format(file, pic.shape, pic.size/1024/1024)

plt.figure(figsize=(6,5))
plt.imshow(pic)
#plt.gca().set_aspect('equal', adjustable='box')
plt.title(title, loc='left')
plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=None, hspace=None)
plt.show()

print(f'Shape of the image {pic.shape}')
print(f'hieght {pic.shape[0]} pixels')
print(f'width {pic.shape[1]} pixels')

"""
Red pixel indicates: Altitude
Blue pixel indicates: Aspect
Green pixel indicates: Slope
"""


#####################################################################
print("****************************************************")
tema = "8. Detecting High Pixel of Each Channel"; print("** %s\n" % tema)## 
#####################################################################

pic_red               = pic.copy()
pic_green             = pic.copy()
pic_blue              = pic.copy()
pic_final             = pic.copy()

red_mask              = pic[:, :, 0] < 180
green_mask            = pic[:, :, 1] < 180
blue_mask             = pic[:, :, 2] < 180
final_mask            = np.logical_and(red_mask, green_mask, blue_mask)

pic_red[red_mask]     = 0
pic_green[green_mask] = 0
pic_blue[blue_mask]   = 0
pic_final[final_mask] = 40

plt.figure() #figsize=(6,5)
plt.subplot(2,2,1)
plt.imshow(pic_red)
plt.title('Red Mask')

plt.subplot(2,2,2)
plt.imshow(pic_green)
plt.title('Green Mask')

plt.subplot(2,2,3)
plt.imshow(pic_blue)
plt.title('Blue Mask')

plt.subplot(2,2,4)
plt.imshow(pic_final)
plt.title('Final Mask')

plt.suptitle(tema)
plt.subplots_adjust(left=None, bottom=0, right=None, top=None, wspace=0.25, hspace=None)
plt.show()


print("****************************************************")
print("** END                                            **")
print("****************************************************")
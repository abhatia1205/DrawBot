# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# Useful Helper Functions

# %% [code]
def get_random_image(parent_file = '/kaggle/input/human-faces/Humans'):
    image_dir = os.path.join(parent_file, random.choice(os.listdir(parent_file)))
    img = np.array(Image.open(image_dir).convert('L'))
    print(img.shape)
    imgplot = plt.imshow(img, cmap = 'gray')
    return img
    
    
    

# %% [code]
get_random_image()

# %% [code]
from scipy import ndimage
import math
import itertools
#slow af
import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return magnitude_spectrum

def convolve(img, kernel):
    kernel_size = len(kernel)//2
    x, y = img.shape
    new_array = np.zeros(img.shape)
    for i in range(x):
        for j in range(y):
            x_array = [k+i for k in range(-kernel_size, kernel_size+1) if (k+i >= 0 and k+i < x)]
            y_array = [k+j for k in range(-kernel_size, kernel_size+1) if (k+j >= 0 and k+j < y)]
            avg = 0
            for a in x_array:
                for b in y_array:
                    avg += img[a][b]*kernel[a-i][b-j]
            new_array[i][j] = avg
    return new_array

def get_contrast_histogram(image):
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0.0, 255])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()

def plot_series(series):
    plt.figure()
    plt.title("Series Plot")
    plt.xlabel("series location")
    plt.ylabel("intensity")

    plt.plot(range(len(series)), series)  # <- or here
    plt.show()


def gaussianBlurring(img, filter_size = False, sigma = False):
    img_x, img_y = img.shape
    if(not filter_size):
        #Arbitrary model to determine filttter size
        filter_size = math.sqrt(max(img_x, img_y))/2.5
    #Round up to nearest odd. Size will be 2*size + 1
    size = filter_size//2
    if(not sigma):
        sigma = filter_size
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-1*((x**2 + y**2) / (2.0*sigma**2))) * normal
    print(g)
    print(sum(g.flatten()))
    new_img = ndimage.filters.convolve(img, g)
    plt.imshow(new_img)
    return new_img
    
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    #returns in radians
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

def suppression(intensity_img, direction_img, negative= True):
    x_size, y_size = intensity_img.shape
    new_intensity = np.zeros(intensity_img.shape)
    fourthpi = 0.785398163
    #iterate over intensity image
    for i in range(1, x_size-1):
        for j in range(1, y_size-1):
            # Check gradient direction
            # round graient direction to nearest pi/4
            # if negative, get both neighbors, else only get one
            # check neighbors in said direction
            # if largest, keep, else, surpress
            direction = round(direction_img[i][j]/fourthpi)
            dictionary = {0: (1,0),
                         1: (1,1),
                         2: (0,1),
                         3: (-1, 1),
                         4: (-1, 0),
                         5: (-1, -1),
                         6: (0, -1),
                         7: (1, -1)}
            if(negative):
                x, y = dictionary[direction]
                neighbors = [(x,y), (-1*x, -1*y)]
            else:
                neighbors = [(x,y)]
            
            max_index = (0,0)
            for a in neighbors:
                x, y = a
                if (intensity_img[i+x][j+y] > intensity_img[i][j]):
                    max_index = a
            if(max_index == (0,0)):
                new_intensity[i][j] = intensity_img[i][j]
    
    return new_intensity

"""def auto_thresholded(img):
    img_x, img_y = img.shape
    new_img = np.zeros(img.shape)
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    for i in range(img_x):
        for j in range(img_y):"""
            
            

def auto_canny(img, filter_size=5, sigma=5):
    """
    Gaussian Blurring
    Sobel Filter Calculations
    Non-maximal Suprresion
    Auto Thresholding
    """
    blurred_img = gaussianBlurring(img, filter_size, sigma)
    norm_img, theta_img = sobel(blurred_img)
    suppresed_img = suppresion(norm_img, theta_img)
    return auto_thresholded(suppresed_img)
    

# %% [code]
img = get_random_image()
img = gaussianBlurring(img)
plt.imshow(img, cmap='gray')

# %% [code]
norm_img, theta_img = sobel_filters(img)
suppressed = suppression(norm_img, theta_img, negative= True)
plt.imshow(suppressed, cmap='gray')

# %% [code]
plt.imshow(np.uint8(norm_img), cmap='gray')

# %% [code]
get_contrast_histogram(suppressed)

# %% [code]
get_contrast_histogram(norm_img)

# %% [code]
import cv2

img = get_random_image()
print(img)
plt.figure()
plt.imshow(img, cmap='gray')
img = gaussianBlurring(img)
print(img)
norm_img, theta_img = sobel_filters(img)
print(img)
suppressed = suppression(norm_img, theta_img, negative= True)
print(img)
plt.figure()
plt.imshow(suppressed, cmap='gray')
sigma = 0.33
v = np.median(img)
print(img)
print(v)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edged = cv2.Canny(img, lower, upper)
plt.figure()
plt.imshow(edged, cmap='gray')

# %% [code]
import skimage.measure    
gray = get_random_image()
img_x, img_y = gray.shape
filter_size = int(math.sqrt(min(img_x, img_y))/2.5)
k_size = (filter_size//4)*2 + 1

from skimage.filters.rank import entropy
from skimage.morphology import disk

entr_img = entropy(gray, disk(10))
plt.figure()
plt.imshow(entr_img)
get_contrast_histogram(entr_img)

img = cv2.GaussianBlur(gray, (5, 5), 7)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
sigma = 0.3
v = np.median(img)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edged = cv2.Canny(img, lower, upper)
plt.figure()
plt.imshow(edged, cmap='gray')


# %% [markdown]
# # Trying Image to line

# %% [code]
def canny(img, filters = 5, sigma = 7, threshold = 0.3):
    img = cv2.GaussianBlur(img, (filters, filters), sigma)
    sigma = threshold
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    plt.figure()
    plt.imshow(edged, cmap='gray')
    return edged

    

# %% [code]
img = get_random_image()
canny_img = canny(img)

# %% [code]
from scipy import stats
import numpy as np
import pylab as pl
from matplotlib import collections  as mc

def canny_to_lines(img, r2_threshold = 0.95, neighbor_radius = 2):
    #takes in canny edge detected image, spits out sequence of lines to draw
    #input is binary image
    """
    running_coordinates = []
    img --> img
    duplicate_img --> img
    pixelstack =[]
    r2_queue = []
    for pixel in row:
        if pixel is white and not seen:
            pixel_stack.add(pixel)
            while(pixel_stack.pop has neighbors that havent been seen and r2(r2_queue) > r2_threshold):
                r2_queue.add(pixel)
                pixel_stack.add(unseen_neighbors)
            running_coordinates.add(r2_queue[0], r2_queue[-1])
            reset queues
                
                
    
    """
    img_x, img_y = img.shape
    running_coordinates = []
    seen = set()
    nradius = neighbor_radius
    #iterate over rows
    def unseenNeighbors(tup):
        tup_x, tup_y = tup
        x = range(-1*nradius,nradius+1)
        y = range(-1*nradius,nradius+1)
        neighbors = []
        for i in x:
            for j in y:
                new_tup_x, new_tup_y = (tup_x + i, tup_y + j)
                if ( new_tup_x < img_x and new_tup_x > -1 and
                   new_tup_y < img_y and new_tup_y > -1 and
                   not( (new_tup_x, new_tup_y) in seen) and
                   img[new_tup_x][new_tup_y] > 0):
                    neighbors.append( (new_tup_x, new_tup_y) )
        return neighbors
                            
    def r2(l):
        if(len(l) == 0 or len(l) == 1):
            return 1.0
        x = [i[0] for i in l]
        y = [i[1] for i in l]
        return stats.linregress(x,y)[2] # third tuple value is r^2
    
    for i in range(img_x):
        for j in range(img_y):
            pixel_stack = []
            r2_queue = []
            #if pixel is white and unseen
            if(img[i][j] > 0 and not((i,j) in seen)):
                pixel_stack.append((i,j))
                while( len(pixel_stack) > 0  and len(unseenNeighbors(pixel_stack[-1])) > 0 and r2(r2_queue) > r2_threshold):
                    tup = pixel_stack.pop()
                    r2_queue.append(tup)
                    seen.add(tup)
                    pixel_stack += unseenNeighbors(tup)
                if(len(r2_queue) > 0):
                    running_coordinates.append([r2_queue[0], r2_queue[-1]])
    
    lc = mc.LineCollection(running_coordinates,linewidths=2)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    
    return running_coordinates
                
                    
    

# %% [code]
img = get_random_image()
canny_img = canny(img)


# %% [code]
l = canny_to_lines(canny_img, r2_threshold = 0.8, neighbor_radius = 5)

# %% [code]
len(l)

# %% [code]

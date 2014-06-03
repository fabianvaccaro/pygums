#Basic functions for batch.py
#AoI - > Area of Interest - > The area of the image corresponding to the chewing gum

#Includes
from skimage.filter import canny
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filter import sobel
from skimage.morphology import watershed
from skimage import io, color
from skimage.filter.rank import entropy, median, mean, mean_bilateral
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from numpy import array
from skimage import exposure, img_as_uint, img_as_ubyte, feature, util


#Function to generate file-name of image from sample number <III>, chewing cycles <CC>, side <SS>, and file extension <EXT>
def sample_fname(III, CC, SS, EXT):
	nsample = "%03d" % III
	ncycle =  str(CC)
	#Uncomment the following line if the chewing cycles <CC> part is formatted with zeros (e.g.: 01, 02 ...09, 10, 11... etc.)
	#ncycle = "%02d" % CC
	result = nsample + '-' + ncycle + '-' + SS + EXT
	return result

	
#Function to load an image from the database, returns RGB
#	Parameters:
#		filename		- >	File-name of the image
#		database_path 	- >	Relative PATH to the image database
#	Returns:
#		rgb				- > Image in RGB
def load_image(filename, database_path):
	rgb = Image.open(database_path + filename)	#Load image from database
	return rgb	#Return RGB image

	
#Function to generate the segmentation mask, returns binary matrix
#	Parameters:
#		rgb		- >		Original image in RGB
#		SMT		- >		Segmentation Marker Threshold used for the watershed algorithm
#	Returns
#		segmentation_mask	- >	Binary 2D array with the same dimensions of the original image, 1s represent Area of Interest (AoI)
def smask(rgb, SMT):
	rgb_eq = exposure.equalize_hist(rgb, nbins=256)	#RGB image with equalized histogram
	lab = color.rgb2lab(rgb_eq)	#Transformation from RGB to Lab color model
	im = np.asarray(lab)	#Express lab as a numpy array
	bchannel = im[:,:,2]	#Selection of the b channel
	bchannel = np.asarray(bchannel)	#Express bchannel as a numpy array
	bchannel.flags.writeable = True	#Set bchannel writable flag as True
	markers = np.zeros_like(bchannel)	#Creates a 2D array with the same dimensions as bchannel
	markers[bchannel < SMT] = 1		#Sets markers pixels as 1 where the value of corresponding position in bchannel is lower than SMT
	markers[bchannel > SMT] = 2		#Sets markers pixels as 2 where the value of corresponding position in bchannel is higher than SMT
	elevation_map = sobel(bchannel)	#Creates an elevation map for the watershed algorithm
	segmentation = watershed(elevation_map, markers)	#Segmentation using watershed algorithm
	segmentation_mask = ndimage.binary_fill_holes(segmentation - 1) #Reduce the number of holes in the segmentation mask
	return segmentation_mask	#Returns the segmentation mask
	
def HSVanalysis(rgb, segmentation_mask):
	hsv = color.rgb2hsv(rgb)	#Transformation from RGB to HSV color model
	hchannel = np.copy(hsv[:,:,0])	#Creates a copy of the H channel
	hchannel_interest = hchannel[segmentation_mask != 0] #Extracts the values of the pixels of hchannel that are inside the AoI, discards the rest
	hchannel_interest_std = np.std(hchannel_interest)	#Calculates the Standard Deviation of the values of the pixels in the AoI ot the H channel
	hchannel_interest_hist = np.histogram(hchannel_interest, bins = 200)	#Computes the histogram of hchannel_interest for 200 bins, returns array of the form [(values),(bins)]
	hchannel_interest_hist_max = np.float64(np.amax(hchannel_interest_hist[0])) #Extracts the maximum value of the hchannel_interest_hist
	h_hist0_tmp = np.copy(hchannel_interest_hist[0])	#Creates a temporal copy of the hchannel_interest_hist values
	hchannel_interest_hist_normed = h_hist0_tmp / hchannel_interest_hist_max	#Generates the normalized histogram of the AoI of the H channel
	hchannel_interest_hist_normed_std = np.std(hchannel_interest_hist_normed)	#Calculates the Standard Deviation of the normalized histogram of the AoI of the H channel
	results = (hchannel_interest_std, hchannel_interest_hist_normed_std)	#List features: (STD of the values of the pixels , STD of the normalized histogram) ; from the AoI of the H channel
	return results


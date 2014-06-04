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

#####################################################################################################################################################################################
#Function to format the file-name of image from sample number <III>, chewing cycles <CC>, side <SS>, and file extension <EXT>
def sample_fname(III, CC, SS, EXT):
	nsample = "%03d" % III
	ncycle =  str(CC)
	#Uncomment the following line if the chewing cycles <CC> part is formatted with zeros (e.g.: 01, 02 ...09, 10, 11... etc.)
	#ncycle = "%02d" % CC
	result = nsample + '-' + ncycle + '-' + SS + EXT
	return result

#####################################################################################################################################################################################
#Function to load an image from the database, returns RGB
#	Parameters:
#		filename		- >	File-name of the image
#		database_path 	- >	Relative PATH to the image database
#	Returns:
#		rgb				- > Image in RGB
def load_image(filename, database_path):
	rgb = Image.open(database_path + filename)	#Load image from database
	rgb = np.asarray(rgb)	#transforms PIL image into a numpy array
	rgb = rgb[:,:,0:3]		#Takes the first three channels, discards additional channels
	return rgb	#Return RGB image as a (n , m , 3) numpy array

#####################################################################################################################################################################################	
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

#####################################################################################################################################################################################	
#Function to extract pixel value and histogram STD features from the AoI of a 2D Image
#	Parameters:
#		input_image			- >		2D value image
#		segmentation_mask	- >		Segmentation Mask: Binary 2D array with the same dimensions of the original image, 1s represent Area of Interest (AoI)
#		nbins				- >		Number of bins to be computed for the histogram analysis. Default value is 200
#	Returns
#		IM_Features	- >	Array: (IM_AOI_STD, IM_AOI_HIST_NORM_STD) ; from the AoI of the image
#				IM_AOI_STD			- >	 Standard deviation of the values of the pixels of the image
#				IM_AOI_HIST_NORM_STD	- >	 Standard deviation of the normalized histogram of the image
def ColorHistogramAnalysis(input_image, segmentation_mask, nbins = 200):
	IM = np.copy(input_image)	#Creates a copy of the input image
	IM_AOI = IM[segmentation_mask != 0] #Extracts the values of the pixels of IM that are inside the AoI, discards the rest
	IM_AOI_STD = np.std(IM_AOI)	#Calculates the Standard Deviation of the values of the pixels in the AoI of the image
	IM_AOI_HIST = np.histogram(IM_AOI, bins = nbins)	#Computes the histogram of the AoI of the Image for (200 by default) bins, returns array of the form [(values),(bins)]
	IM_AOI_HIST[0][0] = 0	#Clears the first bin of the histogram to reduce aberrant values
	IM_AOI_HIST_MAX = np.float64(np.amax(IM_AOI_HIST[0])) #Extracts the maximum value of the IM_AOI_HIST
	IM_HIST0_TMP = np.copy(IM_AOI_HIST[0])	#Creates a temporal copy of the IM_AOI_HIST values
	IM_AOI_HIST_NORM = IM_HIST0_TMP / IM_AOI_HIST_MAX	#Generates the normalized histogram of the AoI of image
	IM_AOI_HIST_NORM_STD = np.std(IM_AOI_HIST_NORM)	#Calculates the Standard Deviation of the normalized histogram of the AoI of the image
	IM_Features = (IM_AOI_STD, IM_AOI_HIST_NORM_STD)	#List features: (STD of the values of the pixels , STD of the normalized histogram) ; from the AoI of the image
	return IM_Features

#####################################################################################################################################################################################	
#Function to compute the Entropy Image of a 2D image after rescaling intensity and clearing non-AoI area 
#	Parameters:
#		input_image			- >		2D value image
#		segmentation_mask	- >		Segmentation Mask: Binary 2D array with the same dimensions of the original image, 1s represent Area of Interest (AoI)
#		dsize 				- >		Size of the sampling area used for the Entropy algorithm. Default value is 5
#	Returns:
#		RES_ENTRO	- >	 Entropy Image consisting of a 2D Numpy Array of the same size of the input image after rescaling intensity and clearing non-AoI area
def getEntropyImage(input_image, segmentation_mask, dsize = 5):
	IM = IM = np.copy(input_image)	#Creates a copy of the input image
	IM[segmentation_mask==0] = 0	#Clears the area around AoI, setting all non-AoI pixels as 0
	RES = exposure.rescale_intensity(IM, in_range=(0, 255))	#Rescales the intensity of the pixels of the image to the range 0 -> 255
	RES_ENTRO = entropy(RES, disk(dsize))	#Computes the entropy image of the rescaled input image
	return RES_ENTRO	
	
#####################################################################################################################################################################################	
#Function to extract a set of Mixing Features of the digitalized image of a sample of a Mixing Ability Test using chewing gums of two different colors
#	Parameters:
#		rgb					- >		RGB image of the digitalized image
#		segmentation_mask	- >		Segmentation Mask: Binary 2D array with the same dimensions of the original image, 1s represent Area of Interest (AoI)
#		dsize = 5			- >		Size of the sampling area used for the Entropy algorithm. Default value is 5
#		nbins				- >		Number of bins to be computed for the histogram analysis. Default value is 200
#	Returns:
#		MIXIG_FEATURES		- >		Array : (A_AOI_STD, A_AOI_HIST_NORM_STD, A_ENT_AOI_STD, A_ENT_AOI_HIST_NORM_STD, B_AOI_STD, B_AOI_HIST_NORM_STD, B_ENT_AOI_STD, B_ENT_AOI_HIST_NORM_STD, H_AOI_STD, H_AOI_HIST_NORM_STD)
#				A_AOI_STD				- >	 Standard deviation of the values of the pixels of the A channel from the LAB color model
#				A_AOI_HIST_NORM_STD		- >	 Standard deviation of the normalized histogram of the A channel from the LAB color model
#				A_ENT_AOI_STD			- >  Standard deviation of the values of the pixels of the rescaled entropy image of the A channel from the LAB color model
#				A_ENT_AOI_HIST_NORM_STD	- >	 Standard deviation of the normalized histogram of the entropy image of the A channel from the LAB color model
#				B_AOI_STD				- >	 Standard deviation of the values of the pixels of the B channel from the LAB color model
#				B_AOI_HIST_NORM_STD		- >	 Standard deviation of the normalized histogram of the B channel from the LAB color model
#				B_ENT_AOI_STD			- >  Standard deviation of the values of the pixels of the rescaled entropy image of the B channel from the LAB color model
#				B_ENT_AOI_HIST_NORM_STD	- >	 Standard deviation of the normalized histogram of the entropy image of the B channel from the LAB color model
#				H_AOI_STD			- >	 Standard deviation of the values of the pixels of the H channel from the HSV color model
#				H_AOI_HIST_NORM_STD	- >	 Standard deviation of the normalized histogram of the H channel from the HSV color model
def MixingFeaturesExtraction(rgb, segmentation_mask, nbins = 200, dsize = 5):
	im = np.copy(rgb)	#Creates a copy of the input image
	lab = color.rgb2lab(rgb)	#Transformation from RGB to Lab color model
	hsv = color.rgb2hsv(rgb)	#Transformation from HSV to Lab color model
	(A_AOI_STD, A_AOI_HIST_NORM_STD) = ColorHistogramAnalysis(lab[:,:,1], segmentation_mask, nbins)
	(A_ENT_AOI_STD, A_ENT_AOI_HIST_NORM_STD) = ColorHistogramAnalysis(getEntropyImage(lab[:,:,1],segmentation_mask, dsize), segmentation_mask, nbins)
	(B_AOI_STD, B_AOI_HIST_NORM_STD) = ColorHistogramAnalysis(lab[:,:,2], segmentation_mask, nbins)
	(B_ENT_AOI_STD, B_ENT_AOI_HIST_NORM_STD) = ColorHistogramAnalysis(getEntropyImage(lab[:,:,2],segmentation_mask, dsize), segmentation_mask, nbins)
	(H_AOI_STD, H_AOI_HIST_NORM_STD) = ColorHistogramAnalysis(hsv[:,:,0], segmentation_mask, nbins)
	MIXIG_FEATURES = (A_AOI_STD, A_AOI_HIST_NORM_STD, A_ENT_AOI_STD, A_ENT_AOI_HIST_NORM_STD, B_AOI_STD, B_AOI_HIST_NORM_STD, B_ENT_AOI_STD, B_ENT_AOI_HIST_NORM_STD, H_AOI_STD, H_AOI_HIST_NORM_STD)
	return MIXIG_FEATURES
	
	
	


#This script will process each image in a Mixing Ability Test database and compute a set
#of features extracted from the pixel color values, histograms, and entropy analysis. 
#This script uses the following model:
#
#
#
#
#Finally, a set of files will be generated: 
#										features.npy  - > 		A numpy style file containing the resulting
#																extracted features in a 2D array
#										batch_results.txt - >	A readable TXT file containing the results of the process.
#File-names in the database must follow this patter:
#   ###-CC-S    - >  Where ### is a three digit integer for the sample/patient (e.g. 001 or 068, etc.) ,
#   				 CC is a two digit integer for the number of chewing cycles registered in the sample,
#				     S is an uppercase A or B, representing the side of the sample that is scanned/photographed


#Import required libraries

import pygums as pygums

#Define constants
SAMPLE_INIT = 7
SAMPLE_END = 17
CHEWING_CYCLES = (3, 6, 9, 12, 15, 18, 21, 25)
DATABASE_PATH = 'database/'
FILE_EXTENSION = '.tif'
DISCK_ENTROPY = 4
SEGMENTATION_MARKER_THRESHOLD = 15


#Main loop
for patient in range(SAMPLE_INIT, SAMPLE_END):
	for cycle in CHEWING_CYCLES:
		#load each side of the sample
		sideA= pygums.load_image(pygums.sample_fname(patient, cycle, 'A', FILE_EXTENSION), DATABASE_PATH) #original image of side A of the sample
		sideB= pygums.load_image(pygums.sample_fname(patient, cycle, 'B', FILE_EXTENSION), DATABASE_PATH) #original image of side B of the sample
		
		#Generate segmentation masks for each side of the sample in order to separate the Area of Interest (AoI)
		sideA_smask = pygums.smask(sideA, SEGMENTATION_MARKER_THRESHOLD)	#Segmentation mask for the side A of the sample
		sideB_smask = pygums.smask(sideB, SEGMENTATION_MARKER_THRESHOLD)	#Segmentation mask for the side B of the sample
		
		#Extract features from the the AoI of the H channel in the HSV color model.
		#Array of the form:  (STD of the values of the pixels , STD of the normalized histogram)
		sideA_HAnalysis = pygums.HSVanalysis(sideA, sideA_smask)	#Features from the H channel for the side A of the sample 
		sideB_HAnalysis = pygums.HSVanalysis(sideB, sideB_smask)	#Features from the H channel for the side B of the sample
		
		
		




	


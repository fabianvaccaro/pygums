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
import numpy as np

#Define constants
SAMPLE_INIT = 7
SAMPLE_END = 8
CHEWING_CYCLES = (3, 6, 9, 12, 15, 18, 21, 25)
DATABASE_PATH = 'database/'
FILE_EXTENSION = '.tif'
DISCK_ENTROPY = 4
SEGMENTATION_MARKER_THRESHOLD = 15

#Define variables

#TEST_FEATURES - > Array(PATIENT, CYCLES, SIDE, A_AOI_STD, A_AOI_HIST_NORM_STD, A_ENT_AOI_STD, A_ENT_AOI_HIST_NORM_STD, B_AOI_STD, B_AOI_HIST_NORM_STD, B_ENT_AOI_STD, B_ENT_AOI_HIST_NORM_STD, H_AOI_STD, H_AOI_HIST_NORM_STD)
TEST_FEATURES = []
EVALUATION = []


#Main loop
for patient in range(SAMPLE_INIT, SAMPLE_END):
	for cycle in CHEWING_CYCLES:
		#load each side of the sample
		sideA= pygums.load_image(pygums.sample_fname(patient, cycle, 'A', FILE_EXTENSION), DATABASE_PATH) #original image of side A of the sample
		sideB= pygums.load_image(pygums.sample_fname(patient, cycle, 'B', FILE_EXTENSION), DATABASE_PATH) #original image of side B of the sample
		
		#Generate segmentation masks for each side of the sample in order to separate the Area of Interest (AoI)
		sideA_smask = pygums.smask(sideA, SEGMENTATION_MARKER_THRESHOLD)	#Segmentation mask for the side A of the sample
		sideB_smask = pygums.smask(sideB, SEGMENTATION_MARKER_THRESHOLD)	#Segmentation mask for the side B of the sample
		
		#Extract a set of Mixing Features for each side 
		SIDEA_MIXIG_FEATURES = pygums.MixingFeaturesExtraction(sideA, sideA_smask, nbins = 200, dsize = 5)
		SIDEB_MIXIG_FEATURES = pygums.MixingFeaturesExtraction(sideB, sideB_smask, nbins = 200, dsize = 5)
		

		#Store extracted features into TEST_FEATURES array
		TEMP_ARRAY_A = np.append((patient, cycle, 0),SIDEA_MIXIG_FEATURES)	# Creates a temporary array to append patient, cycle, side (0  ->  A) and SIDEA_MIXIG_FEATURES
		TEMP_ARRAY_B = np.append((patient, cycle, 1),SIDEB_MIXIG_FEATURES)	# Creates a temporary array to append patient, cycle, side (1  ->  B) and SIDEB_MIXIG_FEATURES		
		TEST_FEATURES.append(TEMP_ARRAY_A)	#Appends sample information and side A features to TEST_FEATURES
		TEST_FEATURES.append(TEMP_ARRAY_B)	#Appends sample information and side B features to TEST_FEATURES
		print 'Paciente: ' + str(patient) + ' ,  ciclo: ' + str(cycle)
		
		
TEST_FEATURES = np.asarray(TEST_FEATURES)	#Transform TEST_FEATURES to a numpy array form

#Evaluation loop
for cycle in CHEWING_CYCLES:
	TEMP = TEST_FEATURES[TEST_FEATURES[:,1] == np.float64(cycle)]
	A_AOI_STD_MEAN = np.mean(TEMP[:,3])
	A_AOI_HIST_NORM_STD_MEAN = np.mean(TEMP[:,4])
	A_ENT_AOI_STD_MEAN = np.mean(TEMP[:,5])
	A_ENT_AOI_HIST_NORM_STD_MEAN = np.mean(TEMP[:,6])
	B_AOI_STD_MEAN = np.mean(TEMP[:,7])
	B_AOI_HIST_NORM_STD_MEAN = np.mean(TEMP[:,8])
	B_ENT_AOI_STD_MEAN = np.mean(TEMP[:,9])
	B_ENT_AOI_HIST_NORM_STD_MEAN = np.mean(TEMP[:,10])
	H_AOI_STD_MEAN = np.mean(TEMP[:,11])
	H_AOI_HIST_NORM_STD_MEAN = np.mean(TEMP[:,12])
	EVALUATION.append((cycle, A_AOI_STD_MEAN, A_AOI_HIST_NORM_STD_MEAN, A_ENT_AOI_STD_MEAN, A_ENT_AOI_HIST_NORM_STD_MEAN, B_AOI_STD_MEAN, B_AOI_HIST_NORM_STD_MEAN, B_ENT_AOI_STD_MEAN, B_ENT_AOI_HIST_NORM_STD_MEAN, H_AOI_STD_MEAN, H_AOI_HIST_NORM_STD_MEAN))

EVALUATION = np.asarray(EVALUATION)	

	
np.save('features',	TEST_FEATURES)
np.savetxt('batch_results.txt', EVALUATION, fmt='%1.4f')




	


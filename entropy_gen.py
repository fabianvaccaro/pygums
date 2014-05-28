from PIL import Image
import numpy as np
import Image
import matplotlib.pyplot as plt
from skimage.filter.rank import entropy, median, mean, mean_bilateral
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage import exposure, util
import pymeanshift as pms
from numpy import array

#load images
original_image=Image.open('/home/pi/Desktop/webcam/test.jpeg')
#gris =original_image.convert('L')
im = np.asarray(original_image)
gris = im[:,:,0]
arr = np.asarray(gris)
arr.flags.writeable = True
image = img_as_ubyte(original_image)

#preprocess image
#arr = mean(arr, disk(20))
#arr = mean_bilateral(arr.astype(np.uint16), disk(8), s0=4 , s1=4)
#(arr,labels_image,number_regions)=pms.segment(arr,spatial_radius=6,range_radius=4.5,min_density=50)

#plot original image
fig, [(ax0, ax1, ax2),(ax3, ax4, ax5)] = plt.subplots(ncols=3, nrows=2, figsize=(15,8))
img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Imagen original')
ax0.axis('off')
fig.colorbar(img0, ax=ax0)

#plot entropy image
entro=entropy(arr, disk(3))
img1 = ax1.imshow(entro,cmap=plt.cm.jet)
ax1.set_title('Entropia')
ax1.axis('off')
fig.colorbar(img1, ax=ax1)


#plot histogram of entropy image
ax2.set_title('Histograma')
ax2.hist(entro.ravel(), 256, histtype='step', color='black')
ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax2.set_xlim(0,7)
ax2.set_yticks([])

#plot sample extracted from image
sample_location = (320,240)
sample_patch = arr[140:340,220:420]
img3 = ax3.imshow(sample_patch, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
ax3.set_title('Muestra')
ax3.axis('off')
fig.colorbar(img3, ax=ax3)

#plot entropy image of sample
entro4=entropy(sample_patch, disk(3))
img4 = ax4.imshow(entro4,cmap=plt.cm.jet)
ax4.set_title('Entropia de la muestra')
ax4.axis('off')
fig.colorbar(img4, ax=ax4)

#plot histogram of entropy image of the sample
ax5.set_title('Histograma de la muestra')
ax5.hist(entro4.ravel(), 40, histtype='step', color='black')
ax5.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax5.set_xlim(0,7)
ax5.set_yticks([])


#display results
plt.show()

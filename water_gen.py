from skimage.filter import canny
import numpy as np
import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filter import sobel
from skimage.morphology import watershed
from skimage import io, color
from skimage.filter.rank import entropy, median, mean, mean_bilateral
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage import exposure, util
from numpy import array
from skimage import exposure, img_as_uint, img_as_ubyte

#preparar plot
fig, [(ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)] = plt.subplots(ncols=3, nrows=3, figsize=(15,8))

#cargar imagenes >>> ori
ori = Image.open('C:/mat/pygums/wingums/test.jpeg')

#ecualizar histograma
ori = exposure.equalize_hist(ori, nbins=256)

#convertir RGB a CIE-Lab >>> lab
lab = color.rgb2lab(ori)

#extraer capa R como imagen de trabajo >>> arr
im = np.asarray(lab)
gris = im[:,:,2]
arr = np.asarray(gris)
arr.flags.writeable = True

#establecer marcadores para segmentacion por watershed
markers = np.zeros_like(arr)
markers[arr < 0] = 1
markers[arr > 0] = 2

#mascara de segmetacion por watershed
elevation_map = sobel(arr)
segmentation = watershed(elevation_map, markers)

#remocion de agujeros en la mascara de segmentacion
sinholes = ndimage.binary_fill_holes(segmentation - 1)

#igualar los pixeles del fondo en 0 usando la mascara de segmentacion >>> arrm
arrm = np.copy(arr)
arrm[sinholes==0] = 0

#reescalado de la arrm de que tiene numeros negativos al rango de 0 a 255
reescalado = exposure.rescale_intensity(arrm, in_range=(0, 255))

#imagen de entropia e histograma de entropia de arrm reescalado
entro_arrm_img = entropy(reescalado, disk(3))
entro_arrm_hist = exposure.histogram(reescalado, nbins = 100)
entro_arm_masked = entro_arrm_img[sinholes!=0]

#histograma del color de arrm reescalado dentro del area de interes >>> arrm_masked
arrm_masked = reescalado[sinholes!=0]
arrm_color_hist = exposure.histogram(arrm_masked, nbins = 256)




#plot del proceso
img0 = ax0.imshow(ori, cmap=plt.cm.gray)
ax0.set_title('Imagen original')
ax0.axis('off')
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(arr, cmap=plt.cm.gray)
ax1.set_title('Capa b* con histograma ecualizado')
ax1.axis('off')
fig.colorbar(img0, ax=ax1)

img2 = ax2.imshow(sinholes, cmap=plt.cm.gray)
ax2.set_title('Mascara de segmentacion')
ax2.axis('off')
fig.colorbar(img0, ax=ax2)

img3 = ax3.hist(arrm_masked.ravel(), 256, histtype='step', color='black')
ax3.set_title('Histograma de Color')
ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax3.set_xlim(0,0.5)
ax3.set_yticks([])

img4 = ax4.imshow(entro_arrm_img, cmap=plt.cm.jet)
ax4.set_title('Imagen de entropia')
ax4.axis('off')
fig.colorbar(img4, ax=ax4)

img5 = ax5.hist(entro_arm_masked.ravel(), 100, histtype='step', color='black')
ax5.set_title('Histograma de Entropia')
ax5.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax5.set_xlim(0,6)
ax5.set_yticks([])



plt.show()
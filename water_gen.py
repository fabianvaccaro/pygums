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
from numpy import array
from skimage import exposure, img_as_uint, img_as_ubyte, feature, util

#fijar constantes
DISCK_ENTRO = 4
MARKER_THRESHOLD = 15


#preparar plot
fig, [(ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)] = plt.subplots(ncols=3, nrows=3, figsize=(15,8))

#cargar imagenes >>> ori
directorio = 'tests/'
extension = '.tif'
nombre_fichero = 'S2'
ori = Image.open(directorio + nombre_fichero + extension)


#ecualizar histograma >>> ori_eq
ori_eq = exposure.equalize_hist(ori, nbins=256)

#convertir RGB a CIE-Lab >>> lab
lab = color.rgb2lab(ori_eq)

#extraer capa b* para la segmentacion >>> arr
im = np.asarray(lab)
gris = im[:,:,2]
arr = np.asarray(gris)
arr.flags.writeable = True

#extraer capa a* para extraccion de caracteristicas sin ecualizar>>> chic
lab_chic = color.rgb2lab(ori)
im = np.asarray(lab_chic)
gris = im[:,:,1]
chic = np.asarray(gris)
chic.flags.writeable = True



#establecer marcadores para segmentacion por watershed
markers = np.zeros_like(arr)
markers[arr < MARKER_THRESHOLD] = 1
markers[arr > MARKER_THRESHOLD] = 2

#mascara de segmetacion por watershed
elevation_map = sobel(arr)
segmentation = watershed(elevation_map, markers)

#remocion de agujeros en la mascara de segmentacion
sinholes = ndimage.binary_fill_holes(segmentation - 1)

#igualar los pixeles del fondo en 0 usando la mascara de segmentacion >>> arrm, chicm
arrm = np.copy(arr)
arrm[sinholes==0] = 0

chicm = np.copy(chic)
chicm[sinholes==0] = 0

#reescalado de arrm de que tiene numeros negativos al rango de 0 a 255
reescalado = exposure.rescale_intensity(arrm, in_range=(0, 255))
chic_reesc = exposure.rescale_intensity(chicm, in_range=(0, 255))

#imagen de entropia e histograma de entropia de arrm reescalado
entro_arrm_img = entropy(reescalado, disk(DISCK_ENTRO))
entro_arrm_hist = np.histogram(entro_arrm_img[sinholes!=0], bins = 100)
entro_arm_masked = entro_arrm_img[sinholes!=0]
max_arrm = np.float64(np.amax(entro_arrm_hist[0]))
entro_hist_tmp = np.float64(np.copy(entro_arrm_hist[0]))
entro_arrm_hist_normed = entro_hist_tmp / max_arrm
print 'STD Entropia b*: ' + str(np.std(entro_arrm_hist_normed))

#imagen de entropia e histograma de entropia de chicm reescalado
entro_chicm_img = entropy(chic_reesc, disk(DISCK_ENTRO))
entro_chicm_masked = entro_chicm_img[sinholes!=0]
entro_chicm_masked = entro_chicm_masked[entro_chicm_masked != 0]
entro_chicm_hist = np.histogram(entro_chicm_masked, bins = 100)
max_chicm_h = np.float64(np.amax(entro_chicm_hist[0]))
entro_chicm_tmp = np.float64(np.copy(entro_chicm_hist[0]))
entro_chicm_hist_normed = entro_chicm_tmp / max_chicm_h
print 'STD Entropia a*: ' + str(np.std(entro_chicm_hist_normed))




#histograma del color de arrm reescalado dentro del area de interes >>> arrm_color_hist
arrm_masked = reescalado[sinholes!=0]
arrm_color_hist = exposure.histogram(arrm_masked, nbins = 256)

#histograma del color de chicm reescalado dentro del area de interes >>> chicm_color_hist
chicm_masked = chic_reesc[sinholes!=0]
chicm_color_hist = exposure.histogram(chicm_masked, nbins = 256)
chicm_color_hist[0][0] = 0
max_chicm = np.float64(np.amax(chicm_color_hist[0]))
chicm_hist_tmp = np.float64(np.copy(chicm_color_hist[0]))
chicm_hist_normed = chicm_hist_tmp / max_chicm
print 'STD Color a*: ' + str(np.std(chicm_hist_normed))




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

img3 = ax3.plot(arrm_color_hist[1].ravel(), arrm_color_hist[0].ravel())
ax3.set_title('Histograma de color en canal b*')


img4 = ax4.imshow(entro_arrm_img, cmap=plt.cm.jet)
ax4.set_title('Imagen de entropia canal b*')
ax4.axis('off')
fig.colorbar(img4, ax=ax4)

img5 = ax5.hist(entro_arm_masked.ravel(), 100, histtype='step', color='black')
ax5.set_title('Histograma de Entropia')
ax5.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax5.set_xlim(0,6)
ax5.set_yticks([])


img6 = ax6.plot(chicm_color_hist[1].ravel(), chicm_hist_normed.ravel())
ax6.set_title('Histograma de color en canal a*')


img7 = ax7.imshow(entro_chicm_img, cmap=plt.cm.jet)
ax7.set_title('Imagen de entropia canal a*')
ax7.axis('off')
fig.colorbar(img7, ax=ax7)

img8 = ax8.hist(entro_chicm_masked.ravel(), 100, histtype='step', color='black')
ax8.set_title('Histograma de Entropia canal a*')
ax8.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax8.set_xlim(0,6)
ax8.set_yticks([])
plt.savefig('capturas_a/' + nombre_fichero + '.png')
plt.show()
plt.close()


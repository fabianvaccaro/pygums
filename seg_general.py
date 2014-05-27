from PIL import Image
import pymeanshift as pms
import numpy as np
import Image
original_image=Image.open('/home/pi/Desktop/webcam/test.jpeg')
(segmented_image,labels_image,number_regions)=pms.segment(original_image,spatial_radius=6,range_radius=4.5,min_density=50)
img=Image.fromarray(segmented_image,'RGB')
img.save('/home/pi/Desktop/webcam/lego.jpeg')

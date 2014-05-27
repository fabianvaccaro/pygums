Python 2.7.3 (default, Jan 13 2013, 11:20:46) 
[GCC 4.6.3] on linux2
Type "copyright", "credits" or "license()" for more information.
>>> from PIL import Image
>>> import mymeanshift as pms

Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    import mymeanshift as pms
ImportError: No module named mymeanshift
>>> import mymeanshift as pms

Traceback (most recent call last):
  File "<pyshell#2>", line 1, in <module>
    import mymeanshift as pms
ImportError: No module named mymeanshift
>>> import pymeanshift as pms
>>> original_image=Image.open('/home/pi/meanshift/test.jpeg')
>>> segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6,range_radius=4.5, min_density=50)
SyntaxError: invalid syntax
>>> (segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6,range_radius=4.5, min_density=50)
SyntaxError: invalid syntax
>>> (segmented_image,labels_image,number_regions) = pms.segment(original_image, spatial_radius=6,range_radius=4.5, min_density=50)
SyntaxError: invalid syntax
>>> (segmented_image,labels_image,number_regions)=pms.segment(original_image,spatial_radius=6,range_radius=4.5,min_density=50)
>>> segmented_image.save('/home/pi/Desktop/webcam/lol.jpeg')

Traceback (most recent call last):
  File "<pyshell#9>", line 1, in <module>
    segmented_image.save('/home/pi/Desktop/webcam/lol.jpeg')
AttributeError: 'numpy.ndarray' object has no attribute 'save'
>>> segmented_image.save('lol.jpeg')

Traceback (most recent call last):
  File "<pyshell#10>", line 1, in <module>
    segmented_image.save('lol.jpeg')
AttributeError: 'numpy.ndarray' object has no attribute 'save'
>>> import numpy as np
>>> import Image
>>> import Image
>>> from scipy.misc import toimage

Traceback (most recent call last):
  File "<pyshell#14>", line 1, in <module>
    from scipy.misc import toimage
ImportError: No module named scipy.misc
>>> from scipy import misc

Traceback (most recent call last):
  File "<pyshell#15>", line 1, in <module>
    from scipy import misc
ImportError: No module named scipy
>>> img=Image.fromarray(segmented_image,'RGB')
>>> img.save('/home/pi/meanshift/lego.jpeg')
>>> img.save('/home/pi/Desktop/webcam/lego.jpeg')
>>> original_image=Image.open('/home/pi/meanshift/test.jpeg')
>>> original_image.save('/home/pi/Desktop/webcam/lego.jpeg')
>>> original_image=Image.open('/home/pi/Desktop/webcam/test.jpeg')
>>> original_image.save('/home/pi/Desktop/webcam/lego.jpeg')
>>> (segmented_image,labels_image,number_regions)=pms.segment(original_image,spatial_radius=6,range_radius=4.5,min_density=50)
>>> img=Image.fromarray(segmented_image,'RGB')
>>> img.save('/home/pi/Desktop/webcam/lego.jpeg')
>>> 

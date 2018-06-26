'''script for I/O -->clustering -->generating story --sklearn'''
''' @author anurag singh'''
import numpy as np
import os
import glob
import sys
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

# resize an image using the PIL image library

import PIL
from PIL import Image
src_dir = sys.argv[1]#input("Please enter your absolute source image directory, all images in *.jpg format\n")

inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

ds=[]
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
	#print(jpgfile)
	image = load_img(jpgfile, target_size=inputShape)
	image = img_to_array(image)
	#image = np.expand_dims(image, axis=0)
	image = preprocess(image)
	ds.append(np.array(image))
data = np.array(ds)    

print("Images taken")

from keras.applications import VGG19
Network = VGG19
model = Network(weights="imagenet", include_top=False, pooling='avg')
X = model.predict(data,batch_size=10)


np.save("/".join(src_dir.split("/")[:-1])+'/features/'+src_dir.split("/")[-1]+'.npy',X)
print("feature vectors saved")

from __future__ import division
from PIL import Image
import os
import math
import sys
#set root directory
img_path=sys.argv[1]   #'C:/Users/anurag singh/Desktop'
dest_path=sys.argv[2]
def cut_in_blocks(image_path,out_name,outdir,sliceHeight,sliceWidth,step):
    img = Image.open(image_path) #open the image
    image_format=image_path.split('.')[-1]
    imageHeight , imageWidth = img.size #get image dimensions
    if(imageHeight%step!=0 or imageWidth%step!=0):
        imageWidth=imageWidth-imageWidth%step
        imageHeight=imageHeight-imageHeight%step
        img = img.resize((imageWidth,imageHeight), Image.ANTIALIAS)
    
    left=0
    right=0
    while(left<imageHeight):
        while(right<imageWidth):
            bbox =(right,left,right+sliceWidth,left+sliceHeight)
            current_slice=img.crop(bbox)# Crop image based on created bounds
            # Save your new cropped image.
            current_slice.save(dest_path+'/slice_'+str(right)+'_'+str(left)+'_'+str(step)+'.'+image_format)
            right+=step
        left+=step
        right=0
    return
os.makedirs(dest_path, exist_ok=True)
cut_in_blocks(img_path,dest_path,"",32,32,32)
'''
if __name__=='__main__':
    #iterate in all set of directory files
    for subdir,dir,files in os.walk(root_dir):
        for file in files:
            cut_in_blocks(subdir+"/"+file,newfiles,subdir,32,32,5)
'''

import numpy as np
import os
import sys,glob
import shutil
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

src_dir=sys.argv[1]#the absolute path to folder where all the cropped images are located 
feature_filename="/".join(src_dir.split("/")[:-1])+'/features/'+src_dir.split("/")[-1]+'.npy'
features = np.load(feature_filename)
print("features loaded clutering now")
#now clustering the input
hyperparameter=1
cluster=int(features.shape[0]*hyperparameter)

def reconstruct_image(src_dir,labels,dx):
    #number_of_blocks_widthwise=img_width/32
    directory=src_dir+"_summary"
    img_list=[]
    for index in labels:
        #i=int(index_of_image/number_of_blocks_widthwise)*32
        #j=int((index_of_image%number_of_blocks_widthwise)*32)
        #print(index,dx[index])
        image = Image.open(directory+"/"+str(dx[index])+".jpg")
        image.load()
        img_list.append(np.asarray(image))
        image.close()
    img_list=np.array(img_list)
    return img_list
def cluster_the_dataset(features,cluster):
    kmeans=KMeans(n_clusters=cluster,init='k-means++',random_state=3425,n_jobs=-1)
    kmeans.fit(features)
    dx=[]
    dx,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_,features)
    np.save("/".join(src_dir.split("/")[:-1])+'/dx/'+src_dir.split("/")[-1]+'_'+str(hyperparameter)+'.npy',np.array(dx))
    np.save("/".join(src_dir.split("/")[:-1])+'/dx/'+src_dir.split("/")[-1]+'_labels_'+str(hyperparameter)+'.npy',np.array(kmeans.labels_))
    return np.array(kmeans.labels_),np.array(dx)

#dest_dir = input("Please enter your absolute dest image directory, all images in *.jpg format\n")
def make_summary_directory(src_dir,dx):
    dest_dir=src_dir+'_summary'
    os.makedirs(dest_dir,exist_ok=True)
    counter=0
    i = 0
    for jpgfile in os.listdir(src_dir):
        if( counter>dx[-1]):
            #print(counter,dx[-1])
            break;
        if(counter==dx[i]):
            shutil.copy2(src_dir+"/"+jpgfile, dest_dir+"/"+str(dx[i])+".jpg")
            #print(dest_dir+"/"+str(dx[i])+".jpg")
            i+=1
        counter+=1

if __name__=="__main__":
    labels,dx=cluster_the_dataset(features,cluster)
    shadow_dx=dx
    shadow_dx.sort()
    #print(labels)
    #print(dx)
    #print(shadow_dx)
    make_summary_directory(src_dir,shadow_dx)
    img_list=reconstruct_image(src_dir,labels,dx)
    print(img_list.shape)
    np.save("img_list.npy",img_list)

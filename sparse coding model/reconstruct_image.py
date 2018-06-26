import numpy as np
import os,sys
from PIL import Image
import scipy.misc
def make_summary_vector(features,dx):
    temp=[]
    for index in dx:
        temp.append(features[index])
    summary_t=np.array(temp)
    return summary_t,np.transpose(summary_t)
def make_image_vector(image_name):
    directory="E:/btp/test/"+image_name+"_summary"
    image_list=[]
    for jpgfile in os.listdir(directory):
            #print(jpgfile)
            image = Image.open(directory+"/"+jpgfile)
            image.load()
            image_list.append(np.asarray(image))
            image.close()
    data=np.array(image_list)
    return data.reshape((data.shape[1],data.shape[2],data.shape[3],data.shape[0]))
horizontal,vertical=1952,1952
image_name=sys.argv[1]#input('enter name of image\n')
path_to_image=sys.argv[2]
feature_filename="E:/btp/test/features/"+image_name+".npy"
features=np.load(feature_filename)
train_features=np.transpose(features)

dxfile="E/btp/test/dx/"+image_name+".npy"
dx=np.load(dxfile)


#features=np.load(sys.argv[1])
#train_features=np.transpose(features)
#dx=np.load('dx/'+sys.argv[2])
summary_t,summary=make_summary_vector(features,dx)
w=np.random.rand(summary_t.shape[0],features.shape[0])
sigma=summary_t.shape[0]/feature.shape[0]
img=Image.open(path_to_image)
height,width=img.size
img=img.resize((int(sigma*height)/10,int(sigma*width)/10))
w=w.clip(min=0)
parameter=5
for i in range(40):
    #print(reconstruction_loss(w,summary,train_features,parameter))
    w=(w*summary_t.dot(train_features))/(summary_t.dot(summary).dot(w)+parameter)
summary_images=make_image_vector(image_name)
reconstructed_images=np.dot(summary_images,w)
reconstructed_images=reconstructed_images.reshape((reconstructed_images.shape[3],reconstructed_images.shape[0],reconstructed_images.shape[1],reconstructed_images.shape[2]))
img=img.resize((height,width))
reconstructed_images.resize((horizontal,vertical,3))
img.show()
#scipy.misc.toimage(reconstructed_images).show()
#scipy.misc.imsave("E:/btp/test/test_images/"+image_name+"_retrived.jpg", reconstructed_images)




from PIL import Image
import os
image_path=sys.argv[1]
image_src=Image.open(image_path)
height,width=image_src.size
result=Image.new("RGB",(height,width),color='white')
summary_directory=sys.argv[2]
image_list=os.listdir(summary_directory)
for item in image_list:
    index=int(item.split(".")[0])
    img=Image.open('summary_directory/'+item)
    x=index//61
    y=int(index%61)
    result.paste(img,(32*x,32*y,32*x+32,32*y+32))

result.show()
result.save('E:/btp/test/test_images/img1_retrived.jpg')

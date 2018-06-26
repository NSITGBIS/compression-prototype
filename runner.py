import sys
import subprocess
src_img_path=input("enter the src img path")
dest_path=input("enter the path where crop images will go")
subprocess.call(['./cropinblocks.py',srcdir,dest_img])
subprocess.call(['./fve.py',dest_path])
subprocess.call(['./cluster.py',dest_path])

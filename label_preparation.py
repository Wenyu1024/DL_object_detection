import os
import rope
import pandas as pd
import shutil
import numpy as np
from PIL import Image
import scipy.misc
from scipy import stats
import requests


dir="/home/cloud-user/wenyu/data/labels"
dir1= "/home/cloud-user/wenyu/data/labels1"
dir2="/home/cloud-user/wenyu/data/labels2"
#dir3="/media/wenyu/wenyu/labels3"
os.chdir(dir)
list= os.listdir()
for name in list:
    os.chdir(dir)
    df=pd.read_csv(name, delimiter=" ", header = None)
    df.columns=['cls','x','int','x','x1','y1','x2','y2','x','x','x','x','x','x','x']
    df1= df[df['cls'].isin(['cell'])]
    df2=df
    df2.x1[(df['x1'] < 0)]= 0.00
    df2.y1[(df['y1'] < 0)]= 0.00
    df2.x2[(df['x2'] > 1280)]= 1280.00
    df2.y2[(df['y2'] > 1024)]= 1024.00
    os.chdir(dir1)
    df1.to_csv(name,sep=' ', header= False, index= False)
    os.chdir(dir2)
    df2.to_csv(name,sep=' ', header= False, index= False)
	
	
#remember you are not working on the original tif
#but instead the png 3-255	
dir_img='/home/cloud-user/wenyu/data_preparation/img'
dir_img1='/home/cloud-user/wenyu/data_preparation/img1'	
img_list= os.listdir(dir_img)
os.chdir(dir_img)
for file in img_list:
    file='Week10_40111_B04_01_w1.tif.tif'
	shutil.copy(file, dir_img1)
    os.chdir(dir_img1)
    name= file
    #img=Image.open(name)
    img= scipy.misc.imread(name)
    img2 = img * (255/4095)   #using larger arbitury values like 8000
    #or generate a number based on the pixel distruibution.
    img2[img2>255] = 255
    #img2=img2*0.5+ 110
    scipy.misc.imsave('foo1.tif', img2)
    img2=Image.open('foo1.tif')
    img2=img2.convert('RGB')
    name= name.split('.')[0]
    name= name+'.tif.png'
    os.chdir(dest1)
    img2.save(name)

	
	
train_img_list= img_list[0:11000]

	
	
#copy the file to proper place
train_img='/home/cloud-user/wenyu/DIGITS_data/batch1/train/img'
train_lbl='/home/cloud-user/wenyu/DIGITS_data/batch1/train/lbl'
val_img='/home/cloud-user/wenyu/DIGITS_data/batch1/val/img'
val_lbl='/home/cloud-user/wenyu/DIGITS_data/batch1/val/lbl'

dir_img1='/home/cloud-user/wenyu/data_preparation/img'
img_list= os.listdir(dir_img1)
img_list= sorted(img_list)
train_img_list= img_list[0:11000]
val_img_list= img_list[11001:]

dir_lbl1='/home/cloud-user/wenyu/data_preparation/labels1'
lbl_list= os.listdir(dir_lbl1)

dir_lbl2='/home/cloud-user/wenyu/data_preparation/labels2'
lbl_list= os.listdir(dir_lbl2)
train_lbl='/home/cloud-user/wenyu/DIGITS_data/batch2/train/lbl'
val_lbl='/home/cloud-user/wenyu/DIGITS_data/batch2/val/lbl'

lbl_list=sorted(lbl_list)
train_lbl_list= lbl_list[0:11000]
val_lbl_list= lbl_list[11001:]


os.chdir('/home/cloud-user/wenyu/data_preparation/img')
for img in train_img_list:
	shutil.copy(img, train_img)
	
for img in val_img_list:
	shutil.copy(img, val_img)

os.chdir(dir_lbl1)
os.chdir(dir_lbl2)
for lbl in train_lbl_list:
	shutil.copy(lbl, train_lbl)
	
for lbl in val_lbl_list:
	shutil.copy(lbl, val_lbl)

for name in test:
	a=split(name, '.')
	b=a[0]
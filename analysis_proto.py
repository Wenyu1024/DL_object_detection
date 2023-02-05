# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:04:43 2017

@author: wenyu
"""
import os
import copy
import rope
import pandas as pd
import shutil
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use('tkagg')    #YAAA!!  this finally makes the Damn thing work
#import matplotlib.pyplot as plt
import scipy.misc
from scipy import stats
import requests
import time
from numpy imp
#The time module is part of Python's standard library. It's installed along with the rest of Python, and you don't need to (nor can you!) install it with pip.
#%%
#pick the nuclei images
os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\data\\images\\Week1_22141")
path = os.getcwd()
tif_list = os.listdir(path)
index_tif = list(range(0,len(tif_list)-1,  3))
index_tif1 = [tif_list[i] for i in index_tif]

for x in index_tif1:
        shutil.copy(x, "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\images2")

#%%
#pick the related label files
os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\image analysis\\original dataset for cell\\labels from Lassi\\labels\\anal2_forLassi")
path = os.getcwd()
txt_list = os.listdir(path)
index_txt = list()
for name in txt_list:
    if "22141" in name:
        index_txt.append(name)

import shutil
for x in index_txt:
        shutil.copy(x, "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\labels")
        

result =   pd.DataFrame({'x': [],
                         'y': [],
                         'long': [],
                         'short': []})
for name in index_txt:
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels")
    shutil.copy(name, "C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels2")
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels2")
    dataset = pd.read_csv(name, delimiter="  ", header = None,)
    dataset1 = dataset[[0,1,208,209]]
    dataset1.columns = ["x","y","long","short"]
    result = result.append(dataset1)
    dataset1.to_csv(name, sep='\t',index= False, header=False) 


plt.scatter(result[["long"]],result[["short"]])
#plt.show()
#plt.hist(result[["long"]])  #sth wrong? its a dataframe...
#plt.show() #so use plot method for df instead
#type(result[["long"]])
result["long"].hist(bins= 100)
result["short"].hist(bins= 100)
#looks normally distributed
#use 68.3% of the data???
long_1 = result["long"].describe()["mean"]+result["long"].describe()["std"]
long_2 = result["long"].describe()["mean"]-result["long"].describe()["std"]
short_1 = result["short"].describe()["mean"]+result["short"].describe()["std"]
short_2 = result["short"].describe()["mean"]-result["short"].describe()["std"]
#so the smallest cell nuclei should be 33*26
#for the # 1280*1024 image size
# generate the grid paper 
#size 16*16
#number 80*64

######################################################
#%%
#deal with object rectangules, transform them as squares.
for name in index_txt:
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels2")
    shutil.copy(name, "C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels3")
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels3")
    df=pd.read_csv(name, delimiter="\t", header = None)
    df.columns = ["x","y","long","short"]
    df['cls'] = "Cell"
    df.cls[(df['long'] > long_1)] = "dontcare"
    df.cls[(df['long'] < long_2)] = "dontcare"
    df.cls[(df['short'] > short_1)] = "dontcare"
    df.cls[(df['short'] < short_2)] = "dontcare"
    df["x1"] = df["x"]- 0.5*df["long"] 
    df["x2"] = df["x"]+ 0.5*df["long"] 
    df["y1"] = df["y"]- 0.5*df["long"] 
    df["y2"] = df["y"]+ 0.5*df["long"] 
    df.cls[(df['x1'] < 0)]= "dontcare"
    df.cls[(df['y1'] < 0)]= "dontcare"
    df.cls[(df['x2'] > 1280)]= "dontcare"
    df.cls[(df['y2'] > 1024)]= "dontcare"
#    #now you want to change the index back to start at 0
#    df = df.reset_index()
#    del df['index']
    
    #now first generate a df with 15 col each length m*n
    #all the values could be 0
    #using a length-fixed df accerlate the computation
    output= np.zeros((80*64,15), dtype=np.int)
    output = pd.DataFrame(output,columns=['cls_grid','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','x',])
    for n in range(1,80) :
        for m in range(1,64) :
            x_grid = 8+ 16*(n-1)
            y_grid = 8+ 16*(m-1)
            dis_last=9999999999
            for x_obj,y_obj,x1_obj,y1_obj,x2_obj,y2_obj,long, in zip(df["x"],df["y"],df["x1"],df["y1"],df["x2"],df["y2"],df["long"]):
                dis= (x_obj-x_grid)**2+ (y_obj-y_grid)**2
                if dis < 16**2+ long**2 and dis < dis_last:
                    output["cls_grid"][m+64*(n-1)] = 1
                    output["x1"][m+64*(n-1)] = (x1_obj-x_grid)
                    output["y1"][m+64*(n-1)] = (y1_obj-y_grid)
                    output["x2"][m+64*(n-1)] = (x2_obj-x_grid)
                    output["y2"][m+64*(n-1)] = (y2_obj-y_grid)
    output.to_csv(name,sep=',', header= False, index= False)



################################################
##Please notice that input data for digits is not grid based but object 
##based. each txt should have 15 columes including x1, y1, x2, y2
##please refer to https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/'
for name in index_txt:
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels2")
    shutil.copy(name, "C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels3")
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels3")
    df=pd.read_csv(name, delimiter="\t", header = None)
    df.columns = ["x","y","long","short"]
    df['cls'] = "Cell"
    df.cls[(df['long'] > long_1)] = "DontCare"
    df.cls[(df['long'] < long_2)] = "DontCare"
    df.cls[(df['short'] > short_1)] = "DontCare"
    df.cls[(df['short'] < short_2)] = "DontCare"
    df["x1"] = df["x"]- 0.5*df["long"] 
    df["x2"] = df["x"]+ 0.5*df["long"] 
    df["y1"] = df["y"]- 0.5*df["long"] 
    df["y2"] = df["y"]+ 0.5*df["long"] 
    df.cls[(df['x1'] < 0)]= "DontCare"
    df.cls[(df['y1'] < 0)]= "DontCare"
    df.cls[(df['x2'] > 1280)]= "DontCare"
    df.cls[(df['y2'] > 1024)]= "DontCare"
    df= df.round(2)
    col_len = len(df["x"])
    output= np.full([col_len,15],0.0)
    output=output.round(1)
    output=pd.DataFrame(output,columns=['cls','x','int','x','x1','y1','x2','y2','x','x','x','x','x','x','x'])
    output['cls'] = df["cls"]
    output["x1"] = df["x1"]
    output["x2"] = df["x2"]
    output["y1"] = df["y1"]
    output["y2"] = df["y2"]
    output["int"] = int(0)
    output.to_csv(name,sep=' ', header= False, index= False)
#############################################
#rename images and labels
os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\data\\images2")
index_label = os.listdir()
path = os.getcwd()
files = os.listdir(path)
i = 1
for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i).zfill(5) +'.tif'))
    i = i+1

os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\labels3")
index_label = os.listdir()
path = os.getcwd()
files = os.listdir(path)
i = 1
for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i).zfill(5) +'.txt'))
    i = i+1


#######################################
#    ##play with code
###rememeber you used np.full (see onenote)
###but you lost the code here
#from PIL import Image
#os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\images4")
#path=os.getcwd()
#dir2= os.listdir(path)
#n=1
#for file in dir2:
#    name= '00001.tif'
#    img= Image.open(name)
#    img.mode
#    img.getdata()
#    img_rgb= img.convert('RGB')
#    img.getpixel((0,0))
#    img_rgb.getpixel((0,0))
#    img_rgb.
#    print(img.format, img.size, img.mode) 
#    print(img_rgb.format, img_rgb.size, img_rgb.mode) 
##   image.mode = 'I'
#    name=name.split('.')[0]
#    name= name+''
#    
#    img=Image.open('foo.png')
#    img.rgb.save("foo2.png")
#    
#    img=np.array(img)
#    stats.describe(img)
#    np.max(img)
#    np.mean(img)
#    np.min(img)
#    img2= img + (127-2)
#    np.mean(img2)
#    scipy.misc.imsave('foo3.png', img2)
########################################################################real code
#
#os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\python dir\\all I need for running the caffe\\data\\images4")
#path=os.getcwd()
#dir2= os.listdir(path)
#def w(x):
#    x = x * (255/4095)
#    return x if x < 255 else 255
#
#for file in dir2:
#    name= file
##    img=Image.open(name)
##    img=img.point(lambda i:i*(255/4095))
##    img=img.convert('RGB')
##    img.save()
#    img= np.array(Image.open(name))
#    
#    img2 = f(img)
#    img2 = img2 * 0.5 + 110
#    # Test code
#    img2 = img2 * (255/4095)
#    img2[img2>255] = 255
#    # End tst code
#    
#    scipy.misc.imsave('foo.png', img2)
#    img3=Image.open('foo.png')
#    img3=img3.convert('RGB')
#    
#    name= name.split('.')[0]
#    name= name+'.png'
#    img3.save(name)
##############################################
##reformat and rescale images
#    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\data\\images2")
#    dir2 = os.listdir()
#    for file in dir2:
#        name= file
#        #img=Image.open(name)
#        img= scipy.misc.imread(name)
#        img2 = img * (255/4095)
#        img2[img2>255] = 255
#        img2=img2*0.5+ 110
#        scipy.misc.imsave('foo.tif', img2)
#        img3=Image.open('foo.tif')
#        img3=img3.convert('RGB')
#        name= name.split('.')[0]
#        name= name+'.png'
#        img3.save(name)
#    #img= np.array(Image.open('foo.png'))
#    #img=np.array(img)
#    
####################################################

#adjust the label based on up left zero point
os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\data\\labels3")
index_txt = os.listdir()

for name in index_txt:
    #name= index_txt[3]
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\data\\labels3")
    shutil.copy(name, "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\labels4")
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\data\\labels4")
    df=pd.read_csv(name, delimiter=" ", header = None)
    df.columns=['cls','x','int','x','x1','y1','x2','y2','x','x','x','x','x','x','x']
    df.y1= 1080- df.y1
    df['y1']= df['y1'].round(decimals=2)
    df.y2= 1080- df.y2
    df['y2']= df['y2'].round(decimals=2)
    df.to_csv(name,sep=' ', header= False, index= False)
    
##############################################
#this steps seems not working properly so in the end I download all the file manuually
import requests
r = requests.get('https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip', allow_redirects=True)
print r.headers.get('content-type')
################################################
#this is for processinng all the zip files from the url
os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large')

import zipfile
filelist= os.listdir()
filelist= filelist[0:53]
for name in filelist:
    with zipfile.ZipFile(name,"r") as zip_ref:
        zip_ref.extractall("targetdir")
###

######################################################################
# the reason for this code is images in week4 do not have proper names for subsequent processing
dir1= 'C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\targetdir3'
os.chdir(dir1)
filelist= os.listdir()
for name in filelist:
    #name= 'Week4_27481'
    os.chdir(dir1)
    dir2= os.path.join(dir1, name)
    os.chdir(dir2)
    tif_list=os.listdir()
    for name2 in tif_list:
        #name2= 'B02_s1_w1FDD9356D-FF65-44C5-8B07-6BEC430F6014.tif'
         os.rename(os.path.join(dir2, name2), os.path.join(dir2, name+ '_' + name2))

###########################################
#%%
os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\targetdir')
filelist2= os.listdir()
for name in filelist2:
    os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\targetdir')
    name= 'Week1_22123'
    #dir= chr(name)
    dir ='C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\targetdir\\' + name
    os.chdir(dir)
    tif_list = os.listdir()
    index_tif = list(range(0,719, 3))
    index_tif1 = [tif_list[i+1] for i in index_tif]
    index_tif2 = index_tif1[8:239]
    for x in index_tif2:
            shutil.copy(x, "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\targetdir4")

##########################################################
#%%
os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\test_set')
dir=os.getcwd()
dest= "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\cell_images_test1"
list1= os.listdir()
index = list(range(0,719,3))
for folder in list1:
    #folder=list1[1]  # comment this 
    dir2= os.path.join(dir, folder)
    os.chdir(dir2)
    list2=os.listdir()
    list2=[list2[n] for n in index]     #????
    list2=list2[8:]      # get the 232
    n=1
    for file in list2:
        #file= list2[9] # comment this 
        shutil.copy(file, dest)
        name= file.split('_')[2]
        num= n%5
        #num= chr(num)  #smile face
        num= str(num)
        name= folder+ '_' + name + '_0' + num + '_w1.tif.tif' 
        os.rename(os.path.join(dest, file), os.path.join(dest, name))
        n= n+1
        if n>4:
            n=n-4 
        else: n=n-0
        
        
#%%
ppt='C:\\Users\\wenyu\\Documents\\image deep learning\\data\\raw_data\\example'
dest= "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\targetdir3"
dir2 = os.listdir(ppt)
dest1= "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\targetdir4"
os.chdir(ppt)
dir2 = os.listdir()
for file in dir2:
    #file=dir2[0]
    #os.chdir(dest)
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
    img2.save(name)
    #img2.save('test2.png')
    #%%
#############################################################
dir="C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels"
os.chdir(dir)
list= os.listdir()
result =   pd.DataFrame({'x': [],
                         'y': [],
                         'long': [],
                         'short': []})
    #list_test= list[:4]
for name in list:
    df = pd.read_csv(name, delimiter="  ", header = None)
    df1 = df[[0,1,208,209]]
    df1.columns = ["x","y","long","short"]
    result = result.
    append(df1)
    #dataset1.to_csv(name, sep='\t',index= False, header=False) 


plt.scatter(result[["long"]],result[["short"]])
#plt.show()
#plt.hist(result[["long"]])  #sth wrong? its a dataframe...
#plt.show() #so use plot method for df instead
#type(result[["long"]])
result["long"].hist(bins= 100)
result["short"].hist(bins= 100)
#looks normally distributed
#use 68.3% of the data???
long_1 = result["long"].describe()["mean"]+ 2*result["long"].describe()["std"]
long_2 = result["long"].describe()["mean"]-2*result["long"].describe()["std"]
short_1 = result["short"].describe()["mean"]+2*result["short"].describe()["std"]
short_2 = result["short"].describe()["mean"]-2*result["short"].describe()["std"]
#%%

dir=os.getcwd()
os.chdir(dir)
list= os.listdir()
#list= 
for name in list:
    #name='Week10_40111_B04_01_w1.tif.txt'
    df=pd.read_csv(name, delimiter="  ", header = None)
    df = df[[0,1,208,209]]
    df.columns = ["x","y","long","short"]
    df['cls'] = "cell"
    df.cls[(df['long'] > long_1)] = "dontcare"
    df.cls[(df['long'] < long_2)] = "dontcare"
    df.cls[(df['short'] > short_1)] = "dontcare"
    df.cls[(df['short'] < short_2)] = "dontcare"
    df["x1"] = df["x"]- 0.5*df["long"] 
    df["x2"] = df["x"]+ 0.5*df["long"] 
    df["y1"] = df["y"]- 0.5*df["long"] 
    df["y2"] = df["y"]+ 0.5*df["long"] 
    df.cls[(df['x1'] < 0)]= "dontcare"
    df.cls[(df['y1'] < 0)]= "dontcare"
    df.cls[(df['x2'] > 1280)]= "dontcare"
    df.cls[(df['y2'] > 1024)]= "dontcare"
    df= df.round(2)
    col_len = len(df["x"])
    output= np.full([col_len,15],0.0)
    output=output.round(1)
    output=pd.DataFrame(output,columns=['cls','x','int','x','x1','y1','x2','y2','x','x','x','x','x','x','x'])
    output['cls'] = df["cls"]
    output["x1"] = df["x1"]
    output["x2"] = df["x2"]
    output["y1"] = df["y1"]
    output["y2"] = df["y2"]
    output["int"] = int(0)
    output.to_csv(name,sep=' ', header= False, index= False)


## data generated by these code have values lower than 0 and larger
##than max. I fixed it in the next piece of code
    
####
#(1)generate the labels that don't have the dontcare record, and
#(2)labels with dontcare but do not have error locations
#(3) true negative dontcare records that can generate labels that having the abnormals 
dir="C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels"
dir1= "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels1"
dir2="C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels2"
dir3="C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels3"
os.chdir(dir)
list= os.listdir()
for name in list:
    #name='Week10_40111_B04_01_w1.tif.txt'
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
    
    
#####################################################################
#generate the labels that using the black background as dontcare
#using the concept of the grid and use those don't care ids
# using long_1 short_1 from previous image
    #--- 64.80653762817383 seconds ---9.22day
import time

os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels5')
list= os.listdir()
for name in list:
    start_time = time.time()
    name=list[0]
    os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels5')
    df = pd.read_csv(name, delimiter="  ", header = None)
    df = df[[0,1,208,209]]
    df.columns = ["x","y","long","short"]
#deal with object rectangules, transform them as squares.
    df['cls'] = "cell"
    df.cls[(df['long'] > long_1)] = "dontcare"
    df.cls[(df['long'] < long_2)] = "dontcare"
    df.cls[(df['short'] > short_1)] = "dontcare"
    df.cls[(df['short'] < short_2)] = "dontcare"
    df["x1"] = df["x"]- 0.5*df["long"] 
    df["x2"] = df["x"]+ 0.5*df["long"] 
    df["y1"] = df["y"]- 0.5*df["long"] 
    df["y2"] = df["y"]+ 0.5*df["long"] 
    df.cls[(df['x1'] < 0)]= "dontcare"
    df.cls[(df['y1'] < 0)]= "dontcare"
    df.cls[(df['x2'] > 1280)]= "dontcare"
    df.cls[(df['y2'] > 1024)]= "dontcare"
    df.x1[(df['x1'] < 0)]= 0
    df.y1[(df['y1'] < 0)]= 0
    df.x2[(df['x2'] > 1280)]= 1280
    df.y2[(df['y2'] > 1024)]= 1024
    #now first generate a df with 15 col each length m*n
    #all the values could be 0
    #using a length-fixed df accerlate the computation
    #x_grid and y_grid is the center of a grid.
    #so that I can use the distance between the center of a grid and an object to 
    #compare with the object lenth(using the long axis here)
    output= np.zeros((80*64,15), dtype=np.int)
    output = pd.DataFrame(output,columns=['cls','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','x'])
    for n in range(1,81) :
        for m in range(1,65) :
            x_grid = 8+ 16*(n-1)
            y_grid = 8+ 16*(m-1)
            dis_last=9999999999
            output.loc[m+64*(n-1)-1,'x1']  = x_grid-8
            output.loc[m+64*(n-1)-1,'y1'] = y_grid-8
            output.loc[m+64*(n-1)-1,'x2'] = x_grid+8
            output.loc[m+64*(n-1)-1,'y2'] = y_grid+8
            output.loc[m+64*(n-1)-1,'cls'] = 'dontcare'
            for x_obj,y_obj,x1_obj,y1_obj,x2_obj,y2_obj,long, in zip(df["x"],df["y"],df["x1"],df["y1"],df["x2"],df["y2"],df["long"]):
                dis= (x_obj-x_grid)**2+ (y_obj-y_grid)**2
                if dis < 16**2+ long**2:
                    output.loc[m+64*(n-1)-1,'cls'] = 'delete'
                dis_last=dis
                
    output1= output[output['cls'].isin(['dontcare'])]            
    os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels1")
    df1=pd.read_csv(name, delimiter=" ", header = None)
    df1.columns=['cls','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','x']
    output2=output1.sample(n=250-len(df1.cls), replace=True) ## 600 not 250
    output2=df1.append(output2,ignore_index=True)
    output2.x1=output2.x1.astype(int)
    output2.y1=output2.y1.astype(int)
    output2.x2=output2.x2.astype(int)
    output2.y2=output2.y2.astype(int)
    os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels3')
    output.to_csv(name,sep=' ', header= False, index= False)
    print("--- %s seconds ---" % (time.time() - start_time))

####################################################################
#try to optimize the script so that I don't need 9.22 day for this

os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels3')
list= os.listdir()
output_proto= np.zeros((80*64,15), dtype=np.int)
output_proto = pd.DataFrame(output_proto, columns=['cls','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','e'])

for n in range(1,81):
    for m in range(1,65):
        x_grid = 8+ 16*(n-1)
        y_grid = 8+ 16*(m-1)
        output_proto.loc[m+64*(n-1)-1,'x1']  = x_grid-8
        output_proto.loc[m+64*(n-1)-1,'y1'] = y_grid-8
        output_proto.loc[m+64*(n-1)-1,'x2'] = x_grid+8
        output_proto.loc[m+64*(n-1)-1,'y2'] = y_grid+8
        output_proto.loc[m+64*(n-1)-1,'cls'] = 'dontcare'
        output_proto.loc[m+64*(n-1)-1,'x_grid']=x_grid
        output_proto.loc[m+64*(n-1)-1,'y_grid']=y_grid



#for name in list:
name='Week5_28921_c02_02_w1.tif.txt'
df=pd.read_csv(name, delimiter=" ", header = None)
df.columns=['cls','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','e']
output=copy.deepcopy(output_proto)
n=0
start_time = time.time()
test1=[]
test2=[]
while n<5120:
    dis_nearest=9999999999
    x_grid=output.loc[n,'x_grid']
    y_grid=output.loc[n,'y_grid']
    n
    for x1,y1,x2,y2 in zip(df["x1"],df["y1"],df["x2"],df["y2"]):
#    while m< len_cell:
#        x1=df.loc[m,'x1']
#        y1=df.loc[m,'y1']
#        x2=df.loc[m,'x2']
#        y2=df.loc[m,'y2']
        long= x2-x1
        x_obj= (x2-x1)/2+x1
        y_obj= (y2-y1)/2+y1
        dis_sq= (x_obj-x_grid)**2+ (y_obj-y_grid)**2
        suml= 16**2+ long**2
        #m+=1
        if dis_sq > dis_nearest:
            continue
        else:
            if dis_sq < suml :
                output.loc[n,'cls'] = 'delete'
                test1.append(suml-dis_sq)
                test2.append(m)
                #break
            else:
                dis_nearest=dis_sq
    n=n+1
    
print("--- %s seconds ---" % (time.time() - start_time))

    output=output.iloc[:,0:15]
    output1= output[output['cls'].isin(['dontcare'])]
    output2=output1.sample(n=600-len(df.cls), replace=True)
    output2=df.append(output2,ignore_index=True)
    output2.x1=output2.x1.astype(int)
    output2.y1=output2.y1.astype(int)
    output2.x2=output2.x2.astype(int)
    output2.y2=output2.y2.astype(int)
    #output2.to_csv(name,sep=' ', header= False, index= False)
    print("--- %s seconds ---" % (time.time() - start_time))

####################################################################
##the right code to generate the dontcare object
##using the cell-only lable files generated previously.
os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels1')
lbl_list= os.listdir()
output_proto= np.zeros((40*32,15), dtype=np.int)
output_proto = pd.DataFrame(output_proto, columns=['cls','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','e'])

##generate the grid paper to used in the next step
for n in range(1,41):
    for m in range(1,33):
        x_grid = 16+ 32*(n-1)
        y_grid = 16+ 32*(m-1)
        output_proto.loc[m+32*(n-1)-1,'x1']  = x_grid-16
        output_proto.loc[m+32*(n-1)-1,'y1'] = y_grid-16
        output_proto.loc[m+32*(n-1)-1,'x2'] = x_grid+16
        output_proto.loc[m+32*(n-1)-1,'y2'] = y_grid+16
        output_proto.loc[m+32*(n-1)-1,'cls'] = 'dontcare'
        output_proto.loc[m+32*(n-1)-1,'x_grid']=x_grid
        output_proto.loc[m+32*(n-1)-1,'y_grid']=y_grid


##given an image, for every grid, test if the grid is overlaped with cells
for name in lbl_list:
start_time = time.time()
name=lbl_list[100]
df=pd.read_csv(name, delimiter=" ", header = None)
df.columns=['cls','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','e']
output=copy.deepcopy(output_proto)
n=0
while n<1280:
    dis_nearest=9999999999
    x_grid=output.loc[n,'x_grid']
    y_grid=output.loc[n,'y_grid']
    for x1,y1,x2,y2 in zip(df["x1"],df["y1"],df["x2"],df["y2"]):
        long= x2-x1
        x_obj= (x2-x1)/2+x1
        y_obj= (y2-y1)/2+y1
        dis_sqr= (x_obj-x_grid)**2+ (y_obj-y_grid)**2
        suml= 8**2+ (long/2)**2
        if dis_sqr > dis_nearest:
            continue
        else:
            if dis_sqr < suml :
                output.loc[n,'cls'] = 'delete'
                break
            else:
                dis_nearest=dis_sqr
    n=n+1

output=output.iloc[:,0:15]
output1= output[output['cls'].isin(['dontcare'])]
output2=output1.sample(n=50, replace=True)
output2=df.append(output2,ignore_index=True)
output2.x1=output2.x1.astype(int)
output2.y1=output2.y1.astype(int)
output2.x2=output2.x2.astype(int)
output2.y2=output2.y2.astype(int)
os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data')
output2.to_csv(name,sep=' ', header= False, index= False)
print("--- %s seconds ---" % (time.time() - start_time))

###############################
#exclude the lbl and imgs without cell
dir= 'C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\labels_null'
os.chdir(dir)
list=os.listdir(dir)
os.chdir("C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\cell_images1")
dest= "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\cell_images_null"
for img in list:
    img= img.split('.')[0] + '.tif.png'
    shutil.copy(img, dest)

#####################################3
# deal with the label #22141
os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\22141_test')
lbl_list= os.listdir()
for name in lbl_list:
    #start_time = time.time()
    name='Week1_22141_G11_04_w1.tif.txt'
    df = pd.read_csv(name, delimiter="  ", header = None)
    df = df[[0,1,208,209]]
    df.columns = ["x","y","long","short"]
#deal with object rectangules, transform them as squares.
    df['cls'] = "cell"
    df.cls[(df['long'] > long_1)] = "dontcare"
    df.cls[(df['long'] < long_2)] = "dontcare"
    df.cls[(df['short'] > short_1)] = "dontcare"
    df.cls[(df['short'] < short_2)] = "dontcare"
    df["x1"] = df["x"]- 0.5*df["long"] 
    df["x2"] = df["x"]+ 0.5*df["long"] 
    df["y1"] = df["y"]- 0.5*df["long"] 
    df["y2"] = df["y"]+ 0.5*df["long"] 
    df.cls[(df['x1'] < 0)]= "dontcare"
    df.cls[(df['y1'] < 0)]= "dontcare"
    df.cls[(df['x2'] > 1280)]= "dontcare"
    df.cls[(df['y2'] > 1024)]= "dontcare"
    df.x1[(df['x1'] < 0)]= 0
    df.y1[(df['y1'] < 0)]= 0
    df.x2[(df['x2'] > 1280)]= 1280
    df.y2[(df['y2'] > 1024)]= 1024
    df1=df.iloc[:,4:]
    df1= df1[df1['cls'].isin(['cell'])]
    df1.x1=df1.x1.astype(int)
    df1.y1=df1.y1.astype(int)
    df1.x2=df1.x2.astype(int)
    df1.y2=df1.y2.astype(int)
    output=copy.deepcopy(output_proto)
    output2=output.iloc[0:len(df1.cls),:]
    output2.cls= df1.cls
    output2.x1= df1.x1
    output2.y1= df1.y1
    output2.x2= df1.x2
    output2.y2= df1.y2
    os.chdir('C:\\Users\\wenyu\\Documents\\image deep learning')
    output2.to_csv(name,sep=' ', header= False, index= False)
######################################################
# three images into one

dir= "C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\3to1"
os.chdir(dir)
file = os.listdir(dir)
#for file in dir2:
    #file=dir2[0]
    #os.chdir(dest)

    #img=Image.open(name)
    img3=[]
    #name_list= ['img_' + str(n) for n in range(0,3)]
    n=0
    for img in file: 
        img= scipy.misc.imread(img)
        img2 = img * (255/8000)   #using larger arbitury values like 8000
    #or generate a number based on the pixel distruibution.
        img2[img2>255] = 255
        img3.append(img2)
    img2.save(test)
    img_rgb= img2.convert('RGB')
r=img3[0]
g=img3[1]
b=img3[2]
    # r, g, and b are 512x512 float arrays with values >= 0 and < 1.
from PIL import Image
import numpy as np
rgbArray = np.zeros((1024,1280,3), 'uint8')
rgbArray[..., 0] = r
rgbArray[..., 1] = g
rgbArray[..., 2] = b
img = Image.fromarray(rgbArray)
img.save('myimg_bgr.png')

    #img2=img2*0.5+ 110
    #scipy.misc.imsave('foo1.tif', img2)
    #img3=Image.open('foo1.tif')
    name= name.split('.')[0]
    name= name+'.tif.png'
    img2.save(name)
    #img2.save('test2.png')
    #%%
    
#Intersection over Union (IoU) for object detectionPython

# define the iou function

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
#    assert bb1['x1'] < bb1['x2']
#    assert bb1['y1'] < bb1['y2']
#    assert bb2['x1'] < bb2['x2']
#    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
#    assert iou >= 0.0
#    assert iou <= 1.0
    return iou



dir_pre= 'C:\\Users\\wenyu\\Documents\\image deep learning\\test_general\\pred'
dir_true= 'C:\\Users\\wenyu\\Documents\\image deep learning\\test_general\\test_label'
iou_threholds= 0.6
TP=0
FP=0
FN=0
pred_list= os.listdir(dir_pre)

os.chdir(dir_pre)
for name in pred_list:
    #name= pred_list[1]
#    name1=name.split('#')
#    name2=name1.split('.')
    name1=name.replace('#','.').replace('_','_').split('.')
    name2=name1[1]
    name2=name2.zfill(3)
    os.rename(name,name2)
pred_list1=os.listdir(dir_pre)

gt_list= os.listdir(dir_true)
gt_list.sort()
n=0
while n<len(pred_list):
    name_pre= pred_list1[n]
    name_true=gt_list[n]
    os.chdir(dir_pre)
    predicted = np.loadtxt(name_pre)
    predicted = predicted[1:,0:4]
    predicted1 = predicted[predicted[:, 0] > 0]
    predicted1 = predicted1[predicted1[:, 1] > 0]
    predicted1 = predicted1[predicted1[:, 2] < 1280]
    predicted1 = predicted1[predicted1[:, 3] <1024]
    os.chdir(dir_true)
    truth= np.loadtxt(name_true,usecols=(1,2,3,4))
    for pred in predicted1:
        iou_list=[0]
        for gt in truth:
            bb1= {'x1':pred[0],'y1':pred[1],'x2':pred[2],'y2':pred[3]}
            bb2= {'x1':gt[0],'x2':gt[1],'y1':gt[2],'y2':gt[3]}
            iou=get_iou(bb1,bb2)
            if iou < 0.6:
                continue
            else: 
                iou_list.append(iou)
        if max(iou_list) <0.6:
            FP+=1
        else: 
            TP+=1
    for gt in truth:
        iou_list=[0]
        for pred in predicted1:
            bb1= {'x1':pred[0],'y1':pred[1],'x2':pred[2],'y2':pred[3]}
            bb2= {'x1':gt[0],'x2':gt[1],'y1':gt[2],'y2':gt[3]}
            iou=get_iou(bb1,bb2)
            iou_list.append(iou)
        if max(iou_list) > 0.6:
            continue
        else: 
            FN+=1
    n+=1
    
    
precision= TP/(TP+FP)
recall=TP/(TP+FN)
map=precision*recall
#>>> from numpy import array
#>>> a = array([1, 2, 3, 1, 2, 3])
#>>> b = a>2 
#array([False, False, True, False, False, True], dtype=bool)
#>>> r = array(range(len(b)))
#>>> r(b)
#[2, 5]


dir= 'C:\\Users\\wenyu\\Documents\\image deep learning\\data\\large\\test_images'
img_list=os.listdir(dir)
img_list.sort()
f=open('f1.txt','w')
for ele in img_list:
    ele1= '/wenyu/test_general/test_images/' + ele
    f.write(ele+'\n')

f.close()
img_list2= list()
for ele in img_list:
    ele1= '/wenyu/test_general/test_images/' + ele
    img_list2.append(ele1)

img_list2=pd.DataFrame(data=img_list2)
img_list3=img_list2.transpose()
img_list3.to_csv('img_list', sep=' ', index= False, header=False)


####
os.chdir()
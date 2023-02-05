import os
import pandas as pd
import numpy as np
import time
import copy
output_proto= np.zeros((40*32,15))
output_proto = pd.DataFrame(output_proto, columns=['cls','x','int1','x','x1','y1','x2','y2','x','x','x','x','x','x','e'])

gs = (32,32) # Grid size
suml = gs[0]**2 + gs[1]**2 # Min distance

##generate the grid paper to used in the next step
for n in range(40):
    for m in range(32):
        x_grid = gs[0]/2 + gs[0] * n
        y_grid = gs[1]/2 + gs[1] * m
        output_proto.loc[m+1+32*n,'x1']  = x_grid-gs[0]/2
        output_proto.loc[m+1+32*n,'y1'] = y_grid-gs[1]/2
        output_proto.loc[m+1+32*n,'x2'] = x_grid+gs[0]/2
        output_proto.loc[m+1+32*n,'y2'] = y_grid+gs[1]/2
        output_proto.loc[m+1+32*n,'cls'] = 'dontcare'
        output_proto.loc[m+1+32*n,'x_grid']=x_grid
        output_proto.loc[m+1+32*n,'y_grid']=y_grid

#path= '/home/user/Desktop/wenyu/data/labels1'
path = '/home/user/Desktop/wenyu/data/labels1'
outpath = '/home/user/Desktop/wenyu/data/labels3'
os.chdir(path)
lbl_list= os.listdir(path)
##given an image, for every grid, test if the grid is overlaped with cells
#for name in lbl_list:
start_time = time.time()
name=lbl_list[3] #this file has the average size in the folder labels1
df=pd.read_csv(name, delimiter=" ", header = None)
df.columns=['cls','x','int1','x','x1','y1','x2','y2','x','x','x','x','x','x','e']
output=copy.deepcopy(output_proto)
tmp = df[['x1','y1','x2','y2']].as_matrix()
coords = np.zeros((tmp.shape[0],2))
coords[:,0] = (tmp[:,2]+tmp[:,0]) / 2
coords[:,1] = (tmp[:,3]+tmp[:,1]) / 2
gridpoints = output[['x_grid','y_grid']].as_matrix()
for n in range(coords.shape[0]):
    dist = ((gridpoints - coords[n,:])**2).sum(axis=1)
    ind = (dist < suml)
    output.loc[ind,'cls'] = 'delete'

output=output.iloc[:,0:15]
output1= output[output['cls'].isin(['dontcare'])]
output2=output1.sample(n=50, replace=True)
df.x1=df.x1.round(decimals=1)
df.y1=df.y1.round(decimals=1)
df.x2=df.x2.round(decimals=1)
df.y2=d.y2.round(decimals=1)
output2=df.append(output2,ignore_index=True)
output2.int1=output2.int1.astype(int)
output2.to_csv(os.path.join(outpath,name),sep=' ', header= False, index= False)
print("--- %s seconds ---" % (time.time() - start_time))


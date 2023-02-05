import os
import pandas as pd
import numpy as np
import time
import copy
output_proto= np.zeros((40*32,15), dtype=np.int)
output_proto = pd.DataFrame(output_proto, columns=['cls','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','e'])

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
path = '/home/lopaavol/Downloads/wenyu/code/labels1'
outpath = '/home/lopaavol/Downloads/wenyu/code/output'
os.chdir(path)
lbl_list= os.listdir(path)
##given an image, for every grid, test if the grid is overlaped with cells
#for name in lbl_list:
start_time = time.time()
name='Week1_22381_G07_04_w1.tif.txt' #this file has the average size in the folder labels1
df=pd.read_csv(name, delimiter=" ", header = None)
df.columns=['cls','x','x','x','x1','y1','x2','y2','x','x','x','x','x','x','e']
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
output2=df.append(output2,ignore_index=True)
output2.x1=output2.x1.astype(int)
output2.y1=output2.y1.astype(int)
output2.x2=output2.x2.astype(int)
output2.y2=output2.y2.astype(int)
output2.to_csv(os.path.join(outpath,name),sep=' ', header= False, index= False)
print("--- %s seconds ---" % (time.time() - start_time))

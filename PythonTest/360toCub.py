import numpy as np
from PIL import Image
import py360convert
from matplotlib import pyplot as plt
import os
for j in [1,2,3,4,5,6,7,8]:
    predataset='/home/wang/workspace/21SmartCity/scene'+str(j)
    outpath='/home/wang/workspace/cubeImg/scene'+str(j)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    fileset = os.listdir(predataset)
    imgspath=[]
    for i in fileset:
        if i[-3:] == "jpg" :
            imgspath.append(os.path.join(predataset,i))

    for i in imgspath:
        cube_dice = np.array(Image.open(i))
        cubes = py360convert.e2c(cube_dice, 1200)
        a = cubes[1200:2400, :1200]
        b = cubes[1200:2400, 1200:2400]
        c = cubes[1200:2400, 2400:3600]
        d = cubes[1200:2400, 3600:4800]
        newname = i.split('/')[-1][:-9]
        outputImgPath = os.path.join(outpath,newname+'_0.jpg')
        plt.imsave(outputImgPath,a)
        outputImgPath = os.path.join(outpath,newname+'_1.jpg')
        plt.imsave(outputImgPath,b)
        outputImgPath = os.path.join(outpath,newname+'_2.jpg')
        plt.imsave(outputImgPath,c)
        outputImgPath = os.path.join(outpath,newname+'_3.jpg')
        plt.imsave(outputImgPath,d)
        print(i)



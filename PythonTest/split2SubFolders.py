import os
from PIL import Image

predataset='/home/wang/workspace/rectilinear3'
outpath='/home/wang/workspace/rig3'


fileset = os.listdir(predataset)
imgspath=[]
for i in fileset:
    if i[-3:] == "jpg" :
        imgspath.append(os.path.join(predataset,i))

for i in imgspath:
    numsub = i.split('_')[-1][0]
    outfolder=os.path.join(outpath, 'sub'+numsub)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    newname = i.split('/')[-1][:-6]+'.jpg'
    outputImgPath = os.path.join(outfolder,newname)
    im=Image.open(i)
    im.save(outputImgPath)
    print(outputImgPath)


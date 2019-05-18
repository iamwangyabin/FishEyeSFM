import os

imgs=[]

for i in os.listdir('leftImg8bit/val'):
    for j in os.listdir('leftImg8bit/val'+'/'+i):
        imgs.append(j)

imgsVal=[]
for i in os.listdir('gtFine/val'):
    for j in os.listdir('gtFine/val'+'/'+i):
        temp=j.split('_')
        if temp[-1] == 'labelTrainIds.png':
            tempname=temp[0]+'_'+temp[1]+'_'+temp[2]+'_leftImg8bit.png'
            if tempname in imgs:
                imgsVal.append(temp[0]+'/'+temp[0]+'_'+temp[1]+'_'+temp[2])

u=imgsVal

fileObject = open('val.list', 'w')
for i in u:
    tem='leftImg8bit/val/'+i+'_leftImg8bit.png gtFine/val/'+i+'_gtFine_labelTrainIds.png'
    fileObject.write(tem)
    fileObject.write('\n')

fileObject.close()

# 下面是训练集

imgs=[]

for i in os.listdir('leftImg8bit/train'):
    for j in os.listdir('leftImg8bit/train'+'/'+i):
        imgs.append(j)

imgsVal=[]
for i in os.listdir('gtFine/train'):
    for j in os.listdir('gtFine/train'+'/'+i):
        temp=j.split('_')
        if temp[-1] == 'labelTrainIds.png':
            tempname=temp[0]+'_'+temp[1]+'_'+temp[2]+'_leftImg8bit.png'
            if tempname in imgs:
                imgsVal.append(temp[0]+'/'+temp[0]+'_'+temp[1]+'_'+temp[2])

u=imgsVal

fileObject = open('train.list', 'w')
for i in u:
    tem='leftImg8bit/train/'+i+'_leftImg8bit.png gtFine/train/'+i+'_gtFine_labelTrainIds.png'
    fileObject.write(tem)
    fileObject.write('\n')

fileObject.close()


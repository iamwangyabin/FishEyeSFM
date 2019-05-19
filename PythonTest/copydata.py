import os
from PIL import Image

def parse(dataset,output):
    for i in os.listdir(dataset):
        for j in os.listdir(os.path.join(dataset,i)):
            filepath = os.path.join(dataset,i,j,'thumbnail.jpg')
            outputpath = os.path.join(output, j+'.jpg')
            im=Image.open(filepath)
            im.save(outputpath)

if __name__ == "__main__":
    dataset='./data'
    output='./output'
    parse(dataset,output)
import json
import csv

with open("sfm-data.json","r", encoding="utf-8") as f:
    data = {}
    cc = json.loads(f.read())
    for i in cc['views']:
        data[i['value']['ptr_wrapper']['data']['filename']] = cc['extrinsics'][i['key']]['value']

csv_file = csv.reader(open('rui_an_20190420_coordinates.csv','r',encoding='gbk'))


csv_data=[]
for stu in csv_file:
    csv_data.append(stu)

import numpy as np

# for i in range(len(data)+1):
#     a=np.array(csv_data[i+1][1:],dtype=np.float)
#     try:
#         b=np.array(data[csv_data[i+1][0]+'.jpg']['center'],dtype=np.float)
#         print(a/b)
#         # print(b)
#         print(" ")
#     except:
#         pass

aa=[]
bb=[]
name=[]
for i in range(len(csv_data)):
    if (csv_data[i][0]+'.jpg') in data:
        a=np.array(csv_data[i][1:],dtype=np.float)
        b=np.array(data[csv_data[i][0]+'.jpg']['center'],dtype=np.float)
        aa.append(a)
        bb.append(b)
        name.append(csv_data[i][0])

aa=np.array(aa)
bb=np.array(bb)

import transformations as tf

def align_reconstruction_naive_similarity(X, Xp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    # Compute similarity Xp = s A X + b
    T = tf.superimposition_matrix(X.T, Xp.T, scale=True)
    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b

s, A, b=align_reconstruction_naive_similarity(bb, aa)
new_b=s*A.dot(bb.T).T+b

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show(aa,bb,name):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in range(len(aa)):
        # plot point
        ax.scatter(aa[i][0],aa[i][1],aa[i][2], c='y')
        ax.scatter(bb[i][0],bb[i][1],bb[i][2], c='r')
        # plot line
        x=np.array([aa[i][0],bb[i][0]])
        y=np.array([aa[i][1],bb[i][1]])
        z=np.array([aa[i][2],bb[i][2]])
        ax.plot(x,y,z,c='b')
        # # plot text
        # label = '%s' % (name[i])
        # ax.text(bb[i][0], bb[i][1], bb[i][2], label, color='red')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

    

show(aa,new_b,name)



data_name = sorted(list(data.keys()))
pose_data={}
for name in data_name:
    pose = data[name]['center']
    pose_data[name] = pose

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# show line in sfm result
# data: dict
# name: data.key
def showSFMresult(data1,name):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    x = []
    y = []
    z = []
    for key in name:
        value=data1[key]
        x.append( value[0] )
        y.append( value[1] )
        z.append( value[2] )
    ax.plot(x,y,z)
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

showSFMresult(pose_data,data_name)

## name:list
##  从真实值找匹配值，并在找不到的地方放弃
def getGroundAlian(name):
    groundtruth={}
    f = open("./groundtruth.txt")
    for line in f:
        if line[0] == '#':
            continue
        data_ = line.split()
        groundtruth[data_[0][:13]]=[float(data_[1]), float(data_[2]), float(data_[3])]
        print(data_[0][:13])
    valueG={}
    new_name=[]
    for key in name:
        try:
            valueG[key]=groundtruth[key[:13]]
            new_name.append(key)
        except:
            print(key)
    return valueG,new_name

ground,new_name=getGroundAlian(data_name)



def Change2Point(data,name):
    points=[]
    for i in name:
        point=data[i]
        points.append(point)
    points=np.array(points)
    return points

LFPoints=Change2Point(pose_data,new_name)
GroundPoints=Change2Point(ground,new_name)

import transformations as tf

def align_reconstruction_naive_similarity(X, Xp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    # Compute similarity Xp = s A X + b
    T = tf.superimposition_matrix(X.T, Xp.T, scale=True)
    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b

####### 
s, A, b=align_reconstruction_naive_similarity(LFPoints, GroundPoints)

new_LFPoints=s*A.dot(LFPoints.T).T+b


# just show points get from above
def show(data1,data2):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    x=data1[:,0]
    y=data1[:,1]
    z=data1[:,2]
    ax.plot(x,y,z,c='b')
    x=data2[:,0]
    y=data2[:,1]
    z=data2[:,2]
    ax.plot(x,y,z,c='r')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


show(new_LFPoints,GroundPoints)

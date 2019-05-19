import json
import csv

with open("sfm-data.json","r", encoding="utf-8") as f:
    data = {}
    aa = json.loads(f.read())
    for i in aa['views']:
        data[i['value']['ptr_wrapper']['data']['filename']] = aa['extrinsics'][i['key']]['value']

csv_file = csv.reader(open('rui_an_20190420_coordinates.csv','r',encoding='gbk'))


csv_data=[]
for stu in csv_file:
    csv_data.append(stu)

import numpy as np

for i in range(len(data)+1):
    a=np.array(csv_data[i+1][1:],dtype=np.float)
    try:
        b=np.array(data[csv_data[i+1][0]+'.jpg']['center'],dtype=np.float)
        print(a)
        print(b)
        print(" ")
    except:
        pass

def show(aa,bb):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in range(len(aa)):
        ax.scatter(aa[i][0],aa[i][1],aa[i][2], c='y')
        ax.scatter(bb[i][0],bb[i][1],bb[i][2], c='r')
        x=np.array([aa[i][0],bb[i][0]])
        y=np.array([aa[i][1],bb[i][1]])
        z=np.array([aa[i][2],bb[i][2]])
        ax.plot(x,y,z,c='b')
        # print(x)
        # print(" ")
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

show(aa,new_b)

aa=[]
bb=[]
for i in range(len(data)+1):
    a=np.array(csv_data[i+1][1:],dtype=np.float)
    try:
        b=np.array(data[csv_data[i+1][0]+'.jpg']['center'],dtype=np.float)
        aa.append(a)
        bb.append(b)
    except:
        pass

aa=np.array(aa)
bb=np.array(bb)






def align_reconstruction_naive_similarity(X, Xp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    # Compute similarity Xp = s A X + b
    T = tf.superimposition_matrix(X.T, Xp.T, scale=True)
    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b

s, A, b=align_reconstruction_naive_similarity(aa, bb)
new_b=s*A.dot(bb.T).T+b
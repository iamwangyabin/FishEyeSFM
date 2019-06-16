import json
import csv

with open(r"D:\newSmartCity\recon2\sfm_data.json","r") as f:
    data = {}
    cc = json.loads(f.read())
    for i in cc['extrinsics']:
        filename=cc['views'][i['key']]['value']['ptr_wrapper']['data']['filename']
        data[filename] = i['value']

csv_file = csv.reader(open(r'D:\SmartCity\scene1_jiading_lib_training\scene1_jiading_lib_training_coordinates.csv','r',encoding='utf-8'))

csv_data=[]
for stu in csv_file:
    csv_data.append(stu)

csv_data = csv_data[1:]

data_train={}
for i in csv_data:
    if (i[0]+'_pano.jpg') in data:
        data_train[i[0]+'.jpg'] = data[i[0]+'_pano.jpg']

import numpy
import numpy as np
import math

def quaternion_matrix(quaternion):
    # Return homogeneous rotation matrix from quaternion.
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -numpy.mean(v0, axis=1)
    M0 = numpy.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -numpy.mean(v1, axis=1)
    M1 = numpy.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = numpy.concatenate((v0, v1), axis=0)
        u, s, vh = numpy.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = numpy.dot(C, numpy.linalg.pinv(B))
        t = numpy.concatenate((t, numpy.zeros((ndims, 1))), axis=1)
        M = numpy.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= numpy.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = numpy.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = numpy.linalg.eigh(N)
        q = V[:, numpy.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # move centroids back
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M

def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    # Return matrix to transform given 3D point set into second point set.
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:3]
    return affine_matrix_from_points(v0, v1, shear=False,
                                     scale=scale, usesvd=usesvd)

def align_reconstruction_naive_similarity(X, Xp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    # Compute similarity Xp = s A X + b
    T = superimposition_matrix(X.T, Xp.T, scale=True)
    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b

aa=[]
bb=[]
name=[]
for i in range(len(csv_data)):
    if (csv_data[i][0]+'.jpg') in data_train:
        a=np.array(csv_data[i][1:],dtype=np.float)
        b=np.array(data_train[csv_data[i][0]+'.jpg']['center'],dtype=np.float)
        aa.append(a)
        bb.append(b)
        name.append(csv_data[i][0])


aa=np.array(aa)
bb=np.array(bb)

s, A, b=align_reconstruction_naive_similarity(bb, aa)
new_b=s*A.dot(bb.T).T+b

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show1(aa,bb,name):
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

show1(aa,new_b,name)


cc=[]

data_name = data.keys()
for i in data_name:
    c =np.array(data[i]['center'],dtype=np.float)
    cc.append(c)

cc=np.array(cc)

new_c=s*A.dot(cc.T).T+b


def show(aa):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in range(len(aa)):
        # plot point
        ax.scatter(aa[i][0],aa[i][1],aa[i][2], c='y')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

# #show(new_c)
# train_img = list()
# for img in csv_data:
#     train_img.append(img[0])
# with open('./results.txt','w') as f:
#     for i,img in enumerate(data_name):
#         name = img.split('.')[0]
#         if name in train_img:
#             f.write(name+',%.4f'%new_c[i][0]+',%.4f'%new_c[i][1]+',%.4f'%new_c[i][2]+'\n')
#
#













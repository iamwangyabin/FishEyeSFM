import json
import csv
import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from affineTrans import test,affine_fit

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

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def error_test(world_coord, new_b, scene):
    x_error = np.sum(np.abs(world_coord[:,0]-new_b[:,0]))
    y_error = np.sum(np.abs(world_coord[:,1]-new_b[:,1]))
    z_error = np.sum(np.abs(world_coord[:,2]-new_b[:,2]))

    num = world_coord.shape[0]
    DRMS = np.sqrt(np.sum(np.square(new_b[:, 0] - world_coord[:, 0]) + np.square(new_b[:, 1] - world_coord[:, 1])) / num)
    DHerr = 20 * np.log10((100 / (DRMS + 0.001)))
    EV = np.sqrt(np.sum(np.abs(new_b[:, 2] - world_coord[:, 2])) / num)
    DVerr = 25 * np.log10(10 / (EV + 0.001))
    # pdb.set_trace()

    with open(score_path,'a') as f:
        f.write('scene'+str(scene)+' score:\n')
        f.write('x error : '+str(x_error)+'\n')
        f.write('y error : '+str(y_error)+'\n')
        f.write('z error : '+str(z_error)+'\n')
        f.write('DHerr : ' + str(DHerr)+'\n')
        f.write('DVerr : ' + str(DVerr)+'\n\n')

def np2dic(testNAME, testIMG):
    testdic = dict()
    for i,name in enumerate(testNAME):
        if name not in list(testdic.keys()):
            testdic[name]=dict()
            testdic[name]['num']=testIMG[i]
            testdic[name]['k']=1
        else:
            testdic[name]['num']+=testIMG[i]
            testdic[name]['k']+=1
    for n in list(testdic.keys()):
        testdic[n]['r']=testdic[n]['num']/testdic[n]['k']
    return testdic

def getGT(gt_path):
    try:
        csv_file = csv.reader(open(gt_path, 'r', encoding='utf-8'))
        csv_data = []
        gt_name = []
        for s in csv_file:
            csv_data.append(s)
            gt_name.append(s[0])
    except:
        csv_file = csv.reader(open(gt_path, 'r', encoding='gbk'))
        csv_data = []
        gt_name = []
        for s in csv_file:
            csv_data.append(s)
            gt_name.append(s[0])
    csv_data = csv_data[1:]
    gt_name = gt_name[1:]
    gt_dict = {}
    for i in csv_data:
        if i[0] not in gt_dict.keys():
            gt_dict[i[0]] = [float(i[1]), float(i[2]), float(i[3])]
    return gt_dict


def show2(aa,bb):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in range(len(bb)):
        # plot point
        ax.scatter(aa[i][0],aa[i][1],aa[i][2], c='y')
        ax.scatter(bb[i][0],bb[i][1],bb[i][2], c='r')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


def affine_res(M, testR):
    resdim = testR[0].shape[0]
    testr = []
    for d in range(testR.shape[0]):
        pt = testR[d]
        res = [0.0 for a in range(resdim)]
        for j in range(resdim):
            for i in range(resdim):
                res[j] += pt[i] * M[i][j + resdim + 1]
            res[j] += M[resdim][j + resdim + 1]
        testr.append(res)
    testr = np.array(testr)
    return testr

def main(gt_path, save_path, data_path, scene, method):
    # ============================  gt =============================
    gt_dict = getGT(gt_path)
    # ===============================================================
    t = list()
    with open(data_path,'r') as f:
        [t.append(line) for line in f.read().splitlines()]
    t = t[4:]
    assert len(t)%2==0

    results = list()
    world_coord = list()
    testIMG = list()
    testNAME = list()
    trainNAME = list()
    for i in range(int(len(t)/2)):
        c = t[i*2].split(' ')
        quat = np.array((float(c[1]),float(c[2]),float(c[3]),float(c[4])))
        rot_matrix = qvec2rotmat(quat)
        r = rot_matrix[:3,:3]
        T = np.array([[float(c[5]),float(c[6]),float(c[7])]]).T
        x,y,z = -r.T.dot(T)

        name = c[-1][:-6]
        # name = c[-1][:-11]
        # print(name)
        import pdb
        # pdb.set_trace()
        # print(gt_dict.keys())
        if name in list(gt_dict.keys()):
            results.append([x[0],y[0],z[0]])
            world_coord.append(gt_dict[name])
            trainNAME.append(name)
        else:
            testIMG.append([x[0],y[0],z[0]])
            testNAME.append(name)
    for i in list(gt_dict.keys()):
        if i not in trainNAME:
            print('gt {} not in data'.format(i))
    #print(gt_dict.keys())
    #print(trainNAME)
    clound_coord = np.array(results)
    world_coord = np.array(world_coord)
    testIMG = np.array(testIMG)
# ========================= mean the results to test==============================
    if method==2 or method==3:
        traindic = np2dic(trainNAME, clound_coord)
        gtdic = np2dic(trainNAME, world_coord)
        for i,p in enumerate(traindic):
            if i==0:
                trainR = traindic[p]['r']
                gtR = gtdic[p]['r']
            else:
                trainR = np.concatenate([trainR,traindic[p]['r']],0)
                gtR = np.concatenate([gtR, gtdic[p]['r']], 0)
        trainR=trainR.reshape((-1,3))
        gtR=gtR.reshape((-1,3))
# =================================================================================
    #show(results)
# ================== 3 way(all / mean / affine trans) to test======================
    if method==1:
        s, A, b = align_reconstruction_naive_similarity(clound_coord, world_coord)
        clound2world=s*A.dot(clound_coord.T).T+b
        error_test(world_coord, clound2world, scene)

    elif method==2:
        s, A, b = align_reconstruction_naive_similarity(trainR, gtR)
        train2gt=s*A.dot(trainR.T).T+b
        error_test(gtR, train2gt, scene)

    elif method==3:
        M, train2gt = test(trainR,gtR)
        train2gt_v2 = affine_res(M, trainR)
        assert train2gt.all() ==train2gt_v2.all()
        error_test(gtR, train2gt, scene)
        # pdb.set_trace()
        # return train2gt, trainNAME

    elif method==4:
        M, train2gt = test(clound_coord, world_coord)
        train2gt_v2 = affine_res(M, clound_coord)
        assert train2gt.all() ==train2gt_v2.all()
        error_test(world_coord, train2gt, scene)

    else:
        raise KeyError('error methods')

    testdic = np2dic(testNAME, testIMG)

    with open(save_path,'w') as f:
        dicname = []
        for i,p in enumerate(testdic):
            dicname.append(p)
            if i == 0:
                testR = testdic[p]['r']
            else:
                testR = np.concatenate([testR, testdic[p]['r']], 0)
        testR = testR.reshape((-1,3))
        # ==================  default or affine =====================
        if method==1 or method==2:
            testr = s*A.dot(testR.T).T+b
        if method==3 or method==4:
            testr = affine_res(M, testR)
            # pdb.set_trace()
        # ===========================================================
        for i,dn in enumerate(dicname):
            f.write(dn+',%.4f'%testr[i][0]+',%.4f'%testr[i][1]+',%.4f'%testr[i][2]+'\n')


global score_path
score_path = '/home/wang/workspace/Results/voca1/score.txt'
if __name__=='__main__':
    import os
    import pdb

    if os.path.exists(score_path):
        os.system('rm '+score_path)
    trains=[]
    for s in [5]:
        save_path = '/home/wang/workspace/Results/voca1/scene'+str(s)+'.txt'
        gt_path = '/home/wang/workspace/FishEyeSFM/PythonTest/scene'+str(1)+'.csv'
        data_path = '/home/wang/workspace/Results/voca1/images.txt'
        main(gt_path, save_path ,data_path ,s, method=3)
        # trains = trains + train.tolist()
    # gt_path = '/home/wang/workspace/FishEyeSFM/PythonTest/scene' + str(6) + '.csv'
    # gt_dict = getGT(gt_path)
    # # with open(os.path.join('/home/wang/workspace/Finalresult/', 'scene6.txt'), 'r') as f:
    # #     datas = [line.split(',') for line in f.read().splitlines()]
    # # data={}
    # # for i in datas:
    # #     data[i[0]] = [float(i[1]), float(i[2]), float(i[3])]
    # aa=[]
    # bb=[]
    # for i in gt_dict.keys():
    #     bb.append(gt_dict[i])
    # trains = np.array(trains)
    # trains = trains.tolist()
    # # pdb.set_trace()
    # show2(trains,bb )

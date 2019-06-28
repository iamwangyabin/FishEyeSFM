import numpy as np

def affine_fit(from_pts, to_pts):
    q = from_pts  
    p = to_pts  
    assert len(q) == len(p)
    assert len(q) > 1
    dim = len(q[0])  # 维度
    assert len(q) > dim

    # 新建一个空的 维度 x (维度+1) 矩阵 并填满  
    c = [[0.0 for a in range(dim)] for i in range(dim+1)]  
    for j in range(dim):  
        for k in range(dim+1):  
            for i in range(len(q)):  
                qt = list(q[i]) + [1]  
                c[k][j] += qt[k] * p[i][j]  

    # 新建一个空的 (维度+1) x (维度+1) 矩阵 并填满  
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]  
    for qi in q:  
        qt = list(qi) + [1]  
        for i in range(dim+1):  
            for j in range(dim+1):  
                Q[i][j] += qt[i] * qt[j]  

    # 判断原始点和目标点是否共线，共线则无解. 耗时计算，如果追求效率可以不用。  
    # 其实就是解n个三元一次方程组  
    def gauss_jordan(m, eps=1.0/(10**10)):  
        (h, w) = (len(m), len(m[0]))  
        for y in range(0, h):  
            maxrow = y  
            for y2 in range(y+1, h):      
                if abs(m[y2][y]) > abs(m[maxrow][y]):  
                    maxrow = y2  
            (m[y], m[maxrow]) = (m[maxrow], m[y])  
            if abs(m[y][y]) <= eps:       
                return False  
            for y2 in range(y+1, h):      
                c = m[y2][y] / m[y][y]  
                for x in range(y, w):  
                    m[y2][x] -= m[y][x] * c  
        for y in range(h-1, 0-1, -1):    
            c = m[y][y]  
            for y2 in range(0, y):  
                for x in range(w-1, y-1, -1):  
                    m[y2][x] -= m[y][x] * m[y2][y] / c  
            m[y][y] /= c  
            for x in range(h, w):         
                m[y][x] /= c  
        return True

    M = [Q[i] + c[i] for i in range(dim+1)]

    if not gauss_jordan(M):  
        return False

    class transformation:  
        """对象化仿射变换."""  
        def To_Str(self):  
            res = ""  
            for j in range(dim):  
                str = "x%d' = " % j  
                for i in range(dim):  
                    str +="x%d * %f + " % (i, M[i][j+dim+1])  
                str += "%f" % M[dim][j+dim+1]  
                res += str + "\n"  
            return res  
  
        def transform(self, pt):
            res = [0.0 for a in range(dim)]  
            for j in range(dim):  
                for i in range(dim):  
                    res[j] += pt[i] * M[i][j+dim+1]  
                res[j] += M[dim][j+dim+1]  
            return res  
    return transformation(), M
  
def test(from_pt, to_pt):  
    trn, M = affine_fit(from_pt, to_pt)
    result = []
    if trn:
        #print(trn.To_Str())
        err = 0.0  
        for i in range(len(from_pt)):  
            fp = from_pt[i]  
            tp = to_pt[i]  
            t = trn.transform(fp)
            result.append(t)
            err += ((tp[0] - t[0])**2 + (tp[1] - t[1])**2)**0.5
        print("拟合误差 = %f" % err)
    return M, np.array(result)
  
if __name__ == "__main__":  
    test()  

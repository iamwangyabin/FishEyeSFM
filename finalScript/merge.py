import os

def sort(datapath):
    with open(os.path.join(datapath, 'results.txt'), 'r') as f:
        datas = [line.split(',') for line in f.read().splitlines()]
    with open('fileorder.txt', 'r') as f:
        target = [line.split(',') for line in f.read().splitlines()]
    with open(r'c.csv', 'r') as f:
        cnn = [line.split(',') for line in f.read().splitlines()]
    ids = list()
    for t in target:
        ids.append(t[0])
    result = dict()
    for d in datas:
        result[d[0]] = d[1:]
    error = dict()
    for d in cnn:
        error[d[0]] = d[1:]
    a = list()
    for t in ids:
        if t in result:
            for i in result[t]:
                t = t + ',' + i
        else:
            for i in error[t]:
                t = t + ',' + i
        a.append(t)
    with open(os.path.join(datapath, 'c.csv'), 'w') as f:
        for i in a:
            f.write(i + '\n')

if __name__ == "__main__":
    # mergeResults(r'C:\Users\ZZBZ\Desktop\result')
    sort(r'/home/wang/workspace/Finalresult')
import numpy as np
import matplotlib.pyplot as pt
from scipy import signal
from sklearn.mixture import GMM
from os import listdir
from os.path import isfile, join
import math as mt
from sgfilter import savitzky_golay
import itertools



f_names = ['hips xp','hips yp','hips zp','hips zr','hips xr','hips yr',\
           'LeftHip xp','LeftHip yp','LeftHip zp','LeftHip zr','LeftHip xr','LeftHip yr',
           'LeftKnee xp','LeftKnee yp','LeftKnee zp','LeftKnee zr','LeftKnee xr','LeftKnee yr',\
           'LeftAnkle xp','LeftAnkle yp','LeftAnkle zp','LeftAnkle zr','LeftAnkle xr','LeftAnkle yr',\
           'RightHip xp', 'RightHip yp','RightHip zp','RightHip zr','RightHip xr','RightHip yr',\
           'RightKnee xp','RightKnee yp','RightKnee zp', 'RightKnee zr', 'RightKnee xr', 'RightKnee yr',\
           'RightAnkle xp','RightAnkle yp','RightAnkle zp','RightAnkle zr','RightAnkle xr','RightAnkle yr',\
           'Chest xp','Chest yp','Chest zp','Chest zr','Chest xr','Chest yr',\
           'Chest2 xp','Chest2 yp','Chest2 zp','Chest2 zr','Chest2 xr','Chest2 yr',\
           'LeftCollar xp','LeftCollar yp','LeftCollar zp','LeftCollar zr','LeftCollar xr','LeftCollar yr',\
           'LeftShoulder xp','LeftShoulder yp','LeftShoulder zp','LeftShoulder zr','LeftShoulder xr','LeftShoulder yr',\
           'LeftElbow xp','LeftElbow yp','LeftElbow zp','LeftElbow zr','LeftElbow xr','LeftElbow yr',\
           'LeftWrist xp','LeftWrist yp','LeftWrist zp','LeftWrist zr','LeftWrist xr','LeftWrist yr',\
           'RightCollar xp','RightCollar yp','RightCollar zp','RightCollar zr','RightCollar xr','RightCollar yr',\
           'RightShoulder xp','RightShoulder yp','RightShoulder zp','RightShoulder zr','RightShoulder xr','RightShoulder yr',\
           'RightElbow xp','RightElbow yp','RightElbow zp','RightElbow zr','RightElbow xr','RightElbow yr',\
           'RightWrist xp','RightWrist yp','RightWrist zp','RightWrist zr','RightWrist xr','RightWrist yr',\
           'Neck xp','Neck yp','Neck zp','Neck zr','Neck xr','Neck yr',\
           'Head xp','Head yp','Head zp','Head zr','Head xr','Head yr',
           ]

rotation = 0

def isNoise(value):
    return abs(sum(value)) < 1


def getHammMatrix(total,mean_list,n_features):
    mat = np.zeros((n_features,n_features), dtype=np.float64)
    for i in range(len(total)):
        for l in total[i]:
            mat[l,total[i]] = 1        
    return mat


def printMocapDataMotion(motion):
    w = round(np.sqrt(len(motion)),0)
    h = int(len(motion)/w+1)
    pt.figure()
    for i in range(len(motion)):
        sp = pt.subplot(w,h,i+1)
        m = savitzky_golay(motion[i],131,3)
        pt.plot(m)
        sp.set_title(f_names[i])
    pt.show()
    return None



def printMotion(motion):
    w = round(np.sqrt(len(motion)),0)
    h = int(len(motion)/w+1)
    pt.figure()
    for i in range(len(motion)):
        pt.subplot(w,h,i+1)
        m = savitzky_golay(motion[i],51,3)
        pt.plot(m)
        
    pt.show()

    return None


def getOneMocapData(file):
    line = file.readline().rstrip()
    while line != 'MOTION':
        line = file.readline().rstrip()
    line = file.readline()
    line = file.readline()
        
    motionVal = np.zeros((0,0),dtype=np.float64)
    line = file.readline()
    while line:
        line = line.rstrip().split()
        line = [float(x) for x in line]  
        if rotation: 
            line = list([line[i*3],line[i*3+1],line[i*3+2]] for i in range(19))
        else: 
            line = list([line[i*3+3],line[i*3+4],line[i*3+5]] for i in range(19))
        line = list(itertools.chain(*line)) 
        
        #print('len(line)',len(line))
        val = np.array(line)
        if not len(motionVal): motionVal = val
        else:
            if len(val): motionVal = np.vstack([motionVal,val])
            else: break
        line = file.readline()
        line = file.readline()
        line = file.readline()
        line = file.readline()
        
        
    motionVal = np.transpose(motionVal)
    fft_len = 15
    fft = np.empty((0,fft_len*2),dtype=np.float64)
    if not len(motionVal): return [],[]
    
    diffs = np.empty((0,len(motionVal[0])-1),dtype=np.float64)
    motionVal1 = np.empty((0,len(motionVal[0])),dtype=np.float64)
    #print('len(motionVal)',len(motionVal))

    for i in range(len(motionVal)):
        m = motionVal[i]
        m = savitzky_golay(m,13,3)
        m = np.diff(m)
        m = savitzky_golay(m,13,3)
        if max(m)-min(m) > 0: m = m / (max(m)-min(m))
        diffs = np.append(diffs,np.array([m]),axis=0)
        motionVal1 = np.append(motionVal1,motionVal[i].reshape(1,len(motionVal[i])),axis=0)
        m = np.fft.fft(m)
        m = m[:fft_len]
        # interlacing the real coefficients and the imaginary coefficients
        a = np.real(m)
        b = np.imag(m)
        m = np.empty((a.size + b.size,), dtype = a.dtype)
        m[0::2] = a
        m[1::2] = b
        #m = a
        #if (max(m)-min(m)) != 0:
        #    m = m / (max(m) - min(m))
        fft = np.append(fft, np.array([m]),axis=0)
            
    file.close()
    #print('len(fft)',len(fft))
    #fft = np.delete(fft,[31,32,33,44,45,46,50,59,61,73,74,75],axis=0)
    #diffs = np.delete(diffs,[31,32,33,44,45,46,50,59,61,73,74,75],axis=0)
    #motionVal1 = np.delete(motionVal1,[31,32,33,44,45,46,50,59,61,73,74,75],axis=0)
    #printMocapDataMotion(motionVal1)
    return fft, diffs

    
    
def getMocapDataMotion():
    dirlist = ['../Data/MocapData/Walk/','../Data/MocapData/Jump/']
    total_mat_list = []
    total_diff_list = []
    for dirname in dirlist:
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        mat_list = []
        diff_list = []
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                print(filename)
                file = open(filename,"r")
                X, diffs = getOneMocapData(file)
                if not len(X) or not len(diffs): continue
                mat_list.append(X)
                diff_list.append(diffs)
        
        total_mat_list.append(mat_list)
        total_diff_list.append(diff_list)      
    
    return total_mat_list, total_diff_list


def getMocapDataEachMotion():
    dirname = '../Data/MocapData/Jump/'
 
    files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    mat_list = []
    diff_list = []
    for f in files:
        if '.txt' in f:
            filename = dirname + f
            print(filename)
            file = open(filename,"r")
            X, diffs = getOneMocapData(file)
            if not len(X) or not len(diffs): continue
            mat_list.append(X)
            diff_list.append(diffs)
    
    return mat_list, diff_list




def getAllMocapDataMotions(dir_list):
    total_corr_mat = []
    for dirname in dir_list:
        
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        corr_mat = []
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                file = open(filename,"r")
                X = getMSRMotion(file,0)
                corr_mat.append(X)
        total_corr_mat.append(corr_mat)

    return total_corr_mat




def printMotions(dirname):
    files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    for f in files:
        if '.txt' in f:
            filename = dirname + f
            file = open(filename,"r")
            X = getMSRMotion(file,0)
    return




def printFeature(dirname, ind):
    files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    files = list(f for f in files if '.txt' in f)
    w = round(np.sqrt(len(files)),0)
    h = int(len(files)/w+1)
    
    pt.figure()
    for f in files:
        if '.txt' in f:
            filename = dirname + f
            file = open(filename,"r")
            x = getMSRMotionFeature(file,ind)
            pt.subplot(h,w,files.index(f)+1)
            pt.plot(x)
    pt.show()
    return




def getMotionCorr(fft):
    n = fft.shape[0]
    mats = np.zeros((n,n),dtype=np.float64)
    for p in range(n):
        for q in range(p+1,n):
            corr = np.corrcoef(fft[p],fft[q])[1,0]
            if np.isnan(corr): mats[[p,q],[q,p]] = 0
            else: mats[[p,q],[q,p]] = corr

    return mats
    

#file = open('../Data/MocapData/Walk/walk-27-thinking-azumi.txt')

def tmp():
    if 0:
        ffts,diffs = getMocapDataEachMotion()
        total_list = []
        for fft in ffts:
            mat = getMotionCorr(fft)
            var_list = []
            for i in range(len(mat)):
                var_list.append(np.var(mat[i,:]))
            th = max(var_list) * 0.9
            l1 = list((var_list.index(x),x) for x in var_list if x > th)
            l2 = sorted(l1, key = lambda x: x[1],reverse=True)
            l3 = list(x[0] for x in l2)
            total_list += l3
        final = list((x,total_list.count(x)) for x in set(total_list))
        final = sorted(final, key = lambda x: x[1],reverse=True)
        final = list(x[0] for x in final)
        print(final)
        
    else: 
        fft,diff = getOneMocapData(file)
        mat = getMotionCorr(fft)
        var_list = []
        for i in range(len(mat)):
            var_list.append(np.var(mat[i,:]))
        th = max(var_list) * 0.9
        l1 = list((var_list.index(x), x) for x in var_list if x > th)
        l2 = sorted(l1, key = lambda x: x[1],reverse=True)
        l3 = list(x[0] for x in l2)
        print('l3', l3)
        return mat
#mat = tmp()

#w,v=np.linalg.eig(mat)
#pt.figure()
#print(w)
#print(v)
#pt.imshow(mat)
#pt.show()



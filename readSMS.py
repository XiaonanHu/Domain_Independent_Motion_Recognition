import numpy as np
import matplotlib.pyplot as pt
from scipy import signal
from sklearn.mixture import GMM
from os import listdir
from os.path import isfile, join
import math as mt
from sgfilter import savitzky_golay
import itertools




f_names = ['hips xp', 'hips yp', 'hips zp', 'hips zr', ' hips xr', 'hips yr',\
           'chest z', 'chest x', 'chest y', 'neck z', 'neck x', 'neck y',\
           'head z', 'head x', 'head y', 'leftCollar z', 'leftCollar x',\
           'leftCollar y', 'leftUpArm z', 'leftUpArm x', 'leftUpArm y',\
           'leftLowArm z', 'leftLowArm x', 'leftLowArm y', 'leftHand z',\
           'leftHand x', 'leftHand z', 'rightCollar z', 'rightCollar x',\
           'rightCollar y', 'rightUpArm z', 'rightUpArm x', 'rightUpArm y',\
           'rightLowArm z', 'rightLowArm x', 'rightLowArm y', 'rightHand z',\
           'rightHand x', 'rightHand y', 'leftUpLeg z', 'leftUpLeg x',\
           'leftUpLeg y', 'leftLowLeg z', 'leftLowLeg x', 'leftLowLeg y',\
           'leftFoot z', 'leftFoot x', 'leftFoot y', 'rightUpLeg z',\
           'rightUpLeg x', 'rightUpLeg y', 'rightLowLeg z', 'rightLowLeg x',\
           'rightLowLeg y', 'rightFoot z', 'rightFoot x', 'rightFoot y', 
]


def isNoise(value):
    return abs(sum(value)) < 1


def getHammMatrix(total,mean_list,n_features):
    mat = np.zeros((n_features,n_features), dtype=np.float64)
    for i in range(len(total)):
        for l in total[i]:
            mat[l,total[i]] = 1        
    return mat


def printSMSMotion(motion):
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


def skipLines(file):
    #for _ in range(3): line = file.readline()
    line = file.readline()
    return line

def getOneSMS(file):
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
        val = np.array(line)
        if not len(motionVal): motionVal = val
        else:
            if len(val): motionVal = np.vstack([motionVal,val])
            else: break
        line = skipLines(file)
        
    motionVal = np.transpose(motionVal)
    fft_len = 31
    fft = np.empty((0,fft_len*2),dtype=np.float64)
    if not len(motionVal): return [],[]
    
    diffs = np.empty((0,len(motionVal[0])-2),dtype=np.float64)
    motionVal1 = np.empty((0,len(motionVal[0])),dtype=np.float64)
    
    
    for i in range(len(motionVal)):
        m = motionVal[i][1:]
        #m = savitzky_golay(m,13,3)
        m = savitzky_golay(m,45,3)
        m = np.diff(m)
        #if i not in [27,28,29,33,34,45,46,47,51,52,61,66,67,78,79,84,85]:
        #m = savitzky_golay(m,13,3)
        m = savitzky_golay(m,45,3)
        
        diffs = np.append(diffs,np.array([m]),axis=0)
        motionVal1 = np.append(motionVal1,motionVal[i].reshape(1,len(motionVal[i])),axis=0)
        m = np.fft.fft(m)
        m = m[:fft_len]
        # interlacing the real coefficients and the imaginary coefficients
        a = np.real(m)
        b = np.imag(m)
        m = np.empty((a.size + b.size,), dtype = a.dtype)
        m[0::2],m[1::2] = a,b
        fft = np.append(fft, np.array([m]),axis=0)
            
    file.close()
    #print('len(fft)',len(fft))
    #fft = np.delete(fft,[31,32,33,44,45,46,50,59,61,73,74,75],axis=0)
    #diffs = np.delete(diffs,[31,32,33,44,45,46,50,59,61,73,74,75],axis=0)
    #motionVal1 = np.delete(motionVal1,[31,32,33,44,45,46,50,59,61,73,74,75],axis=0)
    #printSMSMotion(motionVal1)
    return fft, diffs

    
    
def getSMSMotion():
    dirlist = ['../Data/SMS/WALKS/','../Data/SMS/RUNS/',\
               '../Data/SMS/JUMP/']
    # '../Data/SMS/Mocap/Bending/','../Data/SMS/Mocap/Throwing/',

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
                X, diffs = getOneSMS(file)
                if not len(X) or not len(diffs): continue
                mat_list.append(X)
                diff_list.append(diffs)
        
        total_mat_list.append(mat_list)
        total_diff_list.append(diff_list)      
    
    return total_mat_list, total_diff_list

'''
mat,diff = getSMSMotion()
indices = [[33,34,35],[21,22,23],[51,52,53],[42,43,44],[0,1,2]]
#indices = [[30,31,32],[18,19,20],[48,49,50],[39,40,41],[3,4,5]]

for k in range(10):
    for p in range(len(diff)):
        pt.figure()
        for i in range(len(indices)):
            for j in range(3):
                pt.subplot(5,3,i*3+j+1)
                #f = np.fft.fft(diff[0][k][indices[i][j]])
                #o =  np.fft.ifft()
                #f = signal.medfilt(diff[0][k][indices[i][j]],15)
                #f = f[::4]
                diff2 = np.diff(diff[p][k][indices[i][j]])
                pt.plot(diff2)
                #pt.plot(diff[0][k][indices[i][j]])
        pt.show()
        '''



def getSMSEachMotion():
    dirname = '../Data/SMS/RUNS/'
 
    files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    mat_list = []
    diff_list = []
    for f in files:
        if '.txt' in f:
            filename = dirname + f
            print(filename)
            file = open(filename,"r")
            X, diffs = getOneSMS(file)
            if not len(X) or not len(diffs): continue
            mat_list.append(X)
            diff_list.append(diffs)
    
    return mat_list, diff_list







def getAllSMSMotions(dir_list):
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
    


file = open('../Data/SMS/RUNS/run_circle1.txt')

def tmp():
    if 0:
        ffts,diffs = getSMSEachMotion()
        total_list = []
        for fft in ffts:
            mat = getMotionCorr(fft)
            var_list = []
            for i in range(len(mat)):
                var_list.append(np.var(mat[i,:]))
            th = max(var_list)*0.9
            l1 = list((var_list.index(x),x) for x in var_list if x > th)
            l2 = sorted(l1, key=lambda x: x[1],reverse=True)
            l3 = list(x[0] for x in l2)
            total_list += l3
        final = list((x,total_list.count(x)) for x in set(total_list))
        final = sorted(final,key=lambda x: x[1],reverse=True)
        final = list(x[0] for x in final)
        print(final)
        
    else: 
        fft,diff = getOneSMS(file)
        mat = getMotionCorr(fft)
        var_list = []
        for i in range(len(mat)):
            var_list.append(np.var(mat[i,:]))
        th = max(var_list)*0.9
        l1 = list((var_list.index(x),x) for x in var_list if x > th)
        l2 = sorted(l1,key=lambda x: x[1],reverse=True)
        l3 = list(x[0] for x in l2)        
        print(l3)
        return mat
    
#mat = tmp()

#w,v=np.linalg.eig(mat)
#print(w)
#print(v)
#pt.imshow(mat)
#pt.show()

# Clapping: 25, 35, 36, 24, 51, 34, 60, 25, 38, 49xw
     # right arm y, left arm y, left arm z, right arm x, right foot z, left arm x, 15->34
     # left leg roll x, right arm y, left fore arm y, right foot x 60->49
     
# Throwing: 18, 16, 11, 14, 8, 31, 15, 33, 20, 23 
     # head x, neck y, spine1 z, spine2 z, spine z, left shoulder x, neck x, 18->15
     # left shoulder z, head z, right shoulder z

# Wave two hands: 24, 26, 31, 36, 21, 34, 33, 23, 28, 22
     # right arm x, right arm z, left shoulder x, left arm z, right shoulder x, 24->21
     # left arm x, left shoulder z, right shoulder z, right fore arm y, right shoulder y
     
# Wave one hand: 21, 24, 28, 26, 23, 14, 11, 22, 8, 7
     # right shoulder x, right arm x, right fore arm y, right arm z, right shoulder z, 21->23
     # spine2 z, spine1 z, right shoulder y, spine z, spine y


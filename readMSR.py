import numpy as np
import matplotlib.pyplot as pt
from scipy import signal
from sklearn.mixture import GMM
from os import listdir
from os.path import isfile, join
import math as mt
from sgfilter import savitzky_golay


        
def isNoise(value):
    return abs(sum(value)) < 1


def getOmegaDict(total,mean_list):
    omega_dict = {}
    for i in range(len(total)):
            #omega[i,i] = 1
            if len(total[i]) in omega_dict.keys():
                omega_dict[len(total[i])].append((total[i],mean_list[i]))
            else:
                omega_dict[len(total[i])] = [(total[i],mean_list[i])]   
    return omega_dict


def getOmegaAndOmegaMatrix(total,mean_list,n_features):
    omega = np.zeros((len(total),len(total)), dtype=np.float64)    
    mat = np.zeros((n_features,n_features), dtype=np.float64)
    
    for i in range(len(total)):
        for l in total[i]:
            mat[l,total[i]] = 1        
        for j in range(i+1,len(total)):
            corr = np.corrcoef(mean_list[i],mean_list[j])[1,0]
            omega[i,j] = corr
            omega[j,i] = corr
            
            p = total[i]
            q = total[j]
            if len(p) != 1 and len(q) != 1:
                for k in p:
                    mat[k,q] = corr
    
    return omega, mat

def getHammMatrix(total,mean_list,n_features):
    mat = np.zeros((n_features,n_features), dtype=np.float64)
    
    for i in range(len(total)):
        for l in total[i]:
            mat[l,total[i]] = 1        
    
    return mat

    
def gmmCluster1(fftArrays):
    X = fftArrays
    n_features = fftArrays.shape[0]
    clusMat = np.zeros((n_features,n_features), dtype=np.int32)
    
    for p in range(10):
        for j in range(6,10):
            GM = GMM(n_components = j,covariance_type='full')
            GM.fit(X)
            l = GM.predict(X)
            for c in range(j):
                ind = np.where(l == c)[0]
                for i in range(len(ind)):
                    for k in range(i,len(ind)):
                        clusMat[ind[i]][ind[k]] += 1
               
    # find the cluster by majority vote
    n = clusMat[0][0] # the number of gmm performed
    check = []
    total = []
    mean_list = []
    for i in range(len(clusMat)):
        if i not in check:
            ind = np.where(clusMat[i] >= n*0.65) 
            print('ind is: ', ind)
            
            if type(ind[0]) == np.ndarray:
                mean = np.mean(X[ind[0],:],axis=0)
            else:
                mean = X[ind[0]]
                
            total.append(list(ind[0]))
            check += list(ind[0]) 
            mean_list.append(mean)
                        
    return total, mean_list


def printMSRMotion(motion):
    w = round(np.sqrt(len(motion)),0)
    h = int(len(motion)/w+1)
    pt.figure()
    for i in range(len(motion)):
        pt.subplot(w,h,i+1)
        m = savitzky_golay(motion[i],51,3)

        pt.plot(m)
    pt.show()

    
    return None

def getMSRMotionFeature(file,ind):
    
    motionVal = np.zeros((0,0),dtype = np.float64)
    
    line = file.readline()
    line = file.readline().rstrip().split()
    while line:
        if  line == ['40'] or line == ['80']:
            val = np.zeros((0,0),dtype = np.float64)
            line = file.readline()
            line = line.rstrip().split()
            check = (len(line) > 1)
            while (len(line) > 1):
                line = line[:3]
                line = [float(x) for x in line]
                val = np.append(val,line)
                if not 0: file.readline()
                line = file.readline().rstrip().split()
            if len(val) == 120: val = val[:60]
            val = np.reshape(val,(val.shape[0],1)) #transpose
            if not len(motionVal): motionVal = val
            else: motionVal = np.hstack([motionVal,val])
    val = motionVal[ind]
    val = savitzky_golay(val,91,3)
    return val


def getMSRAction(file):
    motionVal = np.zeros((0,0),dtype=np.float64)
    
    line = file.readline()
    while line:
        val = np.zeros((0,0),dtype=np.float64)
        
        for i in range(20):
            line = line.rstrip().split()
            line = [float(x) for x in line[:3]]
            val = np.append(val,line)
            line = file.readline()
        
        if not len(motionVal): motionVal = val
        else:
            if len(val): motionVal = np.vstack([motionVal,val])
            else: break
    motionVal = np.transpose(motionVal)
    
    #printMSRMotion(motionVal)
    
    fft_len = 15
    fft = np.empty((0,fft_len*2),dtype=np.float64)
    if not len(motionVal): return [],[]
    diffs = np.empty((0,len(motionVal[1])-1),dtype=np.float64)
    for i in range(len(motionVal)):
        m = motionVal[i]
        m = savitzky_golay(m,11,3) # 21
        m = np.diff(m)
        m = savitzky_golay(m,11,3)
        
        diffs = np.append(diffs,np.array([m]),axis=0)
        m = np.fft.fft(m)
        m = m[:fft_len]
        # interlacing the real coefficients and the imaginary coefficients
        a = np.real(m)
        b = np.imag(m)
        m = np.empty((a.size + b.size,), dtype = a.dtype)
        m[0::2],m[1::2] = a,b
        fft = np.append(fft,np.array([m]) ,axis=0)
            
    file.close()    
    
    return fft, diffs
             
            
def getOneMSR(file,both):
    
    motionVal = np.zeros((0,0),dtype = np.float64)
    
    line = file.readline()
    line = file.readline().rstrip().split()
    while line:

        if  line == ['40'] or line == ['80']:
            val = np.zeros((0,0),dtype = np.float64)
            line = file.readline()
            line = line.rstrip().split()
            check = (len(line) > 1)
            while (len(line) > 1):
                line = line[:3]
                line = [float(x) for x in line]
                val = np.append(val,line)
                
                if not both: file.readline()
                
                line = file.readline().rstrip().split()
            if len(val) == 120: val = val[:60]
            val = np.reshape(val,(val.shape[0],1)) #transpose

            
            if not len(motionVal): motionVal = val
            else: motionVal = np.hstack([motionVal,val])
            
    #printMSRMotion(motionVal)
    
    fft_len = 15
    fft = np.empty((0,fft_len*2),dtype=np.float64)
    if not len(motionVal): return [],[]
    
    diffs = np.empty((0,len(motionVal[0])-1),dtype=np.float64)
    for i in range(len(motionVal)):
        m = motionVal[i]
        m = savitzky_golay(m,21,3)
        m = np.diff(m)
        
        if i not in [24,25,36,37]:
            m = savitzky_golay(m,21,3)
            if max(m)-min(m) > 0: m = m / (max(m)-min(m))
            diffs = np.append(diffs,np.array([m]),axis=0)
            m = np.fft.fft(m)
            m = m[:fft_len]
            # interlacing the real coefficients and the imaginary coefficients
            a = np.real(m)
            b = np.imag(m)
            m = np.empty((a.size + b.size,), dtype = a.dtype)
            m[0::2] = a
            m[1::2] = b
            if (max(m)-min(m)) != 0:
                m = m / (max(m) - min(m))
            fft = np.append(fft, np.array([m]),axis=0)
            
    file.close()    
    

    return fft, diff


def getMSRMotion():
    
    dirlist = ['../Data/MSR/Wave_two_hands/','../Data/MSR/Wave_one_arm/',
               '../Data/MSR/Hand_clap/','../Data/MSR/Punching/'] #,\
               #]'../Data/MSR/Bend/','../Data/MSR/Throwing_a_ball/',
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
                X, diffs = getMSRAction(file)
                if not len(X) or not len(diffs): continue
                mat_list.append(X)
                diff_list.append(diffs)
        
        total_mat_list.append(mat_list)
        total_diff_list.append(diff_list)        

    return total_mat_list, total_diff_list

mat,diff = getMSRMotion()

pt.figure()
for i in range(60):
    pt.subplot(8,8,i+1)
    pt.plot(diff[0][0][i])
pt.show()

    


def getAllMSRMotions(dir_list):
    
    total_corr_mat = []
    for dirname in dir_list:
        
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        corr_mat = []
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                #print('filename',filename)
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




def findRelevantFeatures():
    dirname = '../Data/MSR/MSR_dailyActivity2/'
    motions = getAllMSRMotions([dirname])
    motions = motions[0]
    motions1 = motions[:20]
    motions2 = motions[20:]
    
    featureVarList = []
    for i in range(len(motions1[0])): # for each feature
        features = []
        for j in range(len(motions1)):
            features.append(motions1[j][i])
        featureVarList.append(sum(np.var(features,0)))
    
    print(featureVarList)
    s = sorted(featureVarList)
    c = list(featureVarList.index(x) for x in s)
    print(c)
    printFeature(dirname,c[0])
    printFeature(dirname,c[1])
    printFeature(dirname,c[-1])
    print(c[0],c[1],c[2])
    
    featureVarList = []
    for i in range(len(motions2[0])): # for each feature
        features = []
        for j in range(len(motions2)):
            features.append(motions2[j][i])
        featureVarList.append(sum(np.var(features,0)))
    
    print(featureVarList)
    s = sorted(featureVarList)
    c = list(featureVarList.index(x) for x in s)
    print(c)
    printFeature(dirname,c[0])
    printFeature(dirname,c[1])
    printFeature(dirname,c[-1]) 
    print(c[0],c[1],c[2])
    
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
    

#getAllMSRMotions(['../Data/MSR/MSR_dailyActivity1/'])
#fft1 = getAllMSRMotions(['../Data/MSR/MSR_dailyActivity2/'])
#file = open('../Data/MSR/MSR_dailyActivity2/a03_s01_e01_skeleton.txt','r')
#getMSRMotion()
# ********************
#file = open('../Data/MSR/Wave_two_hands/a11_s07_e02_skeleton.txt')
#file = open('../Data/MSR/Wave_two_hands/a11_s02_e01_skeleton.txt')
# ********************
#file = open('../Data/MSR/Throwing_a_ball/a06_s08_e03_skeleton.txt')
#file = open('../Data/MSR/Throwing_a_ball/a06_s03_e01_skeleton.txt')
#file = open('../Data/MSR/Throwing_a_ball/a06_s05_e03_skeleton.txt')
# ********************
#file = open('../Data/MSR/Bend/a13_s01_e01_skeleton.txt')
#file = open('../Data/MSR/Bend/a13_s04_e02_skeleton.txt')
# ********************
#file = open('../Data/MSR/Wave_one_arm/a01_s03_e03_skeleton.txt')
#file = open('../Data/MSR/Wave_one_arm/a02_s03_e01_skeleton.txt')
# ********************
#file = open('../Data/MSR/Hand_clap/a10_s02_e01_skeleton.txt')
#file = open('../Data/MSR/Hand_clap/a10_s04_e01_skeleton.txt')
#file = open('../Data/MSR/Hand_clap/a10_s09_e01_skeleton.txt')
# ********************
#file = open('../Data/MSR/Side-boxing/a12_s01_e03_skeleton.txt')
#file = open('../Data/MSR/Side-boxing/a12_s03_e03_skeleton.txt')
#file = open('../Data/MSR/Side-boxing/a12_s07_e03_skeleton.txt')



def tmp():
    fft,diff = getMSRAction(file)
    
    mat = getMotionCorr(fft)
    var_list = []
    for i in range(len(mat)):
        var_list.append(np.var(mat[i,:]))
    th = max(var_list)*0.95
    l1 = list((var_list.index(x),x) for x in var_list if x > th)
    l2 = sorted(l1,key=lambda x: x[1],reverse=True)
    l3 = list(x[0] for x in l2)
    print(l3)
    return 


    
#w,v=np.linalg.eig(mat)
#print(w)
#print(v)
#pt.imshow(mat)
#pt.show()

#printMotions('../Data/MSR/MSR_dailyActivity2/')
#printFeature('../Data/MSR/MSR_dailyActivity2/',ind)

#findRelevantFeatures()
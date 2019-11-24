import numpy as np
import matplotlib.pyplot as pt
from scipy import signal
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GMM
import numpy as np
import scipy.stats as stats
from os import listdir
from os.path import isfile, join
from sgfilter import savitzky_golay
from sklearn.decomposition import IncrementalPCA, PCA
import mpl_toolkits
        
def notNoise(value):
    if type(value) == list:
        return abs(sum(value)) > 1
    else:
        return abs(sum(value[1])) > 1
    
def gmmClusterTest(fftArrays):
    X = fftArrays
    if X == []:
        return
    n_features = fftArrays.shape[0]
    clusMat = np.zeros((n_features,n_features), dtype=np.int32)
    
    lower = int(n_features / 3)
    upper = int(n_features / 2)
    for p in range(10):
        for j in range(lower,upper):
            GM = GMM(n_components = j,covariance_type='full')
            GM.fit(X)
            l = GM.predict(X)
            for c in range(j):
                ind = np.where(l == c)[0]
                for i in range(len(ind)):
                    for k in range(i,len(ind)):
                        clusMat[ind[i]][ind[k]] += 1
               
    mat = np.zeros((n_features,n_features), dtype=np.int32)
    mean_list = []
    # find the cluster by majority vote
    n = clusMat[0][0] # the number of gmm performed
    check = []
    total = []
    for i in range(len(clusMat)):
        if i not in check:
            ind = np.where(clusMat[i] >= n*0.8)
            for k in ind[0]:
                mat[i][k] = 1
                mat[k][i] = 1
            if type(ind[0]) == np.ndarray:
                mean = np.mean(X[ind[0],:],axis=0)
            else:
                mean = X[ind[0]]
            
            mean = mean/(max(mean)-min(mean))
            total.append(list(ind[0]))
            check += list(ind[0])
            mean_list.append(mean)
    
    t_copy = total[:]
    total.sort(key=len,reverse=True)
    print(total)
    fft_mat = np.empty((0,30), dtype=np.float64)
    pos_list = []
    for i in range(6):
        mean = mean_list[t_copy.index(total[i])]
        pos_list.append(total[i])
        # mean = mean[::2]
        if mean[0] < 0: mean = -mean
        fft_mat = np.append(fft_mat,[mean],axis=0)
        
    var_mat = np.empty((0,10), dtype=np.float64)
    a = np.arange(0,30,3)
    for item in fft_mat:
        tmp = list(np.cov([item[x],item[x+1],item[x+2]]) for x in a)
        var_mat = np.append(var_mat,[tmp],axis=0)
    
    return fft_mat#, pos_list # #mat  var_mat,

def gmmCMUCluster1(fftArrays):
    X = fftArrays
    n_features = fftArrays.shape[0]
    clusMat = np.zeros((n_features,n_features), dtype=np.int32)
    
    for p in range(10):
        for j in range(11,17):
            GM = GMM(n_components = j,covariance_type='full')
            GM.fit(X)
            l = GM.predict(X)
            for c in range(j):
                ind = np.where(l == c)[0]
                for i in range(len(ind)):
                    for k in range(i,len(ind)):
                        clusMat[ind[i]][ind[k]] += 1
    
    # find the cluster by majority vote
    mat = np.zeros((n_features,n_features), dtype=np.int32)
    n = clusMat[0][0] # the number of gmm performed
    check = []
    total = []
    mean_list = []
    for i in range(len(clusMat)):
        if i not in check:
            ind = np.where(clusMat[i] >= n*0.75)
            for k in ind[0]:
                mat[i][k] = 1
                mat[k][i] = 1            
            
            if type(ind[0]) == np.ndarray:
                mean = np.mean(X[ind[0],:],axis=0)
                
            else:
                mean = X[ind[0]]
            total.append(list(ind[0]))
            check += list(ind[0]) 
            mean_list.append(mean)
            
    omega_dict = {}
    
    for i in range(len(total)):
        
        if len(total[i]) in omega_dict.keys():
            omega_dict[len(total[i])].append((total[i],mean_list[i]))
        else:
            omega_dict[len(total[i])] = [(total[i],mean_list[i])]   
 
    return omega_dict, mat

def gmmCMUOmegaMatrix(fftArrays):
    X = fftArrays
    n_features = fftArrays.shape[0]
    clusMat = np.zeros((n_features,n_features), dtype=np.int32)
    
    for p in range(10):
        for j in range(11,17):
            GM = GMM(n_components = j,covariance_type='full')
            GM.fit(X)
            l = GM.predict(X)
            for c in range(j):
                ind = np.where(l == c)[0]
                for i in range(len(ind)):
                    for k in range(i,len(ind)):
                        clusMat[ind[i]][ind[k]] += 1
    
    # find the cluster by majority vote
    mat = np.zeros((n_features,n_features), dtype=np.float64)
    n = clusMat[0][0] # the number of gmm performed
    check = []
    total = []
    mean_list = []
    for i in range(len(clusMat)):
        if i not in check:
            ind = np.where(clusMat[i] >= n*0.75)           
            
            if type(ind[0]) == np.ndarray:
                mean = np.mean(X[ind[0],:],axis=0)
                
            else:
                mean = X[ind[0]]         
                
            total.append(list(ind[0]))
            check += list(ind[0]) 
            mean_list.append(mean)
    
    mat = np.zeros((len(total),len(total)), dtype=np.float64)
    for i in range(len(total)):
            for j in range(i+1,len(total)):    
                mat[i,j] = np.corrcoef(mean_list[i],mean_list[j])[1,0]
                mat[j,i] = np.corrcoef(mean_list[i],mean_list[j])[1,0]
 
    return  mat


def getFFTArrays(file):
    motionVal = np.zeros((0,0),dtype=np.float64)
    line = file.readline().rstrip().split()
    while line:
        if len(line) == 1 and line[0].isdigit():
            val = np.zeros((0,0),dtype=np.float64)
            while True:
                line = file.readline().rstrip().split()
                tmp = list(float(x) for x in line[1:])
                val = np.append(val,tmp)
                if line[0] == 'ltoes': 
                    if not len(motionVal): motionVal = val
                    else: motionVal = np.vstack([motionVal,val])
                    break
        line = file.readline().rstrip().split()
    motionVal = np.transpose(motionVal)
    fft_len = 31
    fft = np.empty((0,fft_len*2),dtype=np.float64)
    if not len(motionVal): return [],[]
    
    diffs = np.empty((0,len(motionVal[0])-1),dtype=np.float64)
    #diffs = np.empty((0,len(motionVal[0])),dtype=np.float64)
    for i in range(len(motionVal)):
        m = motionVal[i]
        m = savitzky_golay(m,65,3)   # originally 125
        m = np.diff(m)
        if i not in [24,25,36,37]: # delete left and right clavicles
            m = savitzky_golay(m,65,3)
            diffs = np.append(diffs,np.array([m]),axis=0)
            m = np.fft.fft(m)[:fft_len]
            # interlacing the real coefficients and the imaginary coefficients
            #m = np.abs(m)
            a,b = np.real(m),np.imag(m)
            m = np.empty((a.size + b.size,), dtype = a.dtype)
            m[0::2],m[1::2] = a,b
            fft = np.append(fft, np.array([m]),axis=0)
    file.close()
    return fft, diffs
            


def getCMUMotionDicts(dir1,dir2):

    files = [f for f in listdir(dir1) if isfile(join(dir1, f))]
    omega_dict1 = []
    for f in files:
        if '.txt' in f:
            filename = dir1 + f
            file = open(filename,"r")
            mat1 = getOmegaDict(file)
            
    files = [f for f in listdir(dir2) if isfile(join(dir2, f))]
    omega_dict2 = []
    for f in files:
        if '.txt' in f:
            filename = dir2 + f
            file = open(filename,"r")
            mat2 = getOmegaDict(file)

    return omega_dict1, omega_dict2

def getCMUMotionHamm(dir1,dir2):

    files = [f for f in listdir(dir1) if isfile(join(dir1, f))]
    hamm_mat1 = []
    for f in files:
        if '.txt' in f:
            filename = dir1 + f
            file = open(filename,"r")
            hamm_mat1.append(mat1)
            
    files = [f for f in listdir(dir2) if isfile(join(dir2, f))]
    hamm_mat2 = []
    for f in files:
        if '.txt' in f:
            filename = dir2 + f
            file = open(filename,"r")
            hamm_mat2.append(mat2)

    return  hamm_mat1, hamm_mat2



def getCMUMotionOmega(dir_list):
    total_omega_mat = []
    for dirname in dir_list:
        
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        omega_mat = []
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                file = open(filename,"r")
                X = getFFTArrays(file)
                mat = gmmCMUOmegaMatrix(X)
                omega_mat.append(mat)
        
        total_omega_mat.append(omega_mat)

    return total_omega_mat

def getMotionCorr(X):
    n_features = X.shape[0]
    corr = np.zeros((n_features,n_features), dtype=np.float64)
    for i in range(n_features):
        for j in range(i+1,n_features):
            corr[i][j] = np.corrcoef(X[i],X[j])[1,0]
            corr[j][i] = np.corrcoef(X[i],X[j])[1,0]
    return corr

def getCMUMotionCorr(dir_list):
    total_corr_mat = []
    for dirname in dir_list:
        
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        corr_mat = []
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                file = open(filename,"r")
                X = getFFTArrays(file)
                mat = getMotionCorr(X)
                corr_mat.append(mat)
        
        total_corr_mat.append(corr_mat)

    return total_corr_mat

def getCMUMotion(dir_list):
    total_corr_mat = []
    total_diff_mat = []
    for dirname in dir_list:
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        print('getCMUMotion: files',files)
        corr_mat = []
        diff_mat = []
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                file = open(filename,"r")
                X, diffs = getFFTArrays(file)
                if not len(X) or not len(diffs): continue
                corr_mat.append(X)
                diff_mat.append(diffs)
        
        total_corr_mat.append(corr_mat)
        total_diff_mat.append(diff_mat)

    return total_corr_mat, total_diff_mat


def getPCA(features,motion):
    if len(features) == 1: 
        m = motion[features[0]].reshape(1,len(motion[0]))
        return m
    
    pca = PCA(n_components=len(features))
    m = np.transpose(motion[features])
    bp = pca.fit_transform(m)
    bp = np.transpose(bp)
    return bp

    return np.squeeze(bp[:,0])




def test1():
    # *********** CMU dataset ***********
    # left arm: left humerus, radius, wrist, hand, fingers, thumb
    larm_s = [[34,35,36],[37],[38],[39,40],[41],[42,43]]
    # left arm: left humerus, hand, thumb
    larm_s = [[34,35,36],[39,40],[42,43]]
    # right arm: right humerus, radius, wrist, hand, fingers, thumb
    rarm_s = [[24,25,26],[27],[28],[29,30],[31],[32,33]]
    # right arm: right humerus, hand, thumb
    rarm_s = [[24,25,26],[29,30],[32,33]]    
    # left leg: left femur, tibia, foot, toes
    lleg_s = [[51,52,53],[54],[55,56],[57]]
    # right leg: right femur, tibia, foot, toes
    rleg_s = [[44,45,46],[47],[48,49],[50]]
    # body trunk: head, upperneck, lowerneck, thorax, upperback, lowerback
    body_s = [[21,22,23],[18,19,20],[15,16,17],[12,13,14],[9,10,11],[6,7,8]]
    j_indices = [larm_s, rarm_s, lleg_s, rleg_s, body_s]
    
    fft,motion_diff = getCMUMotion(["../Data/CMU/Walking/","../Data/CMU/Jumping/"])
    w = 7
    names = ['Walking','Jumping']
    for i in np.arange(1,10,2):
        ind1,ind2 = 2,3
        pt.figure()
        #pt.title('Walking, larm vs rarm')
        joint = j_indices
        motion = motion_diff[1][i] # 0 Walking 1 Jumping
        for p in range(len(joint[ind1])): # for one body part
            #p = getPCA(joint[j],motion)
            for q in range(len(joint[ind1][p])): # for each fine grained feature
                pt.subplot(len(joint[ind1]),6,p*6+q+1)
                pt.plot(motion[joint[ind1][p][q]])
        for p in range(len(joint[ind2])): # for another body part
            for q in range(len(joint[ind2][p])): # for each fine grained feature
                pt.subplot(len(joint[ind2]),6,p*6+q+4)
                pt.plot(motion[joint[ind2][p][q]])
        
           # for k in range(len(joint[j][0])):
           #     pt.subplot(5,3,j*3+k+1)
           #     pt.plot(motion[joint[j][0][k]])
            
           # for k in range(len(joint[j])):
           #     pt.subplot(4,3,j*3+k+1)
           #     pt.plot(motion[joint[j][k]])
            
        pt.show()
        
        
#test1()

def test():
    # *********** CMU dataset ***********
    # left arm: left humerus, radius, wrist, hand, fingers, thumb
    larm_s = [[34,35,36],[37],[38],[39,40],[41],[42,43]]
    # right arm: right humerus, radius, wrist, hand, fingers, thumb
    rarm_s = [[24,25,26],[27],[28],[29,30],[31],[32,33]]
    # left leg: left femur, tibia, foot, toes
    lleg_s = [[51,52,53],[54],[55,56],[57]]
    # right leg: right femur, tibia, foot, toes
    rleg_s = [[44,45,46],[47],[48,49],[50]]
    # body trunk: head, upperneck, lowerneck, thorax, upperback, lowerback
    body_s = [[21,22,23],[18,19,20],[15,16,17],[12,13,14],[9,10,11],[6,7,8]]
    j_indices = [larm_s, rarm_s, lleg_s, rleg_s, body_s]
    
    fft,motion_diff = getCMUMotion(["../Data/CMU/Walking/","../Data/CMU/Jumping/"])
    w = 7
    names = ['Walking','Jumping']
    for i in np.arange(0,10,2):
        motion = motion_diff[0][i]
        '''
        for k in range(2):
            range_list=[]
            
            pt.figure()
            fig, ax = pt.subplots()
            for j in j_indices:
                joint = j[0]
                p = getPCA(joint,motion)
                P = np.abs(np.fft.fft(p))[1:31]
                range_list.append(np.ptp(p))
            range_list = list(x / max(range_list) for x in range_list)
            labels = ['','l-arm','r-arm','l-leg','r-leg','trunk']
            ax.set_xticklabels(labels)
            pt.ylabel('Motion range')
            pt.bar(range(len(range_list)),range_list)
            #pt.set_title(names[k])
            print(names[k])
            pt.show()
            
        '''
        pt.figure()
        for bodypart in j_indices[1:2]:
            y = motion[bodypart[0][0]]
            pt.subplot(2,2,1)
            pt.plot(y)
            pt.subplot(2,2,2)
            f = np.fft.rfft(y)
            pt.bar(np.arange(len(f)),np.abs(f))
            f1 = np.copy(f)
            f1[30:] = np.zeros(len(f1)-30)
            y1 = np.fft.irfft(f1)
            pt.subplot(2,2,3)
            pt.plot(y1)
            pt.subplot(2,2,4)
            pt.bar(np.arange(30),np.abs(f1[:30]))
        pt.show()
            
        '''
        pt.figure()
        for bodypart in j_indices[1:2]:
            pt.subplot(3,2,1)
            pt.plot(motion[bodypart[0][0]])
            pt.subplot(3,2,2)
            f = np.abs(np.fft.fft(motion[bodypart[0][0]]))
            pt.bar(np.arange(len(f)),f)
            pt.subplot(3,2,3)
            n = 31
            f1 = f[1:n]
            f2 = np.fft.fft(motion[bodypart[0][0]])
            f2[n:] = np.zeros(len(f2)-n)
            f4 = np.fft.fft(motion[bodypart[0][0]])
            f4[:n] = np.zeros(n)
            #print('bodypart',bodypart)
            f3 = np.fft.ifft(f2)
            f5 = np.fft.ifft(f4)
            pt.plot(f3)
            pt.subplot(3,2,4)
            pt.bar(np.arange(len(f1)),f1)
            pt.subplot(3,2,5)
            y = motion[bodypart[0][0]]
            fr = np.fft.rfft(y)
            pt.bar(np.arange(len(fr)),np.real(fr))
            #pt.plot(f5)
            pt.subplot(3,2,6)
            pt.bar(np.arange(30),np.abs(fr)[:30])
            #f7 = np.abs(f4[n:])
            #pt.bar(np.arange(len(f7)),f7)
            #f4 = np.fft.fft(motion[bodypart[0][0]])
            #pt.plot(np.fft.ifft(f4))
        pt.show()
        
            
        
        
        joint = body_s
        for i in range(len(joint)):
            count = 1
            for j in range(len(joint[i])):
                pt.subplot(len(joint),4,count+4*i)
                pt.plot(motion[joint[i][j]])
                count += 1
            if len(joint[i]) > 1:
                pt.subplot(len(joint),4,count+4*i)
                p = getPCA(joint[i],motion)
                pt.plot(p)
                
        pt.show()
        
        '''

           # p = getPCA(bodypart[0],motion)
           # pt.subplot(2,2,1)
           # pt.plot(p)
           # count = 2
           # for i in range(len(bodypart[0])):
           #     pt.subplot(2,2,count)
           #     pt.plot(motion[bodypart[0][i]])
           #     count += 1
        '''
        for c in range(len(bodypart)):
            print('bodyparts[c]',bodypart[c])
            p = getPCA(bodypart[c],motion)
            #print('(j_indices.index(bodypart)+1,c+1)',(j_indices.index(bodypart)+1,c+1))
            pt.subplot(len(j_indices)+1,w,count)#(j_indices.index(bodypart)+1,c+1))
            ft = np.abs(np.fft.fft(p))[1:8]
            pt.plot(p)
            print(np.where(ft == max(ft))[0][0])
            count += 1
        pt.subplot(len(j_indices)+1,w,count)
        pt.plot([0,0,0])
        count += 1
        '''
        
        
        
    '''
    fft = fft[0]
    list1 = []
    for item in fft:
        tmp = gmmClusterTest(item)
        if len(tmp) != 0 and not np.isnan(tmp).any(): list1.append(tmp)
    pt.figure()
    for item in list1:
        for i in range(6):
            pt.subplot(2,3,i+1)
            pt.plot(item[i])
    pt.suptitle('CMU walking motions')  
    pt.show()  
    '''
    
#test()

def test2():
    Fs = 100
    f = 3
    sample = 100
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)
    pt.subplot(3,3,1)
    pt.plot(x, y)
    pt.xlabel('sample(n)')
    pt.ylabel('voltage(V)')
    
    f1 = np.fft.fft(y)
    y2 = y[10:]
    pt.subplot(3,3,2)
    pt.bar(np.arange(len(f1)),np.abs(f1))
    pt.subplot(3,3,3)
    pt.bar(np.arange(len(f1)),np.angle(f1))
    pt.subplot(3,3,4)
    pt.plot(y2)
    f2 = np.fft.fft(y2)
    pt.subplot(3,3,5) 
    pt.bar(np.arange(len(f2)),np.abs(f2))
    pt.subplot(3,3,6)
    pt.bar(np.arange(len(f2)),np.angle(f2))
    pt.subplot(3,3,7)
    f3 = np.copy(f2)
    f3[10:] = np.zeros(len(f3)-10)
    y3 = np.fft.ifft(f3)
    pt.plot(y3)
    pt.subplot(3,3,8)
    fr = np.fft.rfft(y)
    pt.bar(np.arange(len(fr)),np.abs(fr))
    pt.subplot(3,3,9)
    pt.bar(np.arange(len(f2)),np.imag(f2))
    pt.show()
    
    return f1,f2, fr

#f1,f2,fr = test2()



def testPCA():
    # left arm: left humerus, radius, wrist, hand, fingers, thumb
    larm_s = [[34,35,36],[37],[38],[39,40],[41],[42,43]]
    # right arm: right humerus, radius, wrist, hand, fingers, thumb
    rarm_s = [[24,25,26],[27],[28],[29,30],[31],[32,33]]
    # left leg: left femur, tibia, foot, toes
    lleg_s = [[51,52,53],[54],[55,56],[57]]
    # right leg: right femur, tibia, foot, toes
    rleg_s = [[44,45,46],[47],[48,49],[50]]
    # body trunk: head, upperneck, lowerneck, thorax, upperback, lowerback
    body_s = [[21,22,23],[18,19,20],[15,16,17],[12,13,14],[9,10,11],[6,7,8]]
    j_indices = [larm_s, rarm_s, lleg_s, rleg_s, body_s]
    
    fft,motion_diff = getCMUMotion(["../Data/CMU/Walking/","../Data/CMU/Running/","../Data/CMU/Jumping/"])   
    corr_list = []
    
    def getPCA(features,motion):
        if len(features) == 1: 
            m = motion[features[0]].reshape(1,len(motion[0]))
            return m
        
        pca = PCA(n_components=len(features))
        m = np.transpose(motion[features])
        bp = pca.fit_transform(m)
        c = pca.components_
        bp = np.transpose(bp)
        
        return bp,c

    # test a pair of joint
    
    for motion_type in motion_diff[2:]:
        corr_list.append([])
        for motion in motion_type[::2]:
            if motion_diff.index(motion_type) == 0:
                print('Walking')
            elif motion_diff.index(motion_type) == 1:
                print('Running')            
            elif motion_diff.index(motion_type) == 2:
                print('Jumping')
            pca1,c1 = getPCA(larm_s[0],motion)
            pca2,c2 = getPCA(rleg_s[0],motion)
            #corr1 = np.corrcoef(c1[0],c2[0])[1,0]
            #corr2 = np.corrcoef(c1[1],c2[1])[1,0]
            #corr3 = np.corrcoef(c1[2],c2[2])[1,0]
            #print('CORR',corr1,corr2,corr3)
            
            #print('pca1',pca1)
            #pca_rgs1 = [np.ptp(pca1[0]),np.ptp(pca1[0]),np.ptp(pca1[1])]
            #pca_rgs2 = [np.ptp(pca2[0]),np.ptp(pca2[0]),np.ptp(pca2[1])]
            #pca_rgs = np.array(pca_rgs1) + np.array(pca_rgs2)
            #pca_rgs = list(x / pca_rgs.max() for x in pca_rgs)
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = pt.figure()
            print('c1.shape,c2.shape',c1.shape,c2.shape)
            print('[c1[0][0],c1[0][1]]',[c1[0][0],c1[0][1]])
            ax = fig.gca(projection='3d')
            
            '''
            #ax.plot([0,c1[0][0]],[0,c1[0][1]],[0,c1[0][2]])
            ax.plot(pca1[0],pca1[1],pca1[2])
            #ax.plot(np.arange(len(pca1[2])),pca1[0],pca1[1])
            #ax.plot([0,c2[0][0]],[0,c2[0][1]],[0,c2[0][2]])
            ax.plot(pca2[0],pca2[1],pca2[2])
            #ax.plot(np.arange(len(pca2[2])),pca2[0],pca2[1])
                    
            '''
            #ax.plot([0,c1[0][0]],[0,c1[1][0]],[0,c1[2][0]])
            ax.plot(motion[larm_s[0][0]],motion[larm_s[0][1]],motion[larm_s[0][2]])
            #ax.plot(np.arange(len(motion[larm_s[0][2]])),motion[larm_s[0][2]],motion[larm_s[0][0]])
            #ax.plot([0,c2[0][0]],[0,c2[1][0]],[0,c2[2][0]])
            ax.plot(motion[rarm_s[0][0]],motion[rarm_s[0][1]],motion[rarm_s[0][2]])
            #ax.plot(np.arange(len(motion[rarm_s[0][2]])),motion[rarm_s[0][2]],motion[rarm_s[0][0]])
            
            
            #ax.plot([0,c1[1][0]],[0,c1[1][1]],[0,c1[1][2]])
            #ax.plot([0,c2[0][0]],[0,c2[0][1]],[0,c2[0][2]])
            #print(c1,c2)
            pt.show()
        
            
            #print('pca_rgs1',pca_rgs1)
            corr1 = np.correlate(c1[0],c2[0])[0]
            
            
            corr1 *= pca_rgs[0]
            corr2 = np.correlate(c1[1],c2[1])[0]
            corr2 *= pca_rgs[1]
            corr3 = np.correlate(c1[2],c2[2])[0] 
            corr3 *= pca_rgs[2]
            print(corr1+corr2+corr3)
            
            corr_list[-1].append(corr1)
            print('CORR',corr1,corr2,corr3)
    #pt.figure()
    #print(corr_list[0])
    #pt.hist(corr_list[0])
    #pt.hist(corr_list[1])
    #pt.hist(corr_list[2])  
    #pt.hist(np.abs(corr_list[0]))
    #pt.hist(np.abs(corr_list[1]))
    #pt.hist(np.abs(corr_list[2]))
    
    #pt.show()
            
    return

#testPCA()


    
def showMutualInformation():
    CMU = [0.09322166, 0.08949422, 0.10162457, 0.09436428, 0.07524391, 0.08296093, 0.17992575, 0.12325613, 0.08628805, 0.0541068, 0.58544146, 0.6686832, 0.50420469, 0.53858235, 0.73706359, 0.60679083, 0.27852752, 0.12891975, 0.1536587, 0.17091746, 0.6330412, 0.27326552, 0.19229427, 0.17842226, 0.42971802]
    KIT = [0.1617607, 0.14213129, 0.09690868, 0.19601783, 0.07621756, 0.04929127, 0.37023107, 0.16547445, 0.09505743, 0.09497183, 0.71273746, 0.65925784, 0.12957277, 0.19608128, 0.48970517, 0.5535977, 0.18925727, 0.37020331, 0.38779938, 0.28915762, 0.68418217, 0.22646468, 0.36752007, 0.4100653, 0.16784169]
    CK = [0.11444502, 0.10645435, 0.06303651, 0.08167782, 0.05730431, 0.05086701, 0.1952731, 0.09719347, 0.06677765, 0.05412152, 0.57028385, 0.62662421, 0.26145385, 0.31092228, 0.53256462, 0.5424809, 0.20946944, 0.2158295, 0.23866927, 0.20594089, 0.59350431, 0.2008261, 0.25952651, 0.26271675, 0.25335011]

    MoCap = [0.07936823, 0.00057435, 0.15272781, 0.0125092, 0.10819523, 0.07084095, 0.2132749, 0.14027805, 0.55185838, 0.46166124, 0.36068574, 0.41131883, 0.32894537, 0.4002077, 0.40231141, 0.09520136, 0.09101298, 0.21893681, 0.22906982, 0.11294132, 0.11829768, 0.09202666, 0.0464373, 0.04440468, 0.22716906]
    Acce = [0.09098793, 0.04066869, 0.15552913, 0.10422842, 0.15166754, 0.14958657, 0.22518471, 0.06821868, 0.48958329, 0.27122982, 0.59258238, 0.64529181, 0.5944917, 0.60609358, 0.43956087, 0.05117799, 0.08967907, 0.20917263, 0.17685853, 0.16991141, 0.11919213, 0.24501697, 0.117299, 0.25015351, 0.120648]
    ma = [0.04525204, 0.01797091, 0.149787, 0.04297278, 0.08514888, 0.0730592, 0.15592765, 0.05888948, 0.40517442, 0.36262069, 0.42112338, 0.45600425, 0.3944224, 0.43948246, 0.39335298, 0.04613628, 0.04083726, 0.11284616, 0.0926927, 0.0732592, 0.08671335, 0.11050526, 0.04749565, 0.07026746, 0.10481892]
    
    m_list1,m_list2 = [],[]
    for d in [CMU,KIT,CK]:
        m = np.zeros((4,10))
        m[0,:] = d[:10]
        m[1,:5] = d[10:15]
        m[2,:5] = d[15:20]
        m[3,:5] = d[20:]
        m_list1.append(m)
    for d in [MoCap,Acce,ma]:
        m = np.zeros((4,10))
        m[0,:] = d[:10]
        m[1,:5] = d[10:15]
        m[2,:5] = d[15:20]
        m[3,:5] = d[20:]
        m_list2.append(m)
    m1 = np.hstack(m_list1)
    m2 = np.hstack(m_list2)
    
    
    pt.figure()
    pt.subplot(2,1,1)
    pt.imshow(m1)
    pt.colorbar()
    pt.subplot(2,1,2)
    pt.imshow(m2)
    pt.colorbar()
    pt.show()
    return


#showMutualInformation()

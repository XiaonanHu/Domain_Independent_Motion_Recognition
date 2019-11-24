import numpy as np
import matplotlib.pyplot as pt
from scipy import signal
from sklearn.mixture import GMM
from sklearn.decomposition import IncrementalPCA, PCA

from os import listdir
from os.path import isfile, join
from sgfilter import savitzky_golay


        
def isNoise(value):
    return abs(sum(value)) < 1


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
            
            #mean = mean/(max(mean)-min(mean))
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
    
    
    return fft_mat#, pos_list # #mat 

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
            
            if type(ind[0]) == np.ndarray:
                mean = np.mean(X[ind[0],:],axis=0)
            else:
                mean = X[ind[0]]
                
            total.append(list(ind[0]))
            check += list(ind[0]) 
            mean_list.append(mean)
                        
    return total, mean_list
    

def getKITFFT(file):
    motionDict = {}
    
    motionVal = np.zeros((0,0),dtype = np.float64)
    
    # ############## READ IN DATA ##############
    for line in file:
        
        if 'MotionFrame' in line:
            while '/MotionFrame' not in line:
                
                if 'JointPosition' in line and '</JointPosition>' in line:
                    line = line.lstrip('<JointPosition>|\t').rstrip('</JointPosition>|\n')
                    line = [float(x) for x in line.split()]
                    if len(line) == 44: line = np.delete(line,[0,1,2,26,27,42,43])
                    val = np.array(line,dtype=np.float64)
                    if len(val) == 39: print(val)
            
                line = file.readline()
             
            if len(val) > 3:  
                if not len(motionVal): motionVal = np.append(motionVal,val)
                else: motionVal = np.vstack([motionVal,val])                
    
    motionVal = np.transpose(motionVal)
    
    fft_len = 31
    fftArrays = np.empty((0,fft_len*2),dtype=np.float64)
    diffs = np.empty((0,motionVal.shape[1]-1),dtype=np.float64)
    #first_list, last_list = [],[]
    
    for i in range(motionVal.shape[0]):
        m = motionVal[i]
        m = savitzky_golay(m,95,3) # 71
        m = np.diff(m)
        m = savitzky_golay(m,95,3)
        
        diffs = np.append(diffs, np.array([m]),axis=0)
        #first = np.mean(first_list)
        #last = np.mean(last_list)
    
        if sum(abs(m)) > 0:
            m = np.fft.fft(m)[:fft_len]            
            # interlacing the real coefficients and the imaginary coefficients
            a,b = np.real(m),np.imag(m)
            tmp = np.empty((a.size + b.size,), dtype = a.dtype)
            tmp[0::2],tmp[1::2] = a,b
            m = tmp
            #m = m / (max(m) - min(m))
            fftArrays = np.append(fftArrays, np.array([m]),axis=0)
        else:
            fftArrays = np.append(fftArrays, np.array([[0]*(fft_len*2)]),axis=0)
    
    return fftArrays, diffs


            
def getAllMotionVals(dirlist, item):
    total_list = []
    for dirname in dirlist:
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        
        tmp_list = []
        
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                file = open(filename,"r")
                X, diffs = getKITFFT(file)
                total, mean_list = gmmClusterTest(X)
                if item in 'omega':
                    tmp,_ = getOmegaAndOmegaMatrix(total,mean_list,X.shape[0]) 
                elif item in 'omegamat':
                    _,tmp = getOmegaAndOmegaMatrix(total,mean_list,X.shape[0]) 
                elif item in 'structure':
                    tmp = getHammMatrix(total,mean_list,X.shape[0])
                tmp_list.append(tmp)
                
        total_list.append(tmp_list)
            
    return total_list
    
            

def getMotionCorr(X):
    n_features = X.shape[0]
    corr = np.zeros((n_features,n_features), dtype=np.float64)
    for i in range(n_features):
        for j in range(i+1,n_features):
            corr[i][j] = np.corrcoef(X[i],X[j])[1,0]
            corr[j][i] = np.corrcoef(X[i],X[j])[1,0]
    return corr



def getKITMotion(dir_list): # getKITMotionCorr
    total_fft_mat = []
    total_diff_mat = []
    for dirname in dir_list:
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        fft_mat = []
        diff_mat = []
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                #print('filename',filename)
                file = open(filename,"r")
                X, diffs = getKITFFT(file)
                fft_mat.append(X)
                diff_mat.append(diffs)
        total_fft_mat.append(fft_mat)
        total_diff_mat.append(diff_mat)

    return total_fft_mat, total_diff_mat




def getPCA(features,motion):
    from sklearn.decomposition import IncrementalPCA, PCA
    
    if len(features) == 1: return motion[features[0]].reshape(1,len(motion[features[0]]))
    pca = PCA(n_components=len(features))
    pca = PCA(n_components=1)
    m = np.transpose(motion[features])
    bp = pca.fit_transform(m)

    return np.transpose(bp)


def test():
    # *********** KIT dataset ***********
    # left arm: left shoulder, elbow, wrist
    larm_t = [[18,19,20],[12,13],[21,22]]
    # right arm: right shoulder, elbow, wrist
    rarm_t = [[32,33,34],[26,27],[35,36]]
    # left leg: left hip, knee, ankle/foot
    lleg_t = [[14,15,16],[17],[9,10,11]]
    # right leg: right hip, knee, ankle/foot
    rleg_t = [[28,29,30],[31],[23,24,25]]
    # body: upperneck, torso, pelvis
    body_t = [[6,7,8],[3,4,5],[0,1,2]]
    j_indices = [larm_t,rarm_t,lleg_t,rleg_t,body_t]
    
    fft,motion_diff = getKITMotion(['../Data/KIT/Walk/','../Data/KIT/Run/','../Data/KIT/Jump/'])
    w = 7
    for motion in motion_diff[2][::2]: # 0 walking 2 jumping
        pt.figure()
        print('\nNew Motion\n')
        rg_list = []
        bp1,bp2 = 2,3
        for c in range(len(j_indices[bp1])): # for each joint
            for k in range(len(j_indices[bp1][c])): # for each fine-grained feature
                pt.subplot(len(j_indices[bp1]),6,c*6+k+1)
                pt.plot(motion[j_indices[bp1][c][k]])
        for c in range(len(j_indices[bp2])): # for each joint
            for k in range(len(j_indices[bp2][c])): # for each fine-grained feature
                pt.subplot(len(j_indices[bp2]),6,c*6+k+4)
                pt.plot(motion[j_indices[bp2][c][k]])
        
        '''for bodypart in j_indices[0:1]:
            for c in range(len(bodypart)):
                tmp = 0
                p = getPCA(bodypart[c],motion)
                for k in range(len(bodypart[c])):
                    #tmp += np.ptp(motion[bodypart[c][k]])
                    tmp += np.ptp(p[k])
                    pt.subplot(len(bodypart),3,c*3+k+1)
                    ft = np.abs(np.fft.fft(p))[1:31]
                    #pt.plot(p[k])
                    pt.plot(motion[bodypart[c][k]])
                rg_list.append(tmp)
        print(rg_list,'\n',rg_list.index(max(rg_list)))'''
        pt.show()
#test()    




def getSignificantFrequency2(bp): # 2cy
    f = np.abs(np.fft.rfft(bp))[1:31]
    f_sort = sorted(f,reverse=True)
    #print('f_sort',f_sort)
    
    if list(f).index(f_sort[0]) == 0:
        if f_sort[0]*0.5 > f_sort[1]: 
            final =list(f).index(f_sort[0]) # originally 0.6
        else: 
            final = list(f).index(f_sort[1])
        
    else:
        threshold = f_sort[0]*0.7
        freqs = list(1 if x > threshold else 0 for x in f)
        inds = np.where(np.array(freqs == 1)) [0]
        # trying to find harmonic frequencies
        for i in range(len(inds)):
            for j in range(i+1,len(inds)):
                if inds[i] > 1:
                    if int(inds[j]/2) in range(inds[i]-1,inds[i]+1):
                        final = inds[i]
        final = list(f).index(f_sort[0]) 
    f1 =  np.fft.rfft(bp)[1:31][final]
    phase = np.arctan(np.real(f1)/np.imag(f1))
    return final,phase
    
def test1():
    indices = [[26,27],[12,13],[28,29,30],[14,15,16],[3,4,5]]    
    print('kit golf')
    fft,diff = getKITMotion(["../Data/KIT/Golf/"])
    for j in range(len(diff[0])):
        motion = diff[0][j]
        m_list = []
        for ind in indices:
            m_list.append(getPCA(ind,motion))
        sf = []
        pv = []
        pt.figure()
        for i in range(len(m_list)):
            pt.subplot(3,3,i+1)
            #pt.plot(motion[indices[i]])
            pt.plot(m_list[i][0])
            #print(m_list[i])
            ind,phase = getSignificantFrequency2(m_list[i][0]) 
            sf.append(ind)
            pv.append(phase)
        print('sf',sf)
        print('pv',pv)
        pt.show()
    '''
    #t = np.linspace(-1, 1, 200, endpoint=False)
    #sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
    widths = np.arange(1, 51)
    #cwtmatr = signal.cwt(sig, signal.ricker, widths)
    for i in range(36,37):
        #pt.subplot(2*num,1,2*i+1)
        pt.plot(motion[i])
        #pt.subplot(2*num,1,2*i+2)
        #print('motion[i].shape',motion[i].shape)
        #wl = signal.cwt(motion[i],signal.ricker,widths)
        
        #pt.imshow(wl)
    '''
test1()

    
    
    
#test()
#getKITMotionCorr(['../KIT/Walk/'])
#ma = max(np.abs(m))*0.1
#print('first',ma,np.where(np.abs(m) > ma))
#if ma != 0:
#    first = np.where(np.abs(m) > ma)[0][0]
#    last = np.where(np.abs(m) > ma)[0][-1]
#    first_list.append(first)
#    last_list.append(last)
#if max(m)-min(m)>0: m = m/(max(m)-min(m))


def testPCA():
    # left arm: left shoulder, elbow, wrist
    larm = [[18,19,20],[12,13],[21,22]]
    # right arm: right shoulder, elbow, wrist
    rarm = [[32,33,34],[26,27],[35,36]]
    # left leg: left hip, knee, ankle/foot
    lleg = [[14,15,16],[17],[9,10,11]]
    # right leg: right hip, knee, ankle/foot
    rleg = [[28,29,30],[31],[23,24,25]]
    # body: upperneck, torso, pelvis
    body = [[6,7,8],[3,4,5],[0,1,2]]
    j_indices = [larm,rarm,lleg,rleg,body]
    
    fft,motion_diff = getKITMotion(["../Data/KIT/Walk/","../Data/KIT/Run/","../Data/KIT/Jump/"])
    corr_list = []
    
    def getPCA(features,motion):
        if len(features) == 1: 
            m = motion[features[0]].reshape(1,len(motion[0]))
            return m
        
        pca = PCA(n_components=len(features))
        m = np.transpose(motion[features])
        bp = pca.fit_transform(m)
        c = pca.components_
        #print(c)
        bp = np.transpose(bp)
        
        return bp,c

    # test a pair of joint
    
    for motion_type in motion_diff:
        corr_list.append([])
        for motion in motion_type:
            if motion_diff.index(motion_type) == 0:
                print('Walking')
            elif motion_diff.index(motion_type) == 1:
                print('Running')            
            elif motion_diff.index(motion_type) == 2:
                print('Jumping')
            pca1,c1 = getPCA(lleg[2],motion)
            pca2,c2 = getPCA(rleg[2],motion)
            #corr1 = np.corrcoef(c1[0],c2[0])[1,0]
            #corr2 = np.corrcoef(c1[1],c2[1])[1,0]
            #corr3 = np.corrcoef(c1[2],c2[2])[1,0]
            #print('CORR',corr1,corr2,corr3)
            
            #print('pca1',pca1)
            pca_rgs1 = [np.ptp(pca1[0]),np.ptp(pca1[1]),np.ptp(pca1[2])]
            pca_rgs2 = [np.ptp(pca2[0]),np.ptp(pca2[1]),np.ptp(pca2[2])]
            pca_rgs = np.array(pca_rgs1) + np.array(pca_rgs2)
            pca_rgs = list(x / pca_rgs.max() for x in pca_rgs)
            
            #print('pca_rgs1',pca_rgs1)
            corr1 = np.correlate(c1[0],c2[0])[0]
            corr2 = np.correlate(c1[1],c2[1])[0]
            corr3 = np.correlate(c1[2],c2[2])[0] 
            print(corr1+corr2+corr3)
            
            corr_list[-1].append(corr1)
            print('CORR',corr1,corr2,corr3)
    pt.figure()
    print(corr_list[0])
    pt.hist(corr_list[0])
    pt.hist(corr_list[1])
    pt.hist(corr_list[2])  
    #pt.hist(np.abs(corr_list[0]))
    #pt.hist(np.abs(corr_list[1]))
    #pt.hist(np.abs(corr_list[2]))
    
    pt.show()
            
    return

#testPCA()
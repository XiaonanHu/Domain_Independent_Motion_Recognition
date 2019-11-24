import numpy as np
import matplotlib.pyplot as pt
import scipy.signal as sg
import scipy.stats as stats
#from scipy.stats import signaltonoise
from sklearn.mixture import GMM
from sklearn.decomposition import IncrementalPCA, PCA
from os import listdir
from os.path import isfile, join
import math as mt
from sgfilter import savitzky_golay
import itertools
import pandas



        
f_names = ['hips xp', 'hips yp', 'hips zp', 'hips zr', ' hips xr', 'hips yr',\
           'spine x', 'spine y', 'spine z', 'spine1 x', 'spine1 y', 'spine1 z',\
           'spine2 x', 'spine2 y', 'spine2 z', 'neck x', 'neck y', 'neck z',\
           'head x', 'head y', 'head z', 'right shoulder x', 'right shoulder y',\
           'right shoulder z', 'right arm x', 'right arm y', 'right arm z',\
           'right fore arm x', 'right fore arm y', 'right fore arm z',\
           'right fore arm roll z', 'left shoulder x', 'left shoulder y',\
           'left shoulder z', 'left arm x', 'left arm y', 'left arm z',\
           'left fore arm x', 'left fore arm y', 'left fore arm z',\
           'left fore arm roll z', 'right up leg x', 'right up leg y',\
           'right up leg z', 'right up leg roll z', 'right leg x', 'right leg y',\
           'right leg z', 'right leg roll z', 'right foot x', 'right foot y',\
           'right foot z', 'right toe base y', 'left up leg x', 'left up leg y',\
           'left up leg z', 'left up leg roll z', 'left leg x', 'left leg y',\
           'left leg z', 'left leg roll z', 'left foot x', 'left foot y',\
           'left foot z'
]



#fs = [34,35,36,24,25,26,61,62,63,49,50,51,53,54,55,41,42,43]

fs = [34,35,36,24,25,26,61,62,63,49,50,51,3,4,5] 

def isNoise(value):
    return abs(sum(value)) < 1


def getHammMatrix(total,mean_list,n_features):
    mat = np.zeros((n_features,n_features), dtype=np.float64)
    
    for i in range(len(total)):
        for l in total[i]:
            mat[l,total[i]] = 1        
    
    return mat



def printBKLMotion(motion):
    w = round(np.sqrt(len(motion)),0)
    h = int(len(motion)/w+1)
    pt.figure()
    for i in fs:
        sp = pt.subplot(w,h,i+1)
        #m = savitzky_golay(motion[i],131,3)
        pt.plot(m)
        sp.set_title(f_names[i])
    pt.show()

    
    return None



def getFeatureFluctuation(traj):
    t = np.diff(traj)
    t1 = np.diff(t)
    t1 = t1 / (max(t1)-min(t1)) * len(t1)
    l = list(1 if np.sign(t[i+1]) != np.sign(t[i]) and t1[i] > 0.4
             else 0 for i in range(len(t)-1))
    return sum(l)



def printMotion1(motion):
    w = round(np.sqrt(len(fs)),0)
    h = int(len(fs)/w+1)
    pt.figure()
    for i in fs:
        ind = fs.index(i)
        sp = pt.subplot(w,h,ind+1)
        win = sg.hann(20)
        m = sg.convolve(motion[i],win,mode='same')/sum(win)
        #m = savitzky_golay(motion[i],151,3)
        pt.plot(m)
        sp.set_title(f_names[i])
    pt.show()

    return None




def printMotion2(motion):
    w = 2 #round(np.sqrt(len(fs)),0)
    h = 3 #int(len(fs)/w+1)
    fs = range(5)
    names = ['left arm','right arm', 'left leg', 'right leg', 'hip']
    pt.figure()
    snr_list = []
    r_list = []
    r1_list,r2_list = [],[]
    c_list = []
    for i in fs:
        ind = fs.index(i)
        m = motion[i]
        sp = pt.subplot(w,h,ind+1)
        #win = signal.hann(20)
        #m = signal.convolve(m,win,mode='same')/sum(win)
        m = savitzky_golay(m,51,3)
        #m = m[::10]
        mf = np.imag(np.fft.fft(m))[1:40]
        tmp = np.correlate(m,m,'full')
        pt.plot(tmp)
        l = len(tmp)
        t1 = np.mean(tmp) + np.std(tmp)
        above = tmp[tmp > t1]
        
        #c = sum(list(1 if (above[i]==False and above[i+1]==True) else 0 for i \
        #             in range(len(above)-1))) 
        d = np.diff(above)
        c = sum(list(1 if (np.sign(d[i])==1 and np.sign(d[i+1])==-1) else 0 for i \
                     in range(len(d)-1))) 
        c_list.append(c)
        t2 = np.mean(tmp) - np.std(tmp)
        pt.plot([t1]*l)
        pt.plot([t2]*l)
        
        sp.set_title(names[i]) 
    pt.subplots_adjust(wspace=0.4,hspace=0.4)
    c_list = np.array(c_list)/max(c_list)
    print(' '.join(str(round(x,2)) for x in c_list))
    pt.show()

    return None


def printMotion(motion):
    w = 3 #round(np.sqrt(len(fs)),0)
    h = 5 #int(len(fs)/w+1)
    pt.figure()
    snr_list = []
    r_list = []
    r1_list,r2_list = [],[]
    for i in fs:
        ind = fs.index(i)
        m = motion[i]
        sp = pt.subplot(w,h,ind+1)
        #win = signal.hann(20)
        #m = signal.convolve(m,win,mode='same')/sum(win)
        m = savitzky_golay(m,31,3)
        m = m[::10]
        #m = sg.medfilt(m,kernel_size=11)
        #m = savitzky_golay(m,33,7)
        mf = np.imag(np.fft.fft(m))[1:40]
        #v = sum(abs(mf[1:18]))/sum(abs(mf[1:]))
        #var_list.append(round(v,2))
        tmp = np.correlate(m,m,'full')
        pt.plot(tmp)
        
        #pt.plot(m)
        snr_list.append(str(round(float(stats.signaltonoise(m)),3))) 
        r2_list.append(str(round(sum(abs(mf[:17]))/sum(abs(mf[17:])),2)))
        r_list.append(str(round(max(abs(mf))/np.mean(abs(mf)),2)))
        r1_list.append(str(round(np.std(m)/np.ptp(m),2)))
        
        
        sp.set_title(f_names[i])
    for i in range(3,len(snr_list)+2,4): snr_list.insert(i,'|')
    for i in range(3,len(r_list)+2,4): r_list.insert(i,'|')
    for i in range(3,len(r1_list)+2,4): r1_list.insert(i,'|')
    for i in range(3,len(r2_list)+2,4): r2_list.insert(i,'|')
    print('snr_lsit: ', ' '.join(x for x in snr_list))
    print('r_list', ' '.join(x for x in r_list))
    print('r1_list',' '.join(x for x in r1_list))
    print('r2_list',' '.join(x for x in r2_list))    
    pt.subplots_adjust(wspace=0.4,hspace=0.4)
    pt.show()

    return None


def skipLines(file):
    for _ in range(11): line = file.readline()
    return line

def getOneBKL(file):
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
        line1 = list(line[i] for i in range(0,len(line),3))
        line2 = list(line[i] for i in range(1,len(line),3))
        line3 = list(line[i] for i in range(2,len(line),3))
        line = list([line1[i],line2[i],line3[i]] for i in range(len(line1)))
        line = list(itertools.chain(*line))

        val = np.array(line)
        if not len(motionVal): motionVal = val
        else:
            if len(val): motionVal = np.vstack([motionVal,val])
            else: break
        line = skipLines(file) # skip to 12th lines

        
        
    motionVal = np.transpose(motionVal)

    fft_len = 30
    fft = np.empty((0,fft_len*2), dtype=np.float64)
    phase = np.empty((0,fft_len), dtype=np.float64)
    if not len(motionVal): return [],[]
    
    #diffs = np.empty((0,len(motionVal[0])),dtype=np.float64)
    val = np.empty((0,len(motionVal[0])-1),dtype=np.float64)
    
    for i in range(len(motionVal)):
        m = motionVal[i]  
        m = savitzky_golay(m,63,3)
        m = np.diff(m)
        if i not in [27,28,29,33,34,45,46,47,51,52,61,66,67,78,79,84,85]:
            m = savitzky_golay(m,63,3)
            #if max(m)-min(m) > 0: m = m / (max(m)-min(m))
            val = np.append(val,m.reshape(1,len(m)),axis=0)
        
            m = np.fft.fft(m)[1:fft_len+1]
            
            # interlacing the real coefficients and the imaginary coefficients
            a, b = np.real(m), np.imag(m)
            m = np.empty((a.size + b.size,), dtype = a.dtype)
            m[0::2], m[1::2] = a,b
            if (max(m) - min(m)) != 0: m = m / (max(m) - min(m))

            fft = np.append(fft, np.array([m]),axis=0)
            phase = np.append(phase, np.array([b]),axis=0)
    file.close()
    fft = np.delete(fft,[31,32,33,44,45,46,50,59,61,73,74,75],axis=0)
    val = np.delete(val,[31,32,33,44,45,46,50,59,61,73,74,75],axis=0)
    return fft, val 

    
def getBKLMotion():
    dirlist = ['../Data/BKL/Mocap/Wave_two_hands/','../Data/BKL/Mocap/Wave_one_hand/',\
               '../Data/BKL/Mocap/Boxing/', '../Data/BKL/Mocap/Jumping_jacks/',\
               '../Data/BKL/Mocap/Throwing/',\
               '../Data/BKL/Mocap/Bending/']
    
    dirlist = ['../Data/BKL/Mocap/Wave_two_hands/','../Data/BKL/Mocap/Boxing/', '../Data/BKL/Mocap/Jumping_jacks/',\
               '../Data/BKL/Mocap/Bending/']    
    
    total_mat_list = []
    total_diff_list = []
    total_phase_list = []
    for dirname in dirlist:
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        mat_list = []
        diff_list = []
        for f in files:
            if '.txt' in f:
                filename = dirname + f
                #print(filename)
                file = open(filename,"r")
                X, diffs = getOneBKL(file)
                if not len(X) or not len(diffs): continue
                mat_list.append(X)
                diff_list.append(diffs)
        
        total_mat_list.append(mat_list)
        total_diff_list.append(diff_list)  
    
    return total_mat_list, total_diff_list


def getBKLEachMotion():
    dirname = '../Data/BKL/Mocap/Clapping/'
    files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    mat_list = []
    diff_list = []
    for f in files:
        if '.txt' in f:
            filename = dirname + f
            #print(filename)
            file = open(filename,"r")
            X, diffs = getOneBKL(file)
            if not len(X) or not len(diffs): continue
            mat_list.append(X)
            diff_list.append(diffs)
    
    return mat_list, diff_list




def getAllBKLMotions(dirname):
    files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    corr_mat = []
    for f in files:
        if '.txt' in f:
            filename = dirname + f
            file = open(filename,"r")
            X,diffs = getOneBKL(file)
            corr_mat.append(diffs)
    return corr_mat



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




# 31, 32, 33, 44, 45, 46, 50, 59, 61, 73, 74, 75
# **********************
#file = open('../Data/BKL/Mocap/Throwing/skl_s01_a08_r01.txt')
#file = open('../Data/BKL/Mocap/Throwing/skl_s02_a08_r03.txt')
#file = open('../Data/BKL/Mocap/Throwing/skl_s08_a08_r01.txt')
# **********************
#file = open('../Data/BKL/Mocap/Wave_two_hands/skl_s01_a05_r05.txt')
#file = open('../Data/BKL/Mocap/Wave_two_hands/skl_s07_a05_r02.txt')
#file = open('../Data/BKL/Mocap/Wave_two_hands/skl_s11_a05_r01.txt')
# ********************** 
#file = open('../Data/BKL/Mocap/Boxing/skl_s02_a04_r04.txt')
#file = open('../Data/BKL/Mocap/Boxing/skl_s03_a04_r04.txt')

# ********************** 
#file = open('../Data/BKL/Mocap/Clapping/skl_s03_a07_r04.txt')
#file = open('../Data/BKL/Mocap/Clapping/skl_s06_a07_r05.txt')
#file = open('../Data/BKL/Mocap/Clapping/skl_s08_a07_r03.txt')
#file = open('../Data/BKL/Mocap/Clapping/skl_s11_a07_r01.txt')
#file = open('../Data/BKL/Mocap/Clapping/skl_s12_a07_r02.txt')
#file = open('../Data/BKL/Mocap/Clapping/skl_s05_a07_r02.txt')
# **********************
#file = open('../Data/BKL/Mocap/Wave_one_hand/skl_s01_a06_r03.txt')
#file = open('../Data/BKL/Mocap/Wave_one_hand/skl_s05_a06_r03.txt')
#file = open('../Data/BKL/Mocap/Wave_one_hand/skl_s10_a06_r04.txt')
# **********************
#file = open('../Data/BKL/Mocap/Jumping_jacks/skl_s01_a02_r03.txt')
#file = open('../Data/BKL/Mocap/Jumping_jacks/skl_s01_a02_r01.txt')
#file = open('../Data/BKL/Mocap/Jumping_jacks/skl_s05_a02_r04.txt')
#file = open('../Data/BKL/Mocap/Jumping_jacks/skl_s08_a02_r05.txt')
#file = open('../Data/BKL/Mocap/Jumping_jacks/skl_s11_a02_r02.txt')
# **********************


def c():
    fs = [34,35,36,24,25,26,61,62,63,49,50,51] #,3,4,5]
    pv = getAllBKLMotions('../Data/BKL/Mocap/Boxing/')
    w2v = getAllBKLMotions('../Data/BKL/Mocap/Wave_two_hands/')
    jjv = getAllBKLMotions('../Data/BKL/Mocap/Jumping_jacks/')
    w1v = getAllBKLMotions('../Data/BKL/Mocap/Wave_one_hand/')
    cv = getAllBKLMotions('../Data/BKL/Mocap/Clapping/')
    print('Punching')
    for m in range(0,len(pv),5):
        #tmp = list(np.corrcoef(pv[m][fs[i]],pv[m][fs[i+3]])[1,0] for i in range(3))
        #tmp = list(np.corrcoef(pv[m][fs[j]],pv[m][fs[k]])[1,0] for j in range(3) for k in range(j+1,3))
        l = list(np.ptp(pv[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])), 
               l[9:12].index(max(l[9:12])),l.index(max(l))]
        print(tmp)
        
    print('Jumping jacks')
    for m in range(0,len(jjv),5):
        #tmp = list(np.corrcoef(jjv[m][fs[i]],jjv[m][fs[i+3]])[1,0] for i in range(3))
        #tmp = list(np.corrcoef(jjv[m][fs[j]],jjv[m][fs[k]])[1,0] for j in range(3) for k in range(j+1,3))
        l = list(np.ptp(jjv[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])),
               l[9:12].index(max(l[9:12])),l.index(max(l))] 
        print(tmp)
        
    print('Wave two hands')
    for m in range(0,len(w2v),5):
        #tmp = list(np.corrcoef(jjv[m][fs[i]],jjv[m][fs[i+3]])[1,0] for i in range(3))
        #tmp = list(np.corrcoef(jjv[m][fs[j]],jjv[m][fs[k]])[1,0] for j in range(3) for k in range(j+1,3))
        l = list(np.ptp(w2v[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])),
               l[9:12].index(max(l[9:12])),l.index(max(l))] 
        print(tmp)
        
    print('Wave one hands')
    for m in range(0,len(w1v),5):
        #tmp = list(np.corrcoef(jjv[m][fs[i]],jjv[m][fs[i+3]])[1,0] for i in range(3))
        #tmp = list(np.corrcoef(jjv[m][fs[j]],jjv[m][fs[k]])[1,0] for j in range(3) for k in range(j+1,3))
        l = list(np.ptp(w1v[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])),
               l[9:12].index(max(l[9:12])),l.index(max(l))] 
        print(tmp) 
        
    print('Clapping')
    for m in range(0,len(cv),5):
        #tmp = list(np.corrcoef(jjv[m][fs[i]],jjv[m][fs[i+3]])[1,0] for i in range(3))
        #tmp = list(np.corrcoef(jjv[m][fs[j]],jjv[m][fs[k]])[1,0] for j in range(3) for k in range(j+1,3))
        l = list(np.ptp(cv[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])),
               l[9:12].index(max(l[9:12])),l.index(max(l))] 
        print(tmp)     
                
            
    return

#c()



def deleteSpikes(f):
    sd = np.std(f)
    m = np.mean(f)
    f = np.array(list(0 if (x > m+1.5*sd or x < m-1.5*sd) else x for x in f))
    return f


def c1():
    fs = [34,35,36,24,25,26,61,62,63,49,50,51,3,4,5]
    pv = getAllBKLMotions('../Data/BKL/Mocap/Boxing/')
    w2v = getAllBKLMotions('../Data/BKL/Mocap/Wave_two_hands/')
    jjv = getAllBKLMotions('../Data/BKL/Mocap/Jumping_jacks/')
    w1v = getAllBKLMotions('../Data/BKL/Mocap/Wave_one_hand/')
    #cv = getAllBKLMotions('../Data/BKL/Mocap/Clapping/')
    ssv = getAllBKLMotions('../Data/BKL/Mocap/SitdownStandup/')
    tv = getAllBKLMotions('../Data/BKL/Mocap/Throwing/')
    bv = getAllBKLMotions('../Data/BKL/Mocap/Bending/')
    jv = getAllBKLMotions('../Data/BKL/Mocap/Jumping/')
    sdv = getAllBKLMotions('../Data/BKL/Mocap/Sit_down/') 
    suv = getAllBKLMotions('../Data/BKL/Mocap/Stand_up/') 
    
    ms = [tv,ssv,jjv,w2v,w1v,pv,bv,sdv,suv,jv]
    mns = ['Throwing','Sitdown and standup','Jumping jacks','Wave two hands',\
           'Wave one hand','Punching','Bending','Sit down','Stand up','Jumping']
    
    #ms = [ssv,bv]
    #mns = ['Sitdown and standup','Bending']
    
    for motion in ms:
        print(mns[ms.index(motion)])
        for m in range(0,len(motion),7):
            #print('i',i,'m',m,'fs[j]',fs[j])
            l = list(np.mean(deleteSpikes(motion[m][j])) for j in fs)
            #print('l',l)
            #l = np.power(l,1/1.2) # originally 1.7
            l /= max(l)
            m = list("{0:.2f}".format(x) for x in l)
            for i in range(3,len(m)+1,4): m.insert(i,'|')
            print(' '.join(x for x in m))                
    return ms

#ms = c1()



def c2():
    fs = [34,35,36,24,25,26,61,62,63,49,50,51,3,4,5]
    #pv = getAllBKLMotions('../Data/BKL/Mocap/Boxing/')
    w2v = getAllBKLMotions('../Data/BKL/Mocap/Wave_two_hands/')
    #jjv = getAllBKLMotions('../Data/BKL/Mocap/Jumping_jacks/')
    #w1v = getAllBKLMotions('../Data/BKL/Mocap/Wave_one_hand/')
    #cv = getAllBKLMotions('../Data/BKL/Mocap/Clapping/')
    #ssv = getAllBKLMotions('../Data/BKL/Mocap/SitdownStandup/')
    #tv = getAllBKLMotions('../Data/BKL/Mocap/Throwing/')
    bv = getAllBKLMotions('../Data/BKL/Mocap/Bending/')
    #jv = getAllBKLMotions('../Data/BKL/Mocap/Jumping/')
    #sdv = getAllBKLMotions('../Data/BKL/Mocap/Sit_down/') 
    #suv = getAllBKLMotions('../Data/BKL/Mocap/Stand_up/') 
    
    ms = [bv,w2v]
    mns = ['Bending','Wave two hands']
    
    
    for motion in ms:
        print(mns[ms.index(motion)])
        for m in range(0,len(motion),14):
            printMotion(motion[m])   
            #print('length: ', len(motion[m][0]))
    return 

#c2()

def c3():
    fs = [34,35,36,24,25,26,61,62,63,49,50,51,3,4,5]
    w2v = getAllBKLMotions('../Data/BKL/Mocap/Wave_two_hands/')
    bv = getAllBKLMotions('../Data/BKL/Mocap/Bending/')
    
    ms = [w2v,bv]
    mns = ['Wave two hands','Bending']
    
    tot_inds,tot_ratios,tot_var,tot_snrs = [],[],[],[]
    for motion in ms:
        print(mns[ms.index(motion)])
        inds_list,ratios_list,var_list,snrs_list = [],[],[],[]
        for m in range(0,len(motion),10):
            inds,ratios,vrs,snrs = [],[],[],[]
            for i in fs:
                win = sg.hann(50)
                mo = sg.convolve(motion[m][i],win,mode='same')/sum(win)    
                
                #mo = savitzky_golay(motion[m][i],51,5)
                f = np.real(np.fft.fft(mo))[1:31]
                
                ind = list(abs(f)).index(max(abs(f)))
                inds.append(ind)
                bf = abs(f[ind])/abs(np.mean(f))
                ratios.append(bf)
                va = np.var(f/(max(f)-min(f)))
                vrs.append(va)
                snr = round(float(stats.signaltonoise(motion[m][i])),2)
                snrs.append(snr)
            inds_list.append(inds)
            ratios_list.append(ratios)
            var_list.append(vrs)
            snrs_list.append(snrs)
        tot_inds.append(inds_list)
        tot_ratios.append(ratios_list)
        tot_var.append(var_list)
        tot_snrs.append(snrs_list)
        
    for inds_list in tot_inds:
        print('New motion')
        for inds in inds_list:
            print(' '.join(str(x) for x in inds))
    for ratio_list in tot_ratios:
        print('New motion')
        for ratios in ratio_list:
            print(' '.join(str(round(x,1)) for x in ratios))
    for var_list in tot_var:
        print('New motion')
        for vrs in var_list:
            print(' '.join(str(round(x*10,3)) for x in vrs))
    for snrs_list in tot_snrs:
        print('New motion')
        for snrs in snrs_list:
            for i in range(3,len(snrs)+2,4): snrs.insert(i,'|')
            print(' '.join(str(x) for x in snrs))
    return 

#c3()



def normalizeSignal(sig):
    r = np.ptp(sig)
    sig -= np.mean(sig)
    sig /= (r/2)
    return sig


def getOscillationNum(body_parts):
    bp_os_num = []
    for bp in body_parts:
        t = np.mean(bp) + np.std(bp)
        above = bp > t
        c = sum(list(1 if (above[i]==False and above[i+1]==True) else 0 for i \
                     in range(len(above)-1)))
        bp_os_num.append(c)
    return np.array(bp_os_num)

def c4():
    fs = [34,35,36,24,25,26,61,62,63,49,50,51,3,4,5]
    features = [[34,35,36],[24,25,26],[61,62,63],[49,50,51],[3,4,5]]
    #w2v = getAllBKLMotions('../Data/BKL/Mocap/Wave_two_hands/')
    bv = getAllBKLMotions('../Data/BKL/Mocap/Bending/')
    #pv = getAllBKLMotions('../Data/BKL/Mocap/Boxing/') 
    w1v = getAllBKLMotions('../Data/BKL/Mocap/Wave_one_hand/')   
    jjv = getAllBKLMotions('../Data/BKL/Mocap/Jumping_jacks/')
    ssv = getAllBKLMotions('../Data/BKL/Mocap/SitdownStandup/')
    tv = getAllBKLMotions('../Data/BKL/Mocap/Throwing/')
    
    #ms = [bv,w1v,w2v,pv]
    #mns = ['Bending','\nWave one hand','\nWave two hands','\nPunching']
    #ms = [jjv,ssv]
    ms = [tv,ssv,bv,w1v]
    #mns = ['Jumping jacks','Sitdown and Standup']
    mns = ['Throwing','\nSitdown and Standup','\nBending','\nWave one hand',]
    
    n = len(features)
    for motion_type in ms: # for each motion type
        print(mns[ms.index(motion_type)])
        for motion in motion_type[::5]: # for each motion sample
            bp_list = []
            for f in features:
                r = getBodyPartRepresentation(f,motion)
                bp_list.append(r)
                
            # body parts' relative magnitudes 
            bp_mag = list(np.std(x) for x in bp_list)
            bp_mag = np.array(bp_mag)/max(bp_mag)
            bp_mag2 = list(np.min(list(np.std(motion[x]) for x in features[i])) for i in range(n))
            bp_mag2 = np.array(bp_mag2)/max(bp_mag2)
            #print(' '.join(str(round(x,2)) for x in bp_mag2))
            
            #l = len(bp_list[0])
            #bp_pnums = list(len(sg.find_peaks_cwt(x,np.arange(int(l/10))),\
            #                        int(l/5)) for x in bp_list)
            
            #print(' '.join(str(x) for x in bp_pnums))


            #print(' '.join(str(round(x,3)) for x in bp_mag))              
            
            # body parts' relative measurement of periodicity (low freq/high freq)
            bp_freq = list(np.real(np.fft.fft(v[::4]))[3:35] for v in bp_list)
            #bp_f_ratio = list(sum(abs(x[:15]))/sum(abs(x[15:])) for x in bp_freq)
            bp_f_ratio = list(sum(abs(x[:15])) for x in bp_freq)
            bp_f_ratio = np.array(bp_f_ratio)/max(bp_f_ratio)             
            #print(' '.join(str(round(x,2)) for x in bp_f_ratio))
            #bp_f_ratio = list(np.ptp(x) for x in bp_freq)
            #bp_f_ratio = list(max(abs(x)) / np.mean(abs(x)) for x in bp_freq)
            
            bp_corr = list(np.corrcoef(bp_list[i],bp_list[j])[1,0] for i in \
                           range(n) for j in range(i+1,n))

            tmp_corr = list(np.corrcoef(motion[features[0][i]],
                            motion[features[1][i]])[1,0] for i in range(3))
            
            bp_osci = getOscillationNum(bp_list)
            print(' '.join(str(x) for x in bp_osci))             

            #print(' '.join(str(round(x,3)) for x in bp_corr)) 
            
            #m = np.median(bp_f_ratio)
            #c = list(1 if x >= m else 0 for x in bp_f_ratio)
            #bp_f_ratio = np.hstack([bp_f_ratio,np.array(c)])
                      
    return 

#c4()


def m():
    
    file_list = [open('../Data/BKL/Mocap/Wave_two_hands/skl_s01_a05_r05.txt'),
                 open('../Data/BKL/Mocap/Wave_one_hand/skl_s01_a06_r03.txt'),
                 #open('../Data/BKL/Mocap/Boxing/skl_s02_a04_r05.txt'),
                 open('../Data/BKL/Mocap/Boxing/skl_s03_a04_r03.txt'),
                 open('../Data/BKL/Mocap/Jumping_jacks/skl_s05_a02_r04.txt'),
                 open('../Data/BKL/Mocap/Clapping/skl_s11_a07_r01.txt'),
                 open('../Data/BKL/Mocap/Throwing/skl_s08_a08_r01.txt')]
    fs = [34,35,36,24,25,26,61,62,63,49,50,51,3,4,5]    

    l = ['larm','rarm','lfoot','rfoot','hip']
    n = [' wave2hands ',' wave1hand ','  punching  ',' jumpjacks  ', 
         '  clapping  ','  throwing  ']
    for f in file_list:
        print('******************* ' + n[file_list.index(f)] + '***********************')
        diff = tmp(f)
        if file_list.index(f) == 2: printMotion(diff)
        mag = []
        
        for i in fs: mag.append(max(diff[i])-min(diff[i]))
        mag = list(round(x/max(mag),2) for x in mag)
        # for i in range(3,len(mag)+1,4): mag.insert(i,'|')
        mag1 = list(round(max(mag[i:i+3]),2) for i in range(0,len(mag),3))
        # mag = ' '.join(str(x) for x in mag)
        mag1 = ' | '.join(str(x) for x in mag1)
        print(mag1)
        
        # mag = ' '.join(str(x) for x in mag)
        # print(mag)
               
               
               
feature_names = ['left wrist x', 'left wrist y', 'left wrist z', 'right wrist x',
                 'right wrist y', 'right wrist z', 'left ankles x', 
                 'left ankles y', 'left ankles z', 'right ankles x', 
                 'right ankles y', 'right ankles z', 'left hips x',
                 'left hips y', 'left hips z', 'right hips x', 'right hips y',
                 'right hips z']

#m()




def getSignificantFrequency(mfs):
    idx_list = []
    for mf in mfs:
        f = abs(mf)
        f_ma_th = max(f) * 0.7
        f = list(1 if x > f_ma_th else 0 for x in f)
        indices = list(np.where(np.array(f)==1)[0])
        idx_list += indices
        
    # if the most common one is 0, then we return 0
    idx_counts = sorted(idx_list,key=idx_list.count,reverse=True)
    if idx_counts[0] == 0 and idx_list.count(0) > len(idx_list)/2 \
       and max(idx_list) < 4: return 0
    # else if the lowest value is 0, we ignore it
    idx_list = list(filter(lambda x: x!=0, idx_list))
    idx_counts = sorted(idx_list,key=idx_list.count,reverse=True)
    return min(idx_list) #idx_counts[0]  


def printMotion4(features,motion,bp_list,path,name):
    import math
    w = 6
    h = 5
    feature_names = ['left arm','right arm', 'left leg', 'right leg', 'hip']
    fig = pt.figure(figsize=(12,7))
    f_list = []
    pca = PCA(n_components=2)
    bp_pca_list1,bp_pca_list2 = [],[]
    
    count = 0
    for i in features:
        for j in i:
            count += 1
            sp = pt.subplot(w,h,count)            
            pt.plot(motion[j]) 
            
        m = motion[i]
        #mf = np.real(np.fft.fft(m))[1:25]
        bp = np.transpose(m) # len-by-3
        bp = pca.fit_transform(bp)
        bp_pca_list1.append(np.squeeze(bp[:,0]))
        bp_pca_list2.append(np.squeeze(bp[:,1]))
        #idx = getSignificantFrequency(f_list)
        
    
    for i in range(len(bp_pca_list1)):
        sp = pt.subplot(w,h,16+i)
        pt.plot(bp_pca_list1[i])
        
    for i in range(len(bp_pca_list1)):
        sp = pt.subplot(w,h,21+i)
        fft = np.fft.fft(bp_pca_list1[i])
        pt.plot(fft[:40])
        sp.set_title(feature_names[i]) 
        
    for i in range(len(bp_pca_list1)):
        sp = pt.subplot(w,h,26+i)
        fft = np.fft.fft(bp_pca_list1[i])
        tmp = np.imag(fft)/np.real(fft)
        phase = list(math.atan(x) for x in tmp)
        pt.plot(np.array(phase)[:40])
        sp.set_title(feature_names[i]) 
        
        
        
    fig.subplots_adjust(wspace=0.4,hspace=0.4) 
    fig.suptitle(name)
    pt.show()
    #print(' '.join(str(x) for x in f_list))
    #fig.savefig(path)
    
    return None

def printMotion3(features,motion):
    w = 3 #round(np.sqrt(len(fs)),0)
    h = 5 #int(len(fs)/w+1)
    #feature_names = ['left arm','right arm', 'left leg', 'right leg', 'hip']
    pt.figure()
    snr_list = []
    r_list = []
    c_list = []
    r1_list,r2_list = [],[]
    f_sums = []
    for i in features:
        ind = features.index(i)
        m = motion[i]
        sp = pt.subplot(w,h,ind+1)
        mf = np.real(np.fft.fft(m))[1:35]
        #tmp = np.correlate(m,m,'full')
        #pt.plot(tmp)
        #l = len(tmp)
        #t1 = np.mean(tmp) + np.std(tmp)
        pt.plot(m) 
        #pt.plot(mf)
        #if not len(f_s): f_s = abs(mf)
        #else: f_s += abs(mf)
        #f_sums.append(f_s)
    #for i in range(5):
        #sp = pt.subplot(w,h,16+i)
        #pt.plot(f_sums[i])
        #c = sum(list(1 if (above[i]==False and above[i+1]==True) else 0 for i \
        #             in range(len(above)-1)))  
        #c_list.append(c)      
        sp.set_title(feature_names[ind]) 
    pt.subplots_adjust(wspace=0.4,hspace=0.4)
    #c_list = np.array(c_list)/max(c_list)
    #print(' '.join(str(round(x,2)) for x in c_list))    
    print('one motion')
    pt.show()

    return None

def getFP(fft):
    f = fft
    f = fft[::2]
    p = fft[1::2]
    ind = 1+list(abs(f[1:])).index(max(abs(f[1:])))
    tmp_f = np.copy(f[1:])
    tmp_f[ind-1] = 0
    bf = abs(f[ind])/np.mean(abs(f[1:])) # np.mean(abs(tmp_f))
    bv = ind
    return bf,bv

def getBodyPartRepresentation(features,motion):
    # Here we use the following criteria:
    #    -- features' standard deviation to features' range
    #    -- how significant is the max freq coefficient
    #    -- how significant is the first 17 frequencies to the rest
    #    -- signal to noise ratio
    
    freq_list = list(np.real(np.fft.fft(motion[f]))[:35] for f in features)
    
    std2ptp_list = list(np.std(motion[f])/np.ptp(motion[f]) for f in features)
    f_ratio1_list = list(max(abs(x))/np.mean(abs(x)) for x in freq_list)
    f_ratio2_list = list(sum(abs(x[:17]))/sum(abs(x[17:])) for x in freq_list)
    #s2n_list = list(stats.signaltonoise(motion[f]) for f in features)
    
    
    std2ptp_rank = list(i[0] for i in sorted(enumerate(std2ptp_list), key=lambda x:x[1]))
    f_ratio1_rank = list(i[0] for i in sorted(enumerate(f_ratio1_list), key=lambda x:x[1]))
    f_ratio2_rank = list(i[0] for i in sorted(enumerate(f_ratio2_list), key=lambda x:x[1]))
    #s2n_rank = list(i[0] for i in sorted(enumerate(s2n_list), key=lambda x:x[1]))
    
    scores = list(0 for x in range(len(features)))
    for i in range(len(features)):
        scores[i] += (std2ptp_rank.index(i) + f_ratio1_rank.index(i) + \
                      f_ratio2_rank.index(i) )#+ s2n_rank.index(i))
        
    f = scores.index(max(scores))
    return motion[features[f]]

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


def test1():

    # BKL --> BKL Acce
    # left arm/wrist, right arm/wrist, left foot/ankles, 
    # right foot/ankles, left up leg/hips, right up leg/hips
    names = ['Wave two hands','Wave one hand','Boxing', 'Jumping jacks',\
               'Throwing','Bending']    
    feature_names = ['left fore arm/wrist', 'right fore arm/wrist', 'left foot/ankles', 
     'right foot/ankles', 'left up leg/hips']
    source_f_indices = [[34,35,36],[24,25,26],[61,62,63],[49,50,51],[3,4,5]]
    target_f_indices = [[0,1,2],[3,4,5],[12,13,14],[15,16,17],[6,7,8]]
    # ********** BKL MoCap **********
    
    # left arm: left shoulder, forearm, arm
    larm = [[31,32,33],[37,38,39],[34,35,36]]
    # right arm: right shoulder, forearm, arm
    rarm = [[21,22,23],[27,28,29],[24,25,26]]
    # left leg: left up leg, leg, foot
    lleg = [[53,54,55],[57,58,59],[61,62,63]]
    # right leg: right up leg, leg, foot
    rleg = [[41,42,43],[45,46,47],[49,50,51]]
    # body: head, neck, spine, spine1, spine2, hips(pos), hip
    body = [[6,7,8],[9,10,11],[12,13,14],[0,1,2],[3,4,5]]
    joint = [larm,rarm,lleg,rleg,body]
    
    source_fft, motion_diff = getBKLMotion()

    #fft,motion_diff = getCMUMotion(["../Data/CMU/Walking/","../Data/CMU/Jumping/"])

    ind1,ind2,ind3,ind4 = 0,1,2,3
    for motion_type in motion_diff[:1]:
        name = names[motion_diff.index(motion_type)]
        print(name) 
        count = 0
        for i in np.arange(0,40,7):
            count += 1
            #pt.figure(figsize=(22,15))
            #pt.tight_layout(pad=0.7, w_pad=0.7, h_pad=0.7)
            #pt.suptitle(names[motion_diff.index(motion_type)])
            motion = motion_type[i] # 0 Walking 1 Jumping
            p = getPCA(larm[0],motion)[0]
            pt.figure()
            
            for j in range(len(larm[0])):
                pt.subplot(2,4,j+1)
                pt.plot(motion[larm[0][j]])
            pt.subplot(2,4,4)
            pt.plot(p[0])
            
            p = getPCA(rarm[0],motion)[0]
            for j in range(len(rarm[0])):
                pt.subplot(2,4,j+5)
                pt.plot(motion[rarm[0][j]])
            pt.subplot(2,4,8)
            pt.plot(p[0]) 
            
            pt.show()
            #for p in range(len(joint[ind1])): # for one body part
            #    pca = getPCA(joint[ind1][p],motion)
            #    for q in range(len(joint[ind1][p])): # for each fine grained feature
            #        pt.subplot(7,6,p*6+q+1)
            #        pt.plot(pca[joint[ind1][p][q]])
                    
            '''
            for p in range(len(joint[ind2])): # for another body part
                for q in range(len(joint[ind2][p])): # for each fine grained feature
                    pt.subplot(7,6,p*6+q+4)
                    pt.plot(motion[joint[ind2][p][q]])
                    
            for p in range(len(joint[ind3])): # for another body part
                for q in range(len(joint[ind3][p])): # for each fine grained feature
                    pt.subplot(7,6,(p+4)*6+q+1)
                    pt.plot(motion[joint[ind3][p][q]])
            for p in range(len(joint[ind4])): # for another body part
                for q in range(len(joint[ind4][p])): # for each fine grained feature
                    pt.subplot(7,6,(p+4)*6+q+4)
                    pt.plot(motion[joint[ind4][p][q]])
            #pt.tight_layout(w_pad=0.1, h_pad=0.1)
            st = '/Users/xiahu/Documents/Projects/Transfer/Expe/BKLMocap_figs/' + name+'_' + str(count) + '.png'
            pt.savefig(st,dpi=300)
            '''
        
    
#test1()


def test():

    # BKL --> BKL Acce
    # left arm/wrist, right arm/wrist, left foot/ankles, 
    # right foot/ankles, left up leg/hips, right up leg/hips
    names = ['Wave two hands','Wave one hand','Boxing', 'Jumping jacks',\
               'Throwing','Bending']    
    feature_names = ['left fore arm/wrist', 'right fore arm/wrist', 'left foot/ankles', 
     'right foot/ankles', 'left up leg/hips']
    source_f_indices = [[34,35,36],[24,25,26],[61,62,63],[49,50,51],[3,4,5]]
    target_f_indices = [[0,1,2],[3,4,5],[12,13,14],[15,16,17],[6,7,8]]
    # ********** BKL MoCap **********
    
    # left arm: left shoulder, forearm, arm
    larm = [[31,32,33],[37,38,39],[34,35,36]]
    # right arm: right shoulder, forearm, arm
    rarm = [[21,22,23],[27,28,29],[24,25,26]]
    # left leg: left up leg, leg, foot
    lleg = [[53,54,55],[57,58,59],[61,62,63]]
    # right leg: right up leg, leg, foot
    rleg = [[41,42,43],[45,46,47],[49,50,51]]
    # body: head, neck, spine, spine1, spine2, hips(pos), hip
    body = [[6,7,8],[9,10,11],[12,13,14],[0,1,2],[3,4,5]]
    joint = [larm,rarm,lleg,rleg,body]
    
    source_fft, motion_diff = getBKLMotion()

    #fft,motion_diff = getCMUMotion(["../Data/CMU/Walking/","../Data/CMU/Jumping/"])

    ind1,ind2,ind3,ind4 = 0,1,2,3
    for motion_type in motion_diff[:]:
        name = names[motion_diff.index(motion_type)]
        print(name) 
        count = 0
        for i in np.arange(0,40,7):
            count += 1
            pt.figure(figsize=(22,15))
            pt.tight_layout(pad=0.7, w_pad=0.7, h_pad=0.7)
            pt.suptitle(names[motion_diff.index(motion_type)])
            motion = motion_type[i] # 0 Walking 1 Jumping
            for p in range(len(joint[ind1])): # for one body part
                #p = getPCA(joint[j],motion)
                for q in range(len(joint[ind1][p])): # for each fine grained feature
                    pt.subplot(7,6,p*6+q+1)
                    pt.plot(motion[joint[ind1][p][q]])
            for p in range(len(joint[ind2])): # for another body part
                for q in range(len(joint[ind2][p])): # for each fine grained feature
                    pt.subplot(7,6,p*6+q+4)
                    pt.plot(motion[joint[ind2][p][q]])
                    
            for p in range(len(joint[ind3])): # for another body part
                for q in range(len(joint[ind3][p])): # for each fine grained feature
                    pt.subplot(7,6,(p+4)*6+q+1)
                    pt.plot(motion[joint[ind3][p][q]])
            for p in range(len(joint[ind4])): # for another body part
                for q in range(len(joint[ind4][p])): # for each fine grained feature
                    pt.subplot(7,6,(p+4)*6+q+4)
                    pt.plot(motion[joint[ind4][p][q]])
            #pt.tight_layout(w_pad=0.1, h_pad=0.1)
            st = '/Users/xiahu/Documents/Projects/Transfer/Expe/BKLMocap_figs/' + name+'_' + str(count) + '.png'
            pt.savefig(st,dpi=300)
        
    
#test()


def m1():
    fs = [34,35,36,24,25,26,61,62,63,49,50,51,3,4,5]
    #pf = getAllBKLMotions('../Data/BKL/Mocap/Boxing/')
    w2f = getAllBKLMotions('../Data/BKL/Mocap/Wave_two_hands/')
    #jjf = getAllBKLMotions('../Data/BKL/Mocap/Jumping_jacks/')
    #ssf = getAllBKLMotions('../Data/BKL/Mocap/SitdownStandup/')
    #w1f = getAllBKLMotions('../Data/BKL/Mocap/Wave_one_hand/')
    #tf = getAllBKLMotions('../Data/BKL/Mocap/Throwing/')
    #cf = getAllBKLMotions('../Data/BKL/Mocap/Clapping/')
    #bf = getAllBKLMotions('../Data/BKL/Mocap/Bending/')
    features = [[34,35,36],[24,25,26],[61,62,63],[49,50,51],[3,4,5]]
    #features = [[34,35,36],[24,25,26],[61,62,63],[49,50,51],[3,4,5]]
    #features = [34,35,36,24,25,26,61,62,63,49,50,51,3,4,5]
    #m_names = ['tf','pf','w1f','jjf','ssf']
    #m_list = [tf,pf,w1f,jjf,ssf]
    #m_names = ['Wave one hand','Bending','JumpingJacks','SitdownStandup','Throwing','Punching']
    #m_list = [w1f,bf,jjf,ssf,tf,pf]
    #m_names = ['SitdownStandup','Bending']
    #m_list = [ssf,bf]
    #m_names = ['Jumping jacks','Sitdown and standup']
    #m_list = [jjf,ssf]
    m_names = ['Wave two hands']
    m_list = [w2f] 
    
    for mo in m_list:
        print('\n',m_names[m_list.index(mo)])
        count = 0
        for motion in mo[::4]: # for each motion sample
            bp_list = []
            bp_max_list = []
            bp_min_list = []
            for f in features:
                r = getBodyPartRepresentation(f,motion)
                bp_list.append(r)   
                ffts = list(np.real(np.fft.fft(m))[1:25] for m in motion[f,:])
                fft_ranges = list(np.ptp(x) for x in ffts)
                bp_max_list.append(max(fft_ranges))
                bp_min_list.append(min(fft_ranges))
                
            result = list(round(bp_max_list[i]/bp_min_list[i],1) for i in range(len(bp_max_list)))
            print(' '.join(str(x) for x in result))
            path = 'BKL/' + m_names[m_list.index(mo)] + '/fig' + str(count) + '.png' 
            printMotion4(features,motion,bp_list,path,m_names[m_list.index(mo)])
            count += 1
    return None

#m1()


def tmp(file):

    fft,diff = getOneBKL(file)
    mat = getMotionCorr(fft)
    #var_list = []
    #for i in range(len(mat)):
    #    var_list.append(np.var(mat[i,:]))
    #th = max(var_list)*0.9
    #l1 = list((var_list.index(x),x) for x in var_list if x > th)
    #l2 = sorted(l1,key=lambda x: x[1],reverse=True)
    #l3 = list(x[0] for x in l2)        
    #print(l3)
    return diff
    
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


def testPCA():
    # left arm: left shoulder, forearm, arm
    larm = [[31,32,33],[37,38,39],[34,35,36]]
    # right arm: right shoulder, forearm, arm
    rarm = [[21,22,23],[27,28,29],[24,25,26]]
    # left leg: left up leg, leg, foot
    lleg = [[53,54,55],[57,58,59],[61,62,63]]
    # right leg: right up leg, leg, foot
    rleg = [[41,42,43],[45,46,47],[49,50,51]]
    # body: head, neck, spine, spine1, spine2, hips(pos), hip
    body = [[6,7,8],[9,10,11],[12,13,14],[0,1,2],[3,4,5]]
    joint = [larm,rarm,lleg,rleg,body]

    
    names = ['wave2hands','wave1hand','boxing','jumpingjacks','throwing','bending']
    
    source_fft, motion_diff = getBKLMotion()
    
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
    for motion_type in np.array(motion_diff)[[2,3,5]]:
        
        corr_list.append([])
        count = 0
        for motion in motion_type[::10]:
            print(names[motion_diff.index(motion_type)])
            pca1,c1 = getPCA(larm[0],motion)
            pca2,c2 = getPCA(rarm[0],motion)
            corr1 = np.corrcoef(c1[0],c2[0])[1,0]
            corr2 = np.corrcoef(c1[1],c2[1])[1,0]
            corr3 = np.corrcoef(c1[2],c2[2])[1,0]
            print('CORR',corr1,corr2,corr3)
            pca_corr1 = np.corrcoef(pca1[0],pca2[0])[1,0]
            print('axis corr', corr1, 'pca corr', pca_corr1, 'corr', corr1 * pca_corr1)
            
        
            from mpl_toolkits.mplot3d import Axes3D
            import pickle
            mat1,mat2 = motion[larm[0]],motion[[rarm[0]]]#pca1,pca2
            
            fig = pt.figure() 
            ax = fig.add_subplot(1,3,1)
            ax.set_title(names[motion_diff.index(motion_type)])
            ax.plot(mat1[0],mat1[2])
            ax.plot(mat2[0],mat2[2])
            ax2 = fig.add_subplot(1,3,2)
            ax2.set_title('feature 0')
            ax2.plot(np.arange(len(mat1[0])),mat1[0])
            ax2.plot(np.arange(len(mat2[0])),mat2[0])
            
            ax3 = fig.add_subplot(1,3,3)
            ax3.set_title('feature 2')
            ax3.plot(np.arange(len(mat1[2])),mat1[2])
            ax3.plot(np.arange(len(mat2[2])),mat2[2])            
            
            #ax.plot([0,c1[0][0]],[0,c1[0][1]],[0,c1[0][2]])
            #ax.plot(mat2[0],mat2[1],mat2[2])
            pt.show()
            #ax.plot(np.arange(len(mat2[0])),mat2[0],mat2[1])
            
            '''
            fig = pt.figure() 
            ax = fig.add_subplot(1,1,1,projection='3d')
            ax.set_title(names[motion_diff.index(motion_type)])
            ax.plot(mat1[0],mat1[1],mat1[2])
            ax.plot([0,c1[0][0]],[0,c1[0][1]],[0,c1[0][2]])
            ax.plot(mat2[0],mat2[1],mat2[2])
            ax.plot([0,c2[0][0]],[0,c2[0][1]],[0,c2[0][2]])
            #ax.plot([0,c1[0][0]],[0,c1[1][0]],[0,c1[2][0]])
            #ax.plot([0,c1[1][0]],[0,c1[1][1]],[0,c1[1][2]])
            #ax.plot([0,c1[2][0]],[0,c1[2][1]],[0,c1[2][2]])
            pt.show()
            #ax.plot(np.arange(len(mat2[0])),mat2[0],mat2[1])
            '''
            
            '''
            fig = pt.figure(figsize = (12,4))    
            ax1 = fig.add_subplot(1,3,1, projection='3d')
            ax1.set_title(names[motion_diff.index(motion_type)])
            ax1.plot(np.arange(len(mat1[0])),mat1[0],mat1[1])
            ax1.plot(np.arange(len(mat2[0])),mat2[0],mat2[1])
            
            ax2 = fig.add_subplot(1,3,2)
            ax2.set_title('feature 0')
            ax2.plot(np.arange(len(mat1[0])),mat1[0])
            ax2.plot(np.arange(len(mat2[0])),mat2[0])
            
            ax3 = fig.add_subplot(1,3,3)
            ax3.set_title('feature 1')
            ax3.plot(np.arange(len(mat1[1])),mat1[1])
            ax3.plot(np.arange(len(mat2[1])),mat2[1])   
            '''
        
            
            '''
            fig = pt.figure(figsize = (12,4))
            ax1 = fig.add_subplot(1,4,1, projection='3d')
            ax1.set_title(names[motion_diff.index(motion_type)])
            
            #ax.plot([0,c1[0][0]],[0,c1[0][1]],[0,c1[0][2]])
            ax1.plot(pca1[0],pca1[1],pca1[2])
            #ax.plot(np.arange(len(pca1[2])),pca1[0],pca1[1])
            #ax.plot([0,c2[0][0]],[0,c2[0][1]],[0,c2[0][2]])
            ax1.plot(pca2[0],pca2[1],pca2[2])
            #ax.plot(np.arange(len(pca2[2])),pca2[0],pca2[1]) 
            
            ax2 = fig.add_subplot(1,4,2)
            ax2.set_title('features 0 & 1')
            ax2.plot(pca1[0],pca1[1])
            ax2.plot(pca2[0],pca2[1])
            
            ax3 = fig.add_subplot(1,4,3)
            ax3.set_title('features 0 & 2')
            ax3.plot(pca1[0],pca1[2])
            ax3.plot(pca2[0],pca2[2])
            
            ax4 = fig.add_subplot(1,4,4)
            ax4.set_title('features 1 & 2')
            ax4.plot(pca1[1],pca1[2])
            ax4.plot(pca2[1],pca2[2])

            '''
            
            '''
            #ax.plot(np.arange(len(motion[rarm_s[0][2]])),motion[rarm_s[0][2]],motion[rarm_s[0][0]])
            from mpl_toolkits.mplot3d import Axes3D
            import pickle
            fig = pt.figure(figsize = (12,4))
            ax1 = fig.add_subplot(1,4,1, projection='3d')
            ax1.set_title(names[motion_diff.index(motion_type)])
            ax1.plot(motion[larm[0][0]],motion[larm[0][1]],motion[larm[0][2]])
            ax1.plot(motion[rarm[0][0]],motion[rarm[0][1]],motion[rarm[0][2]])

            
            ax2 = fig.add_subplot(1,4,2)
            ax2.set_title('features 0 & 1')
            ax2.plot(motion[larm[0][0]],motion[larm[0][1]])
            ax2.plot(motion[rarm[0][0]],motion[rarm[0][1]])
            
            ax3 = fig.add_subplot(1,4,3)
            ax3.set_title('features 0 & 2')
            ax3.plot(motion[larm[0][0]],motion[larm[0][2]])
            ax3.plot(motion[rarm[0][0]],motion[rarm[0][2]])
            
            ax4 = fig.add_subplot(1,4,4)
            ax4.set_title('features 1 & 2')
            ax4.plot(motion[larm[0][1]],motion[larm[0][2]])
            ax4.plot(motion[rarm[0][1]],motion[rarm[0][2]])
            
            
            pt.show()
            #count += 1
            #name_str = 'FigureObject' + str(count) + '.fig.pickle'
            #pickle.dump(fig, open(name_str, 'wb'))
            
            #print('pca1',pca1)
            pca_rgs1 = [np.ptp(pca1[0]),np.ptp(pca1[1]),np.ptp(pca1[2])]
            pca_rgs2 = [np.ptp(pca2[0]),np.ptp(pca2[1]),np.ptp(pca2[2])]
            pca_rgs = np.array(pca_rgs1) + np.array(pca_rgs2)
            pca_rgs = list(x / pca_rgs.max() for x in pca_rgs)
            
            #print('pca_rgs1',pca_rgs1)
            corr1 = np.correlate(c1[0],c2[0])[0]
            corr2 = np.correlate(c1[1],c2[1])[0]
            corr3 = np.correlate(c1[2],c2[2])[0] 
            #print(corr1+corr2+corr3)
            
            corr_list[-1].append(corr1)
            print('CORR',corr1,corr2,corr3)
            '''
            
    '''
    pt.figure()
    print(corr_list[0])
    pt.hist(corr_list[0])
    pt.hist(corr_list[1])
    pt.hist(corr_list[2])  
    pt.hist(corr_list[3])  
    pt.hist(corr_list[4])  
    pt.hist(corr_list[5])  
    #pt.hist(np.abs(corr_list[0]))
    #pt.hist(np.abs(corr_list[1]))
    #pt.hist(np.abs(corr_list[2]))
    '''
    
    #pt.show()
            
    return

#testPCA()

def testEachJoint():
    # left arm: left shoulder, forearm, arm
    larm = [[31,32,33],[37,38,39],[34,35,36]]
    # right arm: right shoulder, forearm, arm
    rarm = [[21,22,23],[27,28,29],[24,25,26]]
    # left leg: left up leg, leg, foot
    lleg = [[53,54,55],[57,58,59],[61,62,63]]
    # right leg: right up leg, leg, foot
    rleg = [[41,42,43],[45,46,47],[49,50,51]]
    # body: head, neck, spine, spine1, spine2, hips(pos), hip
    body = [[6,7,8],[9,10,11],[12,13,14],[0,1,2],[3,4,5]]
    joint = [larm,rarm,lleg,rleg,body]

    
    names = ['wave2hands','wave1hand','boxing','jumpingjacks','throwing','bending']
    names = ['wave2hands','boxing','jumpingjacks','bending']
    source_fft, motion_diff = getBKLMotion()
    print('len(motion_diff)',len(motion_diff))
    print('type(motion_diff[0])',type(motion_diff[0]))
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
        
        return bp,c,pca
    
    def set_axes(ax):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
    # test a pair of joint
    for motion_type in motion_diff:
        
        corr_list.append([])
        count = 0
        for motion in motion_type[::13]:
            print(names[motion_diff.index(motion_type)])
            motion[rarm[0]] = -motion[rarm[0]]
            motion[rarm[1]] = -motion[rarm[1]]
            motion[rarm[2]] = -motion[rarm[1]]
            
            new_mat1,c1,pca1 = getPCA(larm[0],motion)
            new_mat2 = pca1.transform(np.transpose(motion[rarm[0]]))
            new_mat2 = np.transpose(new_mat2)
            print('shape',new_mat1.shape,new_mat2.shape)
            #pca2,c2 = getPCA(rarm[0],motion)
            
            from mpl_toolkits.mplot3d import Axes3D
            import pickle
            mat1,mat2 = motion[larm[0]],motion[rarm[0]]#pca1,pca2
            mat3 = motion[larm[0]]
            fig = pt.figure(figsize=(15,5)) 
            ax = fig.add_subplot(1,5,1,projection='3d')
            ax.set_title(names[motion_diff.index(motion_type)]+' arms')
            #ax.set_title('shoulder')
            
            ax.plot(mat1[0],mat1[1],mat1[2])
            ax.plot(mat2[0],mat2[1],mat2[2])
            set_axes(ax)
            ax2 = fig.add_subplot(1,5,2)
            ax2.set_title('x')
            ax2.plot(np.arange(len(mat1[0])),mat1[0])
            ax2.plot(np.arange(len(mat2[0])),mat2[0])
            ax3 = fig.add_subplot(1,5,3)
            ax3.set_title('y')
            ax3.plot(np.arange(len(mat1[0])),mat1[1])
            ax3.plot(np.arange(len(mat2[0])),mat2[1])
            ax4 = fig.add_subplot(1,5,4)
            ax4.set_title('z')
            ax4.plot(np.arange(len(mat1[0])),mat1[2])
            ax4.plot(np.arange(len(mat2[0])),mat2[2])  
            
            ax5 = fig.add_subplot(1,5,5)
            ax5.set_title('pca')
            ax5.plot(np.arange(len(new_mat1[0])),new_mat1[0])
            ax5.plot(np.arange(len(new_mat2[0])),new_mat2[0])
            
            '''
            ax2 = fig.add_subplot(1,4,2,projection='3d')
            ax2.set_title('forearm')
            ax2.plot(mat2[0],mat2[1],mat2[2])
            set_axes(ax2)
            
            ax3 = fig.add_subplot(1,4,3,projection='3d')
            ax3.set_title('arm')
            ax3.plot(mat3[0],mat3[1],mat3[2])            
            set_axes(ax3)
            
            ax4 = fig.add_subplot(1,4,4,projection='3d')
            ax4.set_title('total')
            ax4.plot(mat1[0]+mat2[0]+mat3[0],mat1[1]+mat2[1]+mat3[1],mat1[2]+mat2[2]+mat3[2])    
            set_axes(ax4)
            '''
            pt.show()
    return

#testEachJoint()


def testJointCorr():
    # left arm: left shoulder, forearm, arm
    larm = [[31,32,33],[37,38,39],[34,35,36]]
    # right arm: right shoulder, forearm, arm
    rarm = [[21,22,23],[27,28,29],[24,25,26]]
    # left leg: left up leg, leg, foot
    lleg = [[53,54,55],[57,58,59],[61,62,63]]
    # right leg: right up leg, leg, foot
    rleg = [[41,42,43],[45,46,47],[49,50,51]]
    # body: head, neck, spine, spine1, spine2, hips(pos), hip
    body = [[6,7,8],[9,10,11],[12,13,14],[0,1,2],[3,4,5]]
    joint = [larm,rarm,lleg,rleg,body]

    
    names = ['wave2hands','wave1hand','boxing','jumpingjacks','throwing','bending']
    names = ['boxing','jumpingjacks','bending']
    source_fft, motion_diff = getBKLMotion()
    print('len(motion_diff)',len(motion_diff))
    print('type(motion_diff[0])',type(motion_diff[0]))
    corr_list = []
    
    # test a pair of joint
    larm = list(itertools.chain(*larm))
    rarm = list(itertools.chain(*rarm))
    joint = larm + rarm
    
    for motion_type in motion_diff:
        corrs = np.zeros((18,18),dtype=np.float64)
        for motion in motion_type[::13]:
            for i in range(18):
                for j in range(18):
                    corrs[i][j] = np.corrcoef(motion[joint[i]],motion[joint[j]])[1,0]
            pt.figure()
            pt.title(names[motion_diff.index(motion_type)])
            pt.imshow(corrs)
            pt.colorbar()
            pt.show()
        
    return
    
#testJointCorr()

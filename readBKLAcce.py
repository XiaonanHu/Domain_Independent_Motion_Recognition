import numpy as np
import matplotlib.pyplot as pt
#from scipy import signal
import scipy.signal as sg
import scipy.stats as stats
from os import listdir
from os.path import isfile, join
import math as mt
from sgfilter import savitzky_golay
import itertools
from sklearn.decomposition import IncrementalPCA, PCA





# create coarser grained representation?
# learn the right scale of representation?
# segment motions into several pieces and find the change of their correlation

feature_names = ['left wrist x', 'left wrist y', 'left wrist z', 'right wrist x',
                 'right wrist y', 'right wrist z', 'left ankles x', 
                 'left ankles y', 'left ankles z', 'right ankles x', 
                 'right ankles y', 'right ankles z', 'left hips x',
                 'left hips y', 'left hips z', 'right hips x', 'right hips y',
                 'right hips z']


def getOneBKL(file):
        
    motionVal = np.zeros((0,0),dtype=np.float64)
    for line in file:
        line = list(float(x) for x in line.split()[:3])
        if not len(motionVal): motionVal = np.array(line)
        else: motionVal = np.vstack([motionVal,np.array(line)])
    
    motionVal = np.transpose(motionVal)
    fft_len = 15
    fft = np.empty((0,fft_len*2),dtype=np.float64)
    if not len(motionVal): return [],[]
    
    diffs = np.empty((0,len(motionVal[0])-1),dtype=np.float64)
    motionVal1 = np.empty((0,len(motionVal[0])),dtype=np.float64)

    for i in range(len(motionVal)):
        m = motionVal[i]
        m = savitzky_golay(m,121,3)
        m = np.diff(m)
        m = savitzky_golay(m,121,3)
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
        if (max(m)-min(m)) != 0:
            m = m / (max(m) - min(m))
        fft = np.append(fft, np.array([m]),axis=0)
    return fft, diffs


    
def readMotions():
    dirroot = '../Data/BKL/BerkeleyMHAD/Accelerometer/'
    dirnames =['Shimmer02/','Shimmer03/','Shimmer04/','Shimmer05/','Shimmer06/']

    first_dir = dirroot + 'Shimmer01/'
    files = [f for f in listdir(first_dir) if isfile(join(first_dir,f))]
    
    motion_list = list([] for i in range(len(files)))
    for file in files:
        motion_list[files.index(file)].append(dirroot+'Shimmer01/'+file)
        name = file[file.find('s'):].rstrip('.txt')
        for d in dirnames:
            dirname = dirroot + d
            fs = [f for f in listdir(dirname) if isfile(join(dirname,f))]
            idx = list(name in f for f in fs).index(True)
            motion_list[files.index(file)].append(dirname + fs[idx])
    
    motion_list = motion_list[1:]
            
    return motion_list




def getFeatureFluctuation(traj):
    t = np.diff(traj)
    t1 = np.diff(t)
    t1 = t1 / (max(t1)-min(t1)) * len(t1)
    l = list(1 if np.sign(t[i+1]) != np.sign(t[i]) and t1[i] > 0.4
             else 0 for i in range(len(t)-1))
    return sum(l)



def filtAmp(m):
    f = np.fft.fft(m)
    ma = max(f)
    mi = min(f)
    f = np.array(list(x if x > ma*0.15 or x < mi*0.15 else 0 for x in f))
    m = np.fft.ifft(f)
    return m



def printMotionDiff(motion):
    w = round(np.sqrt(len(motion)),0)
    h = int(len(motion)/w+1)
    
    pt.figure()
    tmp = []
    for i in range(len(motion)):
        sp = pt.subplot(w,h,i+1)
        pt.plot(np.diff(motion[i]))
        sp.set_title(feature_names[i])
        tmp.append(m)
    pt.subplots_adjust(wspace=0.4,hspace=0.4)
    pt.show()

    return tmp


def printMotion(motion):
    w = round(np.sqrt(len(motion)),0)
    h = int(len(motion)/w+1)
    pt.figure()
    tmp = []
    var_list = []
    fs = [0,1,2,3,4,5,12,13,14,15,16,17,6,7,8]
    for i in range(len(fs)):
        sp = pt.subplot(w,h,i+1)
        #m = np.abs(np.fft.fft(integrate(motion[fs[i]])))[:30]
        m = integrate(motion[fs[i]])
        v = sum(abs(m[1:18]))/sum(abs(m[1:]))
        var_list.append(round(v,2))  
        
        #m = np.cumsum(m)
        #win = signal.hann(10)
        #m = signal.convolve(m,win,mode='same')/sum(win)  
        pt.plot(m[1:])
        sp.set_title(feature_names[i])
    print('variances: ', ' '.join(str(x) for x in var_list))
    pt.subplots_adjust(wspace=0.4,hspace=0.4)
    pt.show()
    return tmp


def printMotion1(motion):
    w = round(np.sqrt(len(motion)),0)
    h = int(len(motion)/w+1)
    #pt.figure()
    #for i in range(len(motion)):
    #    pt.subplot(w,h,i+1)
        #m = savitzky_golay(motion[i],11,3)
    #    pt.plot(motion[i])
    #pt.show()
    
    pt.figure()
    tmp = []
    for i in range(len(motion)):
        sp = pt.subplot(w,h,i+1)

        #m = filtAmp(motion[i])
        #win = signal.hann(10)
        #m = signal.convolve(motion[i],win,mode='same')/sum(win)  
        pt.plot(motion[i][::2])
        sp.set_title(feature_names[i])
    pt.subplots_adjust(wspace=0.4,hspace=0.4)
    pt.show()
    
    
    pt.figure()
    tmp = []
    for i in range(len(motion)):
        sp = pt.subplot(w,h,i+1)

        #m = filtAmp(motion[i])
        #win = signal.hann(10)
        #m = signal.convolve(motion[i],win,mode='same')/sum(win)  
        pt.plot(motion[i][1::2])
        sp.set_title(feature_names[i])
    pt.subplots_adjust(wspace=0.4,hspace=0.4)
    pt.show()    
    
    #tmp = np.vstack(tmp)
    return tmp



def integrate(m):
    m1 = [m[0]]
    for i in range(1,len(m)): m1.append(m1[-1]+1/30*m[i]) # 1/30*m[i] 
    m = m1[1:]
        
    m = np.array(m)
    x = np.arange(len(m))
    p = np.poly1d(np.polyfit(x,m,2))
    r = p(x)
    if min(r) > 0:
        r = list(x-min(r) for x in r)
        
    #A = np.vstack([np.arange(len(m)),np.ones(len(m))]).T
    #s,intersect = np.linalg.lstsq(A,m)[0]
    #end = intersect + (len(m)-1)*s
    #r = np.linspace(intersect,end,len(m))
    
    m = m - r
    return m


def getFFT(motionVal):
    fft_len = 30
    fft = np.empty((0,fft_len*2),dtype=np.float64)
    fourier = np.empty((0,fft_len),dtype=np.float64)
    if not len(motionVal) or len(motionVal[0])<41: return [],[]
    check = [0] * len(motionVal)
    for i in range(len(motionVal)):
        check[i] = abs(0.98-motionVal[i][0])
    ind = check.index(min(check))
    motionVal[ind] = motionVal[ind] - 0.98
    
    vals = np.empty((0,len(motionVal[0])-1),dtype=np.float64)
    for i in range(len(motionVal)):
        m = motionVal[i]     
        m = savitzky_golay(m,23,3)
        m = integrate(m)
        m = savitzky_golay(m,23,3)
        vals = np.vstack([vals,m]) 
        
        m = np.fft.fft(m)
        m = m[1:fft_len+1]
        # interlacing the real coefficients and the imaginary coefficients
        a = np.real(m)
        b = np.imag(m)
        m = np.empty((a.size + b.size,), dtype = a.dtype)
        m[0::2] = a
        m[1::2] = b
        if (max(m)-min(m)) != 0: m = m / (max(m) - min(m))
        fft = np.append(fft, np.array([m]),axis=0)
        fourier = np.append(fourier,np.array([a]),axis=0)
    return fft, vals



def getAcceMotions(motions):
    fft_list = []
    val_list = []
    phase_list = []
    for motion in motions:
        joint_list = []
        for file in motion:
            f = open(file,'r')
            joint = np.empty((0,0))
            for line in f:
                line = list(float(x) for x in line.split()[:3])
                if not len(joint): joint = np.array(line)
                else: joint = np.vstack([joint, np.array(line)])
            joint_list.append(joint)
            f.close()
        mi = min(list(x.shape[0] for x in joint_list))
        motionVal = np.transpose(np.hstack(list(x[:mi,:] for x in joint_list)))
        fft,val = getFFT(motionVal)
        if len(fft): 
            fft_list.append(fft)
            val_list.append(val)
    return fft_list, val_list



def getAcceMotion():
    ml = readMotions()
    wave_two_hands = []
    wave_one_hand = []
    clapping = []
    punching = []
    jumping = []
    jumping_jacks = []
    bending = []
    throwing = []
    sitdown_and_standup = []
    sitdown = []
    standup = []
    T_pose = []
    
    # getting wave two hands
    for motion in ml:
        if 'a05' in motion[0]:
            wave_two_hands.append(motion)
        elif 'a06' in motion[0]:
            wave_one_hand.append(motion)
        elif 'a07' in motion[0]:
            clapping.append(motion)
        elif 'a04' in motion[0]:
            punching.append(motion)
        elif 'a01' in motion[0]:
            jumping.append(motion)
        elif 'a02' in motion[0]:
            jumping_jacks.append(motion)
        elif 'a03' in motion[0]:
            bending.append(motion)
        elif 'a08' in motion[0]:
            throwing.append(motion)
        elif 'a09' in motion[0]:
            sitdown_and_standup.append(motion)
        elif 'a10' in motion[0]:
            sitdown.append(motion)
        elif 'a11' in motion[0]:
            standup.append(motion)
        elif 'a12' in motion[0]:
            T_pose.append(motion)
    fft_list = []
    val_list = []
    phase_list = []
    m_list = [jumping, jumping_jacks, bending, punching, wave_two_hands,\
              wave_one_hand, clapping, throwing, sitdown_and_standup, \
              sitdown, standup, T_pose]
    m_list = [wave_two_hands,wave_one_hand,punching,jumping_jacks,throwing,bending]#,sitdown_and_standup]#sitdown,standup,jumping]
     
    for motion in m_list:
        f,v = getAcceMotions(motion)
        fft_list.append(f)
        val_list.append(v)
    return fft_list,val_list
    




def getAllBKLMotions(dir_list):
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
    
    
def getFeatureReleMag1(features, diffs):
    total_features = []
    for diff in diffs: # for each motion type
        feature_list = []
        for motion in diff: # for each motion sample
            rep = np.zeros(len(features),dtype=np.float64)
            for i in range(len(features)):
                mg = max(motion[features[i],:])- min(motion[features[i],:])
                rep[i] = mg
            if rep.max() > 0: feature_list.append(rep/rep.max())
            else: feature_list.append(rep)
        total_features.append(feature_list)
    return np.vstack(total_features)






# 31, 32, 33, 44, 45, 46, 50, 59, 61, 73, 74, 75
# **********************
#file = open('../Data/BKL/BerkeleyMHAD/Accelerometer/skl_s01_a08_r01.txt')
#file = open('../Data/BKL/Mocap/Throwing/skl_s01_a08_r01.txt')
#file = open('../Data/BKL/Mocap/Throwing/skl_s02_a08_r03.txt')
#file = open('../Data/BKL/Mocap/Throwing/skl_s08_a08_r01.txt')
# **********************
#file = open('../Data/BKL/Mocap/Wave_two_hands/skl_s01_a05_r05.txt')
#file = open('../Data/BKL/Mocap/Wave_two_hands/skl_s07_a05_r02.txt')
#file = open('../Data/BKL/Mocap/Wave_two_hands/skl_s11_a05_r01.txt')
# ********************f** 
#file = open('../Data/BKL/Mocap/Clapping/skl_s03_a07_r04.txt')
#file = open('../Data/BKL/Mocap/Clapping/skl_s06_a07_r05.txt')
#file = open('../Data/BKL/Mocap/Clapping/skl_s11_a07_r01.txt')
# **********************
#file = open('../Data/BKL/Mocap/Wave_one_hand/skl_s01_a06_r03.txt')
#file = open('../Data/BKL/Mocap/Wave_one_hand/skl_s05_a06_r03.txt')
#file = open('../Data/BKL/Mocap/Wave_one_hand/skl_s10_a06_r04.txt')
# **********************




def tmp1():

    m = getAcceMotion()
    ffts, ms = m #,sdv,suv,jv] = m
    #for i in range(0,50,5):
    #    printMotion(cv[i])
    #for i in range(0,50,5): 
    #    printMotion(w2v[i])
    fs = [0,1,2,3,4,5,12,13,14,15,16,17,6,7,8]

    mns = ["Wave two hands","Wave one hand","Punching","Jumping Jacks",\
           "Throwing","Bending"]
    #mns = ['Throwing','Sitdown and standup','Jumping jacks','Wave two hands',\
    #       'Wave one hand','Punching','Bending','Sit down','Stand up','Jumping']
    #ms = [tv,ssv,jjv,w2v,w1v,pv,bv,sdv,suv,jv]

    for motion in ms:
        print(mns[ms.index(motion)])
        for i in range(0,len(motion),7):
            printMotion(motion[i])
            m = list(np.mean(motion[i][j]) for j in fs)
            m /= max(m)
            m = list("{0:.2f}".format(x) for x in m)
            for i in range(3,len(m)+2,4): m.insert(i,'|')
            print(' '.join(str(x) for x in m))
        

    return ms

#ms = tmp1()


def tmp2():
    m = getAcceMotion()
    ffts, [w2v,w1v,pv,jjv,tv,bv] = m #,sdv,suv,jv] = m
    fs = [0,1,2,3,4,5,12,13,14,15,16,17,6,7,8]
    mns = ['Wave two hands','Throwing']
    ms = [w2v,tv]
    
    for i in range(10):
        pt.figure()
        for motions in ms:
            ind = ms.index(motions)
            print(mns[ind])
            motion = motions[1][0]
            fig = pt.subplot(len(ms),2,ind*2+1)
            pt.plot(motion)
            fig.set_title(mns[ind])
            f = np.abs(np.fft.fft(motion))[1:10]
            pt.subplot(len(ms),2,ind*2+2)
            pt.bar(np.arange(len(f)),f)
        pt.show()
        
            #for i in range(0,len(motion),7):
            #    print('length: ',len(motion[i][0]))
            #    printMotion(motion[i])
    return ms

ms = tmp2()



def c3():
    
    m = getAcceMotion()
    ffts, [w2v,w1v,pv,jjv,tv,ssv,bv] = m #,sdv,suv,jv] = m
    fs = [0,1,2,3,4,5,12,13,14,15,16,17,6,7,8]
    
    ms = [w2v,bv]
    mns = ['Wave two hands','Bending']
    
    tot_inds,tot_ratios,tot_var,tot_snrs = [],[],[],[]
    for motion in ms:
        print(mns[ms.index(motion)])
        inds_list,ratios_list,var_list,snrs_list = [],[],[],[]
        for m in range(0,len(motion),10):
            inds,ratios,vrs,snrs = [],[],[],[]
            for i in fs:
                f = np.real(np.fft.fft(motion[m][i]))[1:35]
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
            print(' '.join(str(round(x,3)) for x in ratios))
    for var_list in tot_var:
        print('New motion')
        for vrs in var_list:
            print(' '.join(str(round(x,3)) for x in vrs))
    for snrs_list in tot_snrs:
        print('New motion')
        for snrs in snrs_list:
            for i in range(3,len(snrs)+2,4): snrs.insert(i,'|')
            print(' '.join(str(x) for x in snrs))
    return ms

#ms = c3()


def m():
    fs = list(range(12)) # + [15,16,17]
    motions = tmp1()
    l = ['larm','rarm','lfoot','rfoot','hip']
    n = [' wave2hands ',' wave1hand ','  punching  ',' jumpjacks  ', 
         '  clapping  ','  throwing  ']
    for j in range(len(motions)):
        print('*******************' + n[j] + '***********************')
        mag,mag1 = [],[]
        for i in fs: mag.append(max(motions[j][i])-min(motions[j][i]))
        mag = list(round(x/max(mag),2) for x in mag)
        for i in range(3,len(mag)+1,4): mag.insert(i,'|')  
        #mag1 = list(round(max(mag[i:i+3]),2) for i in range(0,len(mag),3))
        mag = ' '.join(str(x) for x in mag)
        #mag1 = ' | '.join(str(x) for x in mag1)
        print(mag)
    return motions

#ms = m()

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

def m1():
    m = getAcceMotion()
    [w2f,w1f,pf,jjf,tf], _,_ = m  
    fs = list(range(12))
    print('\nw2f')
    for i in range(0,30,5):
        r = list(round(getFP(w2f[i][j])[0],2) for j in fs)
        r1 = list(getFP(w2f[i][j])[1] for j in fs)
        print(' '.join(str(x) for x in r),' ',r1)
    print('\nw1f')
    for i in range(0,30,5): 
        r = list(round(getFP(w1f[i][j])[0],2) for j in fs)
        r1 = list(getFP(w1f[i][j])[1] for j in fs)
        print(' '.join(str(x) for x in r),' ',r1)   
    print('\npf')
    for i in range(0,30,5): 
        r = list(round(getFP(pf[i][j])[0],2) for j in fs)
        r1 = list(getFP(pf[i][j])[1] for j in fs)
        print(' '.join(str(x) for x in r),' ',r1)  
    print('\njjf')
    for i in range(0,30,5): 
        r = list(round(getFP(jjf[i][j])[0],2) for j in fs)
        r1 = list(getFP(jjf[i][j])[1] for j in fs)
        print(' '.join(str(x) for x in r),' ',r1)
    print('\ntf')
    for i in range(0,30,5): 
        r = list(round(getFP(tf[i][j])[0],2) for j in fs)
        r1 = list(getFP(tf[i][j])[1] for j in fs)
        print(' '.join(str(x) for x in r),' ',r1)    
    
    return None

#m1()

def c():
    _,[w2v,w1v,pv,jjv,cv,tv] = tmp1()
    print('Punching')
    for i in range(0,len(pv),10):
        tmp = list(np.corrcoef(pv[i][j],pv[i][j+3])[1,0] for j in range(3))
        #tmp = list(np.corrcoef(pv[i][j],pv[i][k])[1,0] for j in range(3,6) for k in range(j+1,6))
        print(tmp)
    print('\nJumping jacks')
    for i in range(0,len(jjv),10):
        tmp = list(np.corrcoef(jjv[i][j],jjv[i][j+3])[1,0] for j in range(3))
        #tmp = list(np.corrcoef(jjv[i][j],jjv[i][k])[1,0] for j in range(3,6) for k in range(j+1,6))
        print(tmp)
    print('\nWave two hands')
    for i in range(0,len(w2v),10):
        tmp = list(np.corrcoef(w2v[i][j],w2v[i][j+3])[1,0] for j in range(3))
        #tmp = list(np.corrcoef(jjv[i][j],jjv[i][k])[1,0] for j in range(3,6) for k in range(j+1,6))
        print(tmp)        
    
    return

#c()



def c1():
    fs = list(range(12))
    _,[w2v,w1v,pv,jjv,cv,tv] = tmp1()
    
    print('Punching')
    for m in range(0,len(pv),9):
        l = list(np.ptp(pv[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])), 
               l[9:12].index(max(l[9:12])),l.index(max(l))]
        print(tmp)
        
    print('Jumping jacks')
    for m in range(0,len(jjv),9):
        l = list(np.ptp(jjv[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])),
               l[9:12].index(max(l[9:12])),l.index(max(l))] 
        print(tmp)
        
    print('Wave two hands')
    for m in range(0,len(w2v),9):
        l = list(np.ptp(w2v[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])),
               l[9:12].index(max(l[9:12])),l.index(max(l))] 
        print(tmp)
        
    print('Wave one hand')
    for m in range(0,len(w1v),9):
        l = list(np.ptp(w1v[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])),
               l[9:12].index(max(l[9:12])),l.index(max(l))] 
        print(tmp) 
        
    print('Clapping')
    for m in range(0,len(cv),9):
        l = list(np.ptp(cv[m][fs[i]]) for i in range(len(fs)))
        tmp = [l[:3].index(max(l[:3])),l[3:6].index(max(l[3:6])), l[6:9].index(max(l[6:9])),
               l[9:12].index(max(l[9:12])),l.index(max(l))] 
        print(tmp)     
                
    return

#c1()



def tmp1():

    m = getAcceMotion()
    ffts, ms = m #,sdv,suv,jv] = m
    #for i in range(0,50,5):
    #    printMotion(cv[i])
    #for i in range(0,50,5): 
    #    printMotion(w2v[i])
    fs = [0,1,2,3,4,5,12,13,14,15,16,17,6,7,8]

    mns = ["Wave two hands","Wave one hand","Punching","Jumping Jacks",\
           "Throwing","Bending"]
    #mns = ['Throwing','Sitdown and standup','Jumping jacks','Wave two hands',\
    #       'Wave one hand','Punching','Bending','Sit down','Stand up','Jumping']
    #ms = [tv,ssv,jjv,w2v,w1v,pv,bv,sdv,suv,jv]

    for motion in ms:
        print(mns[ms.index(motion)])
        for i in range(0,len(motion),7):
            printMotion(motion[i])
            m = list(np.mean(motion[i][j]) for j in fs)
            m /= max(m)
            m = list("{0:.2f}".format(x) for x in m)
            for i in range(3,len(m)+2,4): m.insert(i,'|')
            print(' '.join(str(x) for x in m))
        

    return ms


def test():


    ffts,motion_diff = getAcceMotion()
    # BKL --> BKL Acce
    # left arm/wrist, right arm/wrist, left foot/ankles, 
    # right foot/ankles, left up leg/hips, right up leg/hips
    names = ['Wave two hands','Wave one hand','Boxing', 'Jumping jacks',\
               'Throwing','Bending']    
    feature_names = ['left fore arm/wrist', 'right fore arm/wrist', 'left foot/ankles', 
     'right foot/ankles', 'left up leg/hips']
    joint = [[0,1,2],[3,4,5],[12,13,14],[15,16,17],[6,7,8]]
    # ********** BKL MoCap **********
    
    ind1,ind2,ind3,ind4 = 0,1,2,3
    for motion_type in motion_diff[:]:
        name = names[motion_diff.index(motion_type)]
        print(name)  
        count = 0
        for i in np.arange(0,40,7):
            pt.figure(figsize=(12,8))
            pt.tight_layout(pad=0.7, w_pad=0.7, h_pad=0.7)
            pt.suptitle(names[motion_diff.index(motion_type)])
            motion = motion_type[i] # 0 Walking 1 Jumping
            count += 1
            for q in range(len(joint[ind1])): # for each fine grained feature
                pt.subplot(4,3,q+1)
                pt.plot(motion[joint[ind1][q]])
                
            for q in range(len(joint[ind2])): # for each fine grained feature
                pt.subplot(4,3,q+4)
                pt.plot(motion[joint[ind2][q]])
                
            for q in range(len(joint[ind3])): # for each fine grained feature
                pt.subplot(4,3,q+7)
                pt.plot(motion[joint[ind3][q]])
            for q in range(len(joint[ind4])): # for each fine grained feature
                pt.subplot(4,3,q+10)
                pt.plot(motion[joint[ind4][q]])
            #pt.tight_layout(w_pad=0.1, h_pad=0.1)
            st = '/Users/xiahu/Documents/Projects/Transfer/Expe/ACCE_figs/' + name+'_' + str(count) + '.png'
            pt.savefig(st,dpi=300)            
            #pt.show()
        
    
#test()


def tmp2():
    print(len(jv))
    for i in range(0,30,3):
        printMotion(jjv[i])
        
    print(len(jjv))
    for i in range(0,30,3):
        printMotion(bv[i])
    print(len(bv))
    for i in range(0,30,3):
        printMotion(tv[i])    
    print(len(tv))
    for i in range(0,30,3):
        printMotion(ssv[i]) 
    print(len(ssv))
    for i in range(0,30,3):
        printMotion(sdv[i]) 
    print(len(sdv))
    for i in range(0,30,3):
        printMotion(suv[i]) 
    print(len(suv))
        
    #t2 =  printMotion(cf[3])
    #tmp1()
    #tmp = printMotion(bf[0])
   # tmp1 = printMotion(bp[0])
   # tmp2 = printMotion(bf[3])
   # tmp3 = printMotion(bp[3])
        



def testPCA():
    # left arm: left shoulder, forearm, arm
    larm = [[0,1,2]]
    # right arm: right shoulder, forearm, arm
    rarm = [[3,4,5]]
    # left leg: left up leg, leg, foot
    lleg = [[12,13,14]]
    # right leg: right up leg, leg, foot
    rleg = [[15,16,17]]
    # body: head, neck, spine, spine1, spine2, hips(pos), hip
    body = [[6,7,8]]
    joint = [larm,rarm,lleg,rleg,body]
    names = ['wave2hands','wave1hand','boxing','jumpingjacks','throwing','bending']
    
    source_fft, motion_diff = getAcceMotion()
    
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
    
    def getPCA2(features1, features2, motion):
        m = np.hstack([motion[features1],motion[features2]])
        m = np.transpose(m)
        
        pca = PCA(n_components=len(features1))
        bp = pca.fit_transform(m)
        c = pca.components_
        bp = np.transpose(bp)
        
        return bp,c,pca    
    
    # test a pair of joint
    for motion_type in np.array(motion_diff)[[0,2,3,5]]:
        corr_list.append([])
        for motion in motion_type[::7]:
            print(names[motion_diff.index(motion_type)])
            #pca1,c1 = getPCA(larm[0],motion)
            #pca2,c2 = getPCA(rarm[0],motion)
            

            #ax1.plot(motion[larm[0][0]],motion[larm[0][1]],motion[larm[0][2]])
            #ax1.plot(motion[rarm[0][0]],motion[rarm[0][1]],motion[rarm[0][2]])
            
            mat1,mat2 = motion[larm[0]],motion[rarm[0]] # pca1, pca2 
            
            #pca_rgs1 = [np.ptp(mat1[0]),np.ptp(mat1[1]),np.ptp(mat1[2])]
            #pca_rgs2 = [np.ptp(pca2[0]),np.ptp(pca2[1]),np.ptp(pca2[2])]
            #pca_rgs = np.array(pca_rgs1) + np.array(pca_rgs2)
            #pca_rgs = list(x / pca_rgs.max() for x in pca_rgs)
            
            #corr1 = np.corrcoef(c1[0],c2[0])[1,0]
            #corr2 = np.corrcoef(c1[1],c2[1])[1,0]
            #corr3 = np.corrcoef(c1[2],c2[2])[1,0]
            #pca_corr1 = np.corrcoef(pca1[0],pca2[0])[1,0]
            #print('axis corr', corr1, 'pca corr', pca_corr1, 'corr', corr1 * pca_corr1)
            
            #print('CORR',corr1,corr2,corr3)
            
            from mpl_toolkits.mplot3d import Axes3D
            import pickle
  
            fig = pt.figure(figsize = (15,5))    
            ax1 = fig.add_subplot(1,5,1, projection='3d')
            ax1.set_title(names[motion_diff.index(motion_type)])
            ax1.plot(mat1[0],mat1[1],mat1[2])
            ax1.plot(mat2[0],mat2[1],mat2[2])
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            
            
            ax2 = fig.add_subplot(1,5,2)
            ax2.set_title('x acceleration')
            ax2.plot(np.arange(len(mat1[0])),mat1[0])
            ax2.plot(np.arange(len(mat2[0])),mat2[0])
            
            ax3 = fig.add_subplot(1,5,3)
            ax3.set_title('y acceleration')
            ax3.plot(np.arange(len(mat1[1])),mat1[1])
            ax3.plot(np.arange(len(mat2[1])),mat2[1])    
            
            ax4 = fig.add_subplot(1,5,4)
            ax4.set_title('z acceleration')
            ax4.plot(np.arange(len(mat1[1])),mat1[2])
            ax4.plot(np.arange(len(mat2[1])),mat2[2])                
            
            
            _,c1,pca1 = getPCA(larm[0],motion)
            #new_mat2_tmp,c2,pca2 = getPCA(rarm[0],motion)
            #print('c1',c1,'c2',c2)
            new_mat1 = pca1.transform(np.transpose(motion[larm[0]]))
            new_mat2 = pca1.transform(np.transpose(motion[rarm[0]]))
            new_mat1 = np.transpose(new_mat1)
            new_mat2 = np.transpose(new_mat2)
            ax5 = fig.add_subplot(1,5,5)
            nmfft1 = np.fft.fft(new_mat1[0])[2:]
            #print('nmfft1',nmfft1)
            nmfft1_tmp = np.abs(nmfft1)
            nmfft1 = np.hstack([np.real(nmfft1),np.imag(nmfft1)])
            
            nmfft2 = np.fft.fft(new_mat2[0])[2:]
            nmfft2_tmp = np.imag(nmfft2)
            #print('nmfft2',nmfft2)
            nmfft2 = np.hstack([np.real(nmfft2),np.imag(nmfft2)])
            
            
            ax5.set_title(str(np.corrcoef(nmfft1,nmfft2)[1,0]))
            #ax5.bar(np.arange(30),nmfft1_tmp[:30])
            ax5.plot(np.arange(len(new_mat1[0])),new_mat1[0])
            ax5.plot(np.arange(len(new_mat2[0])),new_mat2[0])
            
        
            
            '''
                
            ax1 = fig.add_subplot(1,4,1, projection='3d')
            ax1.set_title(names[motion_diff.index(motion_type)])
            ax1.plot(mat1[0],mat1[1],mat1[2])
            ax1.plot(mat2[0],mat2[1],mat2[2])
            
            ax2 = fig.add_subplot(1,4,2)
            ax2.set_title('features 0 & 1')
            ax2.plot(mat1[0],mat1[1])
            ax2.plot(mat2[0],mat2[1])
            
            ax3 = fig.add_subplot(1,4,3)
            ax3.set_title('features 0 & 2')
            ax3.plot(mat1[0],mat1[2])
            ax3.plot(mat2[0],mat2[2])
            
            ax4 = fig.add_subplot(1,4,4)
            ax4.set_title('features 1 & 2')
            ax4.plot(mat1[1],mat1[2])
            ax4.plot(mat2[1],mat2[2])
            '''
            
            
            #corr_list[-1].append(corr1)
            #print('CORR',corr1,corr2,corr3)
            pt.show()

            
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
    
    pt.show()
    '''
    return

#testPCA()

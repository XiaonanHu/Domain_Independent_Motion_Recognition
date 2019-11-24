import random, itertools, warnings
import numpy as np
from sklearn.mixture import GMM
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


from readKIT import getKITMotion
from readCMU import getCMUMotion
from readBKL import getBKLMotion
from readMSR import getMSRMotion
from readSMS import getSMSMotion
from readBKLAcce import getAcceMotion
from readMocapData import getMocapDataMotion

from sgfilter import savitzky_golay

    
                    
warnings.filterwarnings("ignore")

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    


def get_feature_indices(test_name):
    if test_name == "CMU":
        '''
        CMU    <<- zero-shot transfer ->>    KIT
        upperneck, right humerus(right elbow), left humerus(left elbow), right wrist,
        left wrist, right femur(right knee), left femur(left knee), 
        right foot(right ankle), left foot(left ankle), lowerback(pelvis)   
        '''
 
        source_f_indices = [[18,19,20],[24,25,26],[34,35,36],[28],[38],[44,45,46],\
                            [51,52,53],[47,48,49],[54,55,56],[6,7,8]]
        
        target_f_indices = [[6,7,8],[26,27],[12,13],[35,36],[21,22],[31],[17],\
                            [23,24,25],[9,10,11],[3,4,5]]
        
        # right humerus, left humerus, right femur, left femur, lowerback
        source_f_indices = [[24,25,26],[34,35,36],[44,45,46],\
                            [51,52,53],[6,7,8]]

        # right elbow, left elbow, right foot, left foot, pelvis
        target_f_indices = [[26,27],[12,13],       
                            [28,29,30],[14,15,16],[3,4,5]]
                            
             
        feature_names = ['upperneck', 'right humerus(right elbow)',\
                         'left humerus(left elbow)', 'right wrist', 'left wrist', \
                         'right femur(right knee)', 'left femur(left knee)',\
                         'right foot(right ankle)', 'left foot(left ankle)',\
                         'lowerback(pelvis)']
        
    elif test_name == "CMU->SMS":
        '''
        CMU    <<- zero-shot transfer ->>    SMS
        thorax (chest), head, left humerus (left up arm), right humerus (right
        up arm), left
        hand, right hand, left femur (left up leg), right femur (right up leg),
        left tibia (left low leg), right tibia (right low leg), left foot, 
        right foot, left radius (left low arm), right radius (right low arm),
        lower (neck), lowerback (hip)
        '''
        
        source_f_indices = [[12,13,14],[21,22,23],[34,35,36],[24,25,26],\
                            [39,40],[29,30],[51,52,53],[44,45,46],[54],[47],[55,56],\
                            [48,49],[37],[27],[15,16,17], [6,7,8]]
        target_f_indices = [[6,7,8],[12,13,14],[18,19,20],[30,31,32],\
                            [24,25,26],[36,37,38],[39,40,41],[48,49,50],\
                            [42,43,44],[51,52,53],[45,46,47],[54,55,56],[21,22,23],\
                            [33,34,35],[9,10,11],[3,4,5]]  
        
        # right humerus, left humerus, right femur, left femur, lowerback
        source_f_indices = [[24,25,26],[34,35,36],[44,45,46],\
                            [51,52,53],[6,7,8]]
        target_f_indices = [[30,31,32],[18,19,20],[48,49,50],[39,40,41],[3,4,5]]
        target_f_indices = [[33,34,35],[21,22,23],[51,52,53],[42,43,44],[0,1,2]]
        #target_f_indices = [[30,31,32,33,34,35],[18,19,20,21,22,23],[48,49,50,51,52,53],[39,40,41,42,43,44],[0,1,2,3,4,5]]
        feature_names = ['thorax','head','left humerus', 'right humerus',\
                         'left hand', 'right hand', 'left femur',\
                         'right femur', 'left tibia', 'right tibia',  'left foot',\
                         'right foot', 'left radius', 'right radius', 'lowerneck',\
                         'lowerback']
        
        
    elif test_name == "KIT->SMS":
        '''
        KIT    <<- zero-shot transfer ->>    SMS
        thorax (chest), head, left humerus (left up arm), right humerus (right
        up arm), left
        hand, right hand, left femur (left up leg), right femur (right up leg),
        left tibia (left low leg), right tibia (right low leg), left foot, 
        right foot, left radius (left low arm), right radius (right low arm),
        lower (neck), lowerback (hip)
        right humerus, left humerus, right femur, left femur, lowerback
        '''
        
        # KIT
        # right elbow, left elbow, right foot, left foot, pelvis
        source_f_indices = [[26,27], [12,13], [28,29,30], [14,15,16], [3,4,5]]
        # SMS
        target_f_indices = [[33,34,35], [21,22,23], [51,52,53], [42,43,44], [0,1,2]]
        feature_names = ['thorax','head','left humerus', 'right humerus',\
                         'left hand', 'right hand', 'left femur',\
                         'right femur', 'left tibia', 'right tibia',  'left foot',\
                         'right foot', 'left radius', 'right radius', 'lowerneck',\
                         'lowerback']
        
    elif test_name == 'CMU->Mocap':
        '''
        CMU    <<- zero-shot transfer ->>    MocapData
        thorax (chest), head, lowerneck (neck), left humerus (shoulder), right
        humerus (shoulder), left femur (hip), right femur (hip), left radius 
        (elbow), right radius (elbow), left wrist, right wrist, left foot (ankle)
        right foot (ankle), left tibia (knee), right tibia (knee)
        '''

        source_f_indices = [[12,13,14],[21,22,23],[15,16,17],[34,35,36],[24,25,26],\
                            [51,52,53],[44,45,46],[37],[27],[38],[28],[55,56],\
                            [48,49],[54],[47]]
        target_f_indices = [[21,22,23],[54,55,56],[51,52,53],[30,31,32],[43,44,45],\
                            [3,4,5],[12,13,14],[33,34,35],[45,46,47],[36,37,38],[48,49,50],\
                            [9,10,11],[18,19,20],[6,7,8],[15,16,17]]
        feature_names = ['thorax', 'head', 'lowerneck', 'left humerus', 'right humerus',\
                         'left femur', 'right femur', 'left radius', 'right radius',\
                         'left wrist', 'right wrist', 'left foot', 'right foot',\
                         'left tibia', 'right tibia']
        
    elif test_name == 'BKL->Acce':
        '''
        BKL    <<- zero-shot transfer ->>    BKL Acce
        left arm/wrist, right arm/wrist, left foot/ankles, 
        right foot/ankles, left up leg/hips, right up leg/hips
        '''

        feature_names = ['left fore arm/wrist', 'right fore arm/wrist', 'left foot/ankles', 
         'right foot/ankles', 'left up leg/hips']
        source_f_indices = [[34,35,36],[24,25,26],[61,62,63],[49,50,51],[3,4,5]]
        target_f_indices = [[0,1,2],[3,4,5],[12,13,14],[15,16,17],[6,7,8]]

    return source_f_indices, target_f_indices, feature_names
    
    
class Node:
    def __init__(self,feature):
        self.feature = feature
        self.adjacents = {}
        self.edge_vals = []
    def connect(self,node,val):
        self.adjacents[node] = val
        self.edge_vals.append(val)
    def degree(self):
        return len(self.edge_vals)
        
class Edge:
    def __init__(self,fpr,val):   
        self.node1 = Node(fpr[0])
        self.node2 = Node(fpr[1])
        self.node1.connect(self.node2,val)
        self.node2.connect(self.node1,val)

class MotionStructure:
    def __init__(self,fpr,val):
        e = Edge(fpr,val)
        self.head = e.node1
        self.nodes = [e.node1,e.node2]
        self.features = [fpr[0],fpr[1]]
        self.edges = [fpr]
        self.edge_vals = [val]
        self.node_count = 2
        self.edge_count = 1
        self.l1 = None
        self.l2 = None
        self.l3 = None
        
    def find_node(self, fpr_node):
        for node in self.nodes:
            if fpr_node == node.feature:
                return node
        return None
            
    def add_fpr(self, fpr, val):
        node1 = self.find_node(fpr[0])
        node2 = self.find_node(fpr[1])
        if node1 == None: 
            node1 = Node(fpr[0])
            self.node_count += 1
            self.nodes.append(node1)
            self.features.append(fpr[0])
        if node2 == None: 
            node2 = Node(fpr[1])
            self.node_count += 1
            self.nodes.append(node2)
            self.features.append(fpr[1])

        node1.connect(node2,val)
        node2.connect(node1,val)
        self.edge_count += 1
        self.edges.append(fpr)
        self.edge_vals.append(val)
        
    def get_features(self):
        return self.features

    def get_fpr_vals(self):
        return self.edge_vals

    def sort_node_by_edges(self):
        l = sorted(self.nodes, key=lambda x: x.degree(),reverse=True)
        return l


def gmm(arg):
    X = arg[0]
    n_features = arg[1]
    
    lower = int(n_features/3)
    upper = int(n_features/2)  
    mat = np.zeros((n_features,n_features), dtype=np.int32)
    
    for j in range(lower,upper):
        GM = GMM(n_components = j,covariance_type='full')
        GM.fit(X)
        l = GM.predict(X)
        for c in range(j):
            ind = np.where(l == c)[0]
            for i in range(len(ind)):
                for k in range(i,len(ind)):
                    mat[ind[i]][ind[k]] += 1    
    return mat 



def get_motion_corr(fft):
    n = fft[0][0].shape[0]
    total_mats = []
    for i in range(len(fft)): # for each motion type
        mats_list = []
        for j in range(len(fft[i])): # for each motion sample
            mats = np.zeros((n,n),dtype=np.float64)
            for p in range(n):
                for q in range(p+1,n):
                    corr = np.corrcoef(fft[i][j][p],fft[i][j][q])[1,0]
                    if np.isnan(corr): mats[[p,q],[q,p]] = 0
                    else: mats[[p,q],[q,p]] = corr
            mats_list.append(mats)                
        total_mats.append(mats_list)
    return total_mats

    

def motion_recognition(X, Y, X_test, num, num_classes):
   
    clfs = [RandomForestClassifier(max_depth=10, n_estimators=66, max_features=5)]
    motion_indices = [[] for _ in range(num_classes)]
    
    for clf in clfs: # each classifiers choose a few confident samples
        for k in range(5):
            tot_prob = []
            clf.fit(X, Y)
            prob = clf.predict_proba(X_test)
            if not len(tot_prob): tot_prob = prob
            else: tot_prob += prob
            for i in range(num_classes):
                l = list(tot_prob[:,i])
                l_sort = sorted(l, reverse=True)
                inds = list(np.where(l >= l_sort[num])[0])
                motion_indices[i] += inds

    for i in range(num_classes): # congregate the votes and pick top "num" motions
        l = list((x, motion_indices[i].count(x)) for x in set(motion_indices[i]))
        l = sorted(l, key = lambda x:x[1], reverse=True)
        l = list(x[0] for x in l)
        
        motion_indices[i] = l[:num]
    
    print('motion_indices', motion_indices)
    
    return motion_indices


def test_cross_val(X, Y):
    
    names = ["Nearest Neighbors", "Linear SVM", "Linear SVC", "RBF SVM", \
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes"] # "Gaussian Process",
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=1,decision_function_shape='ovr'),
        LinearSVC(loss='squared_hinge',penalty='l2',tol=0.01), # tol=0.00001  
        SVC(gamma=0.02, C=1,decision_function_shape='ovr',tol=0.001), # rbf
        #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=10, max_features=3), # previously 5
        RandomForestClassifier(max_depth=10, n_estimators=66, max_features=5), # previously 5, n_estimators=33, max_features=1
        MLPClassifier(alpha=1,max_iter=300),
        AdaBoostClassifier(),
        GaussianNB()]
        #QuadraticDiscriminantAnalysis()]
    
    dec_list = []
    score_list = []
    for name, clf in zip(names, classifiers):
        score = np.mean( cross_val_score(clf,X,Y,cv=5) )

        print(name,'   ',score)
        target_names = ['Walk','Run','Jump']
        target_names = ['Wave two hands', 'Wave one hand','Punching',\
                        'Jumping jacks','Throwing','Bending','Sitdown Standup']    
        
        score_list.append(score)
    ind = score_list.index(max(score_list))
    return np.array(score_list), names

def test(X, Y, X_test, Y_test, single_domain):
    
    names = ["Nearest Neighbors", "Linear SVM", "Linear SVC", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=1, decision_function_shape='ovr'),
        LinearSVC(loss='squared_hinge',penalty='l2',tol=0.01), # tol=0.00001  
        SVC(gamma=0.02, C=1,decision_function_shape='ovr',tol=0.001), # rbf
        #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=10, max_features=3), # previously 5
        RandomForestClassifier(max_depth=10, n_estimators=66, max_features=5), # previously 5, n_estimators=33, max_features=1
        MLPClassifier(alpha=1,max_iter=300),
        AdaBoostClassifier(),
        GaussianNB()]
    

    score_list = []
    for name, clf in zip(names, classifiers):
        if single_domain:
            score = np.mean(cross_val_score(clf,X_test,Y_test,cv=5))
        else:
            clf.fit(X, Y)
            score = clf.score(X_test,Y_test)
        
        print(name,'   ',score)
        target_names = ['Wave two hands', 'Wave one hand','Punching',\
                        'Jumping jacks','Throwing','Bending','Sitdown Standup']        
            
        score_list.append(score)
    return np.array(score_list), names


def test_transfer(X, Y, X_test, Y_test, op):
    
    names = ["Nearest Neighbors", "Linear SVM", "Linear SVC", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes"] # "Gaussian Process",
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=1, decision_function_shape='ovr'),
        LinearSVC(loss='squared_hinge', penalty='l2', tol=0.01), # tol=0.00001  
        SVC(gamma=0.02, C=1,decision_function_shape='ovr', tol=0.001), # rbf
        DecisionTreeClassifier(max_depth=10, max_features=5), # previously 5
        RandomForestClassifier(max_depth=10, n_estimators=66, max_features=5), # previously 5, n_estimators=33, max_features=1
        MLPClassifier(alpha=1, max_iter=300),
        AdaBoostClassifier(),
        GaussianNB()]
    
    dec_list = []
    score_list = []
    for name, clf in zip(names, classifiers):

        clf.fit(X, Y)
        score = clf.score(X_test,Y_test)
        print(name,'   ',score)
        y_true = Y_test
        y_pred = clf.predict(X_test)
        if op == 0: target_names = ['Walk','Run','Jump']
        elif op == 1: target_names = ['Wave two hands', 'Wave one hand','Punching',\
                        'Jumping jacks','Throwing','Bending','Sitdown Standup']  
        elif op == 2: target_names = ['Walk','Run','Jump']
        elif op == 3: target_names = ['Walk','Run','Jump']
        if False and name == 'Linear SVM': 
            print(clf.support_)
            print('[test_transfer] Support vectors: ', clf.support_vectors_)
        if name == 'Linear SVC' or name == 'Random Forest':#'Random Forest' or name == 'AdaBoost':
            print('[test_transfer] The prediction is', y_pred)
            print(classification_report(y_true, y_pred, target_names=target_names))
            
        score_list.append(score)
    return np.array(score_list), names


# input is 
def get_data(motions):
    motion_list = []
    Y = []
    for i in range(len(motions)): # for each motion type
        for j in range(len(motions[i])):
            motion_list.append(motions[i][j])
        Y += [i]*len(motions[i])
    return motion_list,Y


def get_Y(motions):
    Y = []
    for i in range(len(motions)): # for each motion type
        Y += [i]*len(motions[i])
    return Y


# with shifts of fpr values #(X1,Y1,X1_corr,class_nums1)  
def partition_data(X, Y, X_corr, class_lens, percentage):
    class_num = len(class_lens)
    
    indices = list(random.sample(range(class_lens[i]),
                    int(class_lens[i]*percentage)) for i in range(class_num))
    class_lens = np.cumsum(class_lens)
    idx = indices[0]
    for i in range(1,class_num):
        idx +=  list(x + class_lens[i-1] for x in indices[i])
    
    s = X_corr[0].shape[0] * X_corr[0].shape[1]
    X_train = list(X[i,:] for i in idx)
    X_train = np.vstack(X_train)
    X_corr_train = list(X_corr[i].reshape(1,s) for i in idx)
    X_corr_train = np.vstack(X_corr_train)
    Y_train = list(Y[i] for i in idx)
    X_test = list(X[i,:] for i in range(X.shape[0]) if i not in idx)
    X_test = np.vstack(X_test)
    X_corr_test = list(X_corr[i].reshape(1,s) for i in range(X.shape[0]) if i not in idx)
    X_corr_test = np.vstack(X_corr_test)
    Y_test = list(Y[i] for i in range(X.shape[0]) if i not in idx)
    print(X_train.shape, len(Y_train), X_test.shape, len(Y_test))
    
    return X_train, Y_test, X_corr_train, X_corr_test


    

def get_PCA(features, motion):
    if len(features) == 1: 
        return np.squeeze(motion[features[0]])
    pca = PCA(n_components=1)
    m = np.transpose(motion[features])
    body_part = pca.fit_transform(m)

    return np.squeeze(body_part[:,0])



def get_significant_frequency(body_part):
    f = np.abs(np.fft.fft(body_part))[1:31]
    f_sort = sorted(f, reverse=True)

    if list(f).index(f_sort[0]) == 0:
        if f_sort[0]*0.6 > f_sort[1]: return 0 # originally 0.6
        else: return list(f).index(f_sort[1])

    else:
        threshold = f_sort[0]*0.7
        freqs = list(1 if x > threshold else 0 for x in f)
        inds = np.where(np.array(freqs == 1)) [0]
        # trying to find harmonic frequencies

        for i in range(len(inds)):
            for j in range(i+1,len(inds)):
                if inds[i] > 1:
                    if int(inds[j]/2) in range(inds[i]-1, inds[i]+1):
                        return inds[i]
        return list(f).index(f_sort[0])



def get_PCA2(features, motion):
    if len(features) == 1: return motion[features[0]]
    pca = PCA(n_components=len(features))
    m = np.transpose(motion[features])
    body_part = pca.fit_transform(m)

    return np.transpose(body_part)



def get_best_joint(features,motion):
    cands = np.zeros((1,len(features)),dtype=np.float64)
    p_list = list(get_PCA2(features[i],motion) for i in range(len(features)))
    rgs_p = list(list(np.ptp(p_list[i][x]) for x in range(len(features[i]))) for i in range(len(p_list)))
    rgs_p = list(max(x) for x in rgs_p)
    cands += np.array(rgs_p)
    
    lens = list(len(x) for x in features)
    lens = list(x/max(lens) for x in lens)
    cands += lens
    rgs = list(list(np.std(motion[x]) for x in features[i]) for i in range(len(features)))
    rgs_sum = list(sum(x) for x in rgs)
    rgs_max = list(max(x) for x in rgs)
    rgs1 = list(x/max(rgs_sum) for x in rgs_sum)
    rgs2 = list(x/max(rgs_max) for x in rgs_max)

    rgs2 = list(x for x in rgs2)
    cands = np.squeeze(cands)
    return list(cands).index(cands.max())



def choose_joint(i, features, motion, type_index):
    feat = features[i]
    ind, f_t = None, None
    
    if features[0] == [26,27]: # KIT in CMU<->KIT
        if i == 2:
            f_t = [[28,29,30], [31], [23,24,25]]
        elif i == 3:
            f_t = [[14,15,16], [17], [9,10,11]]
        elif i == 4:
            f_t = [[3,4,5], [6,7,8], [0,1,2]]
        elif i == 0:
            f_t = [[26,27], [32,33,34], [35,36]]
        elif i == 1:
            f_t = [[12,13], [18,19,20], [21,22]]
        ind = get_best_joint(f_t,motion)
        
        feat = f_t[ind]
        
    if False and features[0] == [34,35,36]: # BKL Mocap<->ACCE
        if i == 0:
            f_t = [[31,32,33], [37,38,39], [34,35,36]]
        if i == 1:
            f_t = [[21,22,23], [27,28,29], [24,25,26]]
        if i == 2:
            f_t = [[53,54,55], [57,58,59], [61,62,63]]
        if i == 3:
            f_t = [[41,42,43], [45,46,47], [49,50,51]]
        if i == 4:
            f_t = [[0,1,2], [3,4,5]]
        ind = get_best_joint(f_t, motion)
        if i == 0: ind = 0
        if i == 1: ind = 0
        feat = f_t[ind]
        
    if False and features[0] == [33,34,35]: # CMU<->SMS
        if i == 0:
            f_t = [[30,31,32], [33,34,35]]
        elif i == 1:
            f_t = [[18,19,20], [21,22,23]]
        elif i == 2:
            f_t = [[48,49,50], [51,52,53]]
        elif i == 3:
            f_t = [[39,40,41], [42,43,44]]
        elif i == 4:
            f_t = [[0,1,2], [3,4,5]]
        ind= get_best_joint(f_t, motion)
        feat = f_t[ind]

    return feat, ind, f_t



def get_directional_correlations(body_part_fft_list):
    arm1, arm2, leg1, leg2 = 0, 1, 2, 3
    
    m0 = len(body_part_fft_list[arm1]) 
    m1 = len(body_part_fft_list[arm2])
    m2 = len(body_part_fft_list[leg1])
    m3 = len(body_part_fft_list[leg2]) 

    body_part_direc1 = list(np.corrcoef(body_part_fft_list[arm1][i],body_part_fft_list[arm2][j])[1,0] for i in range(m0) for j in range(m1))
    body_part_direc2 = list(np.corrcoef(body_part_fft_list[leg1][i],body_part_fft_list[leg2][j])[1,0] for i in range(m2) for j in range(m3))
    body_part_direc3 = list(np.corrcoef(body_part_fft_list[arm1][i],body_part_fft_list[leg1][j])[1,0] for i in range(m0) for j in range(m2)) 


    body_part_direc1_rg = list(np.ptp(body_part_fft_list[arm1][i]) for i in range(m0))
    body_part_direc1_rg2 = list(np.ptp(body_part_fft_list[arm2][i]) for i in range(m1))  
     
    body_part_direc1_rg = list((body_part_direc1_rg[i]*body_part_direc1_rg2[j]) for i in range(m0) for j in range(m1))
    body_part_direc1_rg = body_part_direc1_rg/max(body_part_direc1_rg)     
    
    body_part_direc2_rg = list(np.ptp(body_part_fft_list[leg1][i]) for i in range(m2))
    body_part_direc2_rg2 = list(np.ptp(body_part_fft_list[leg2][i]) for i in range(m3))

    body_part_direc2_rg = list((body_part_direc2_rg[i] * body_part_direc2_rg2[j]) for i in range(m2) for j in range(m3))
    body_part_direc2_rg = body_part_direc2_rg / max(body_part_direc2_rg)

    body_part_direc1_r = body_part_direc1 * body_part_direc1_rg
    body_part_direc2_r = body_part_direc2 * body_part_direc2_rg
    

    abs1 = list(abs(x) for x in body_part_direc1_r)
    abs12 = list(abs(x) for x in body_part_direc1)
    body_part_abs_max1 = body_part_direc1_r[abs1.index(max(abs1))]
    body_part_abs_max12 = body_part_direc1[abs12.index(max(abs12))]
    
    abs2 = list(abs(x) for x in body_part_direc2_r)
    body_part_abs_max2 = body_part_direc2_r[abs2.index(max(abs2))]
    abs22 = list(abs(x) for x in body_part_direc2)
    body_part_abs_max22 = body_part_direc2[abs22.index(max(abs22))]    
    abs3 = list(abs(x) for x in body_part_direc3)
    
    body_part_abs_max3 = body_part_direc3[abs3.index(max(abs3))]
    
    body_part_direc = np.array([min(body_part_direc1), max(body_part_direc1), np.nanmean(body_part_direc1),
                         min(body_part_direc2), max(body_part_direc2), np.nanmean(body_part_direc2),
                         min(body_part_direc3), max(body_part_direc3), np.nanmean(body_part_direc3),
                         min(body_part_direc1_r), max(body_part_direc1_r), 
                         min(body_part_direc2_r), max(body_part_direc2_r), 
                         body_part_abs_max1, body_part_abs_max2, body_part_abs_max3])
    
    return body_part_direc



def get_arm_relations(features,motion):

    def get_PCA(features,motion):
        
        pca = PCA(n_components=len(features))
        m = np.transpose(motion[features])
        body_part = pca.fit_transform(m)
        body_part = np.transpose(body_part)
        
        return body_part,pca

    larm,rarm = features[0],features[1]
    if features[0] == [34,35,36]: # BKL Mocap
        motion[rarm] = -motion[rarm]
    lmat,pca = get_PCA(larm,motion)
    rmat = np.transpose(pca.transform(np.transpose(motion[rarm])))
        
    return np.corrcoef(lmat[0],rmat[0])[1,0]



def get_motion_representation(features, diffs, op):  #3rep
    print('Features are:\n', features)
    n = len(features) # number of body parts
    tot_rep = []
    if op:
        names = ['Wave two hands', 'Wave one hand','Punching',\
                 'Jumping jacks','Throwing','Bending','Sitdown Standup']  
    else: 
        names = ['Walking', 'Running', 'Jumping']
    
    for motion_type in diffs: # for each motion type
        rep_list = []   
        for motion in motion_type: # for each motion sample
            body_part_acce_list = []
            body_part_nf_list = []
            body_part_sig_f = []
            body_part_fft_list = []
            body_part_pca_list = []
            body_part_ind_list = []
            # get most important frequency of each body part (# of oscillations)            
            for i in range(n): # for each body part/joint
                # get one feature to represent a body part
                feat, ind, _ = choose_joint(i, features, motion, diffs.index(motion_type))

                body_part_ind_list.append(ind)
                body_part = get_PCA(feat, motion)
                body_part_pca = get_PCA2(feat, motion)
                body_part = savitzky_golay(body_part, 21, 3)
                
                body_part_acce_list.append(np.diff(body_part))
                body_part_nf_list.append(body_part)
                f = get_significant_frequency(body_part)
                axes_num = len(feat)
                fft_list = []
                for j in range(axes_num):
                    ft = np.fft.fft(motion[feat[j]])[1:31]
                    fft_list.append(np.hstack([np.real(ft), np.imag(ft)]))
                body_part_fft_list.append(fft_list)
                body_part_pca_list.append(body_part_pca)
                body_part_sig_f.append(f)
            body_part_sig_f = np.array(body_part_sig_f)
            
            # features' relative min magnitudes for each body part/joint
            body_part_mag2 = list(min(list(np.std(motion[x]) for x in feat)) for i in range(n))

            # body parts freqs' relative magnitude 
            body_part_freq = list(np.abs(np.fft.fft(v))[1:31] for v in body_part_nf_list)
            
            body_part_f_mag = list(max(x) for x in body_part_freq)
            body_part_f_mag = np.array(body_part_f_mag)/max(body_part_f_mag)
            
            body_part_rgs = list(np.std(body_part_pca[0]) for body_part_pca in body_part_pca_list)

            body_part_arm_dc = get_arm_relations(features,motion)
            e = np.e
            
            if max(body_part_mag2) != 0:
                body_part_mag2 = np.array(body_part_mag2)/max(body_part_mag2)  
            
            body_part_direc = get_directional_correlations(body_part_fft_list)
            
            rep = np.hstack([body_part_direc, body_part_sig_f, body_part_mag2, body_part_f_mag])#, body_part_acce_tmp]) # body_part_f_mag, body_part_mag2, body_part_sig_f
            
            if  np.isnan(rep).any():
                print(rep)
                inds = np.where(np.isnan(rep))[0]
                rep[inds] = 0
                
            print('rep',' '.join(str(round(x,2)) for x in rep))
            
            rep_list.append(rep)
        tot_rep.append(rep_list)
    
    return tot_rep




def get_train_test_data(test_name):

    if test_name == "CMU":
        source_fft, source_diff = getCMUMotion(['../Data/CMU/Walking_normal/', '../Data/CMU/Running_normal/', '../Data/CMU/Jumping_normal/'])           
        target_fft, target_diff = getKITMotion(['../Data/KIT/Walk/', '../Data/KIT/Run/', '../Data/KIT/Jump/'])

    elif test_name == "BKL":
        source_fft, source_diff = getBKLMotion()
        target_fft, target_diff = getMSRMotion()  
    
    elif test_name == "CMU->SMS":
        source_fft, source_diff = getCMUMotion(['../Data/CMU/Walking_normal/', '../Data/CMU/Running_normal/', '../Data/CMU/Jumping_normal/'])         
        target_fft, target_diff = getSMSMotion()
        
    elif test_name == "KIT->SMS":
        source_fft, source_diff = getKITMotion(['../Data/KIT/Walk/', '../Data/KIT/Run/', '../Data/KIT/Jump/'])
        target_fft, target_diff = getSMSMotion()

    elif test_name == "CMU->Mocap":
        source_fft, source_diff = getCMUMotion(['../Data/CMU/Walking/', '../Data/CMU/Jumping_medium/', '../Data/CMU/Running/'])
        target_fft, target_diff = getMocapDataMotion()

    elif test_name == "BKL->Acce":
        source_fft, source_diff = getBKLMotion()
        target_fft, target_diff = getAcceMotion()
        
    return source_fft, source_diff, target_fft, target_diff


def baseline(test_name):

    source_fft,source_diff,target_fft,target_diff = get_train_test_data(test_name)    
    source_corr = get_motion_corr(source_fft)
    target_corr = get_motion_corr(target_fft)
    
    ave_scores1,ave_scores2 = [],[]
    itr = 100
    num = 4
    s = source_corr[0][0].shape[0]*source_corr[0][0].shape[1]
    t = target_corr[0][0].shape[0]*target_corr[0][0].shape[1]
    for i in range(itr):
        print('\nIteration: ', i)
        class_lens1 = list(len(x) for x in source_corr)
        class_lens2 = list(len(x) for x in target_corr)
        num_classes = len(source_corr)
        
        num1 = int(np.average(class_lens1)/10)+1
        num2 = int(np.average(class_lens2)/10)+1
        
        X1_train, Y1_train, X1_test, Y1_test = [],[],[],[]
        X2_train, Y2_train, X2_test, Y2_test = [],[],[],[]

        for j in range(num_classes):
            idx1 = random.sample(range(class_lens1[j]),num1) 
            idx1_test = list(k for k in range(class_lens1[j]) if k not in idx1)
            idx2 = random.sample(range(class_lens2[j]),num2) 
            idx2_test = list(k for k in range(class_lens2[j]) if k not in idx2)
            X1_train += list(source_corr[j][k].reshape(1,s) for k in idx1)
            X2_train += list(target_corr[j][k].reshape(1,t) for k in idx2)
            X1_test += list(source_corr[j][k].reshape(1,s) for k in idx1_test)
            X2_test += list(target_corr[j][k].reshape(1,t) for k in idx2_test)
            Y1_train += [j] * num1
            Y2_train += [j] * num2
            Y1_test += [j] * (class_lens1[j] - num1)
            Y2_test += [j] * (class_lens2[j] - num2)
        X1_train,X1_test = np.vstack(X1_train), np.vstack(X1_test)
        X2_train,X2_test = np.vstack(X2_train), np.vstack(X2_test)
        scores1,names = test(X1_train, Y1_train, X1_test, Y1_test, 0)
        scores2,_    = test(X2_train, Y2_train, X2_test, Y2_test, 0)
        if not len(ave_scores1): 
            ave_scores1, ave_scores2 = scores1, scores2
        else: 
            ave_scores1 += scores1
            ave_scores2 += scores2      
      
    ave_scores1 /= itr
    ave_scores2 /= itr
    print('Baseline test: ', test_name, '\n')
    print('Source data')
    for i in range(len(names)):
        print(names[i], '  ', ave_scores1[i])
        
    print('\nTarget data')
    for i in range(len(names)):
        print(names[i], '  ', ave_scores2[i])    

    return     



def coarse_transfer(test_name):

    source_f_indices, target_f_indices, _ = get_feature_indices(test_name)
    _, source_diff, _, target_diff = get_train_test_data(test_name)

    # *************************************************************************
    # ************* get feature representations for each motion ***************
    # *************************************************************************

    if test_name == "CMU": 
        op = 0
    elif test_name == "BKL->Acce": 
        op = 1
    elif test_name == "CMU->SMS": 
        op = 2
    elif test_name == "KIT->SMS": 
        op = 3
    
    print('\n\nGet source representations')
    s_features_new = get_motion_representation(source_f_indices, source_diff, op)

    print('\n\nGet target representations')
    t_features_new = get_motion_representation(target_f_indices, target_diff, op)

    s_features_new = np.vstack(s_features_new)
    
    t_features_new = np.vstack(t_features_new)
    
    Y = get_Y(source_diff)
    
    X = s_features_new
    
    Y_test = get_Y(target_diff)
    X_test = t_features_new
    
    tot_scores, tot_scores2 = [], []

    it = 30
    it_s = 3

    single_domain1, single_domain2 = [], []
    
    for i in range(it_s):
        scores1,_ = test_cross_val(X,Y)
        scores2,names = test_cross_val(X_test,Y_test)  
        
        if not len(single_domain1): 
            single_domain1 = scores1
            single_domain2 = scores2
        else: 
            single_domain1 += scores1
            single_domain2 += scores2
            
    single_domain1 /= it_s
    single_domain2 /= it_s
    
    for i in range(it):    
        print('Iteration',i)
        print('\nTransfer1')
        scores,names = test_transfer(X, Y, X_test, Y_test, op)
        print('\nTransfer2')
        scores2,names = test_transfer(X_test, Y_test, X, Y, op)        
        if not len(tot_scores):
            tot_scores = np.array(scores)
            tot_scores2 = np.array(scores2)
        else:
            tot_scores += np.array(scores)
            tot_scores2 += np.array(scores2)
    tot_scores /= it
    tot_scores2 /= it
    
    print('Single domain1')
    for i in range(len(names)):
        print(names[i], single_domain1[i])
    print('\nSingle domain2')
    for i in range(len(names)):
        print(names[i], single_domain2[i])
        
    print('\nTransfer1')
    for i in range(len(names)):
        print(names[i], tot_scores[i])
    print('\nTransfer2')
    for i in range(len(names)):
        print(names[i], tot_scores2[i])
    return 



def get_confident_data(motion_indices,X):
    new_X = list(X[i,:] for i in range(X.shape[0]))
    inds = list(itertools.chain(*motion_indices))
    X_train_conf = []
    Y_train_conf = list(list(i for j in range(len(motion_indices[i])))\
                        for i in range(len(motion_indices)))
    Y_train_conf = list(itertools.chain(*Y_train_conf))
    for i in range(len(motion_indices)): # for each motion type
        for j in range(len(motion_indices[i])): # for each motion sample
            X_train_conf.append(X[motion_indices[i][j], :])
    X_train_conf = np.vstack(X_train_conf)
    
    new_X = list(new_X[i] for i in range(len(new_X)) if i not in inds)
    new_X = np.vstack(new_X)

    return X_train_conf, Y_train_conf, new_X



def sampling_training_data(X,Y,class_lens,percentage):
    if sum(class_lens) != X.shape[0] or X.shape[0] != len(Y):
        raise('[sampling_training_data] Error: Sample counts do not match')

    class_num = len(class_lens)
    indices = list(random.sample(range(class_lens[i]),
                    int(class_lens[i] * percentage)) for i in range(class_num))

    class_lens = np.cumsum(class_lens)    
    idx = indices[0]
    for i in range(1,class_num):
        idx +=  list(x + class_lens[i-1] for x in indices[i])
    
    X_train = list(X[i,:] for i in idx)
    X_train = np.vstack(X_train)
    X_test = list(X[i,:] for i in range(X.shape[0]) if i not in idx)
    X_test = np.vstack(X_test)
    Y_test = list(Y[i] for i in range(X.shape[0]) if i not in idx)
    
    return X_train, X_test, Y_test 



def augmented_coarse(test_name):
    source_f_indices, target_f_indices, feature_names = get_feature_indices(test_name)
    source_fft, source_diff, target_fft, target_diff = get_train_test_data(test_name)
    
    source_corr = get_motion_corr(source_diff)
    target_corr = get_motion_corr(target_diff)
   
    op = test_name != "CMU"
    s_features_new = get_motion_representation(source_f_indices, source_diff, op)
    t_features_new = get_motion_representation(target_f_indices, target_diff, op)

    X1,X2 = s_features_new, t_features_new #X1,X2 are the representations
    ave_scores1,ave_scores2 = [],[]

    itr = 100
    class_lens1 = list(len(x) for x in source_corr)
    class_lens2 = list(len(x) for x in target_corr)
    
    num_classes = len(source_corr)    
    num1 = int(np.average(class_lens1) / 10) + 1
    num2 = int(np.average(class_lens2) / 10) + 1  

    Y1 = list([i] * class_lens1[i] for i in range(len(class_lens1)))
    Y1 = list(itertools.chain(*Y1))
    Y2 = list([i] * class_lens2[i] for i in range(len(class_lens2)))
    Y2 = list(itertools.chain(*Y2))

    perc = 0.3

    for i in range(itr):
        X1_train,X1_test,Y1_test = sampling_training_data(X1, Y1, class_lens1, perc)    
        X2_train,X2_test,Y2_test = sampling_training_data(X2, Y2, class_lens2, perc)  

        print('CMU recognize KIT')
        motion_indices2 = motion_recognition(X1, Y1, X2_train, num2, num_classes)
        print('KIT recognize CMU')
        motion_indices1 = motion_recognition(X2, Y2, X1_train, num1, num_classes)
        
        X1_train_conf, Y1_train_conf, _ = get_confident_data(motion_indices1, X1_train)
        X2_train_conf, Y2_train_conf, _ = get_confident_data(motion_indices2, X2_train)
        
        X1_train = np.vstack([X1, X2_train_conf])
        Y1_train = Y1 + Y2_train_conf
        X2_train = np.vstack([X2, X1_train_conf])
        Y2_train = Y2 + Y1_train_conf
        print('Y1_train_conf', Y1_train_conf, 'Y2_train_conf', Y2_train_conf)
        print('\nIteration: ', i)

        if test_name == "CMU": 
            print('Augment CMU with KIT data to categorize KIT')
        else: 
            print('Augment BKL with ACCE to categorize ACCE')
        scores1, _ = test_transfer(X1_train, Y1_train, X2_test, Y2_test, op)

        if test_name == "CMU": 
            print('\nAugment KIT with CMU data to categorize CMU')
        else: 
            print('\nAugment ACCE with BKL to categorize BKL')
        scores2, names = test_transfer(X2_train, Y2_train, X1_test, Y1_test, op)
        
        if not len(ave_scores1): ave_scores1 = np.array(scores1)
        else: ave_scores1 += np.array(scores1)
        if not len(ave_scores2): ave_scores2 = np.array(scores2)
        else: ave_scores2 += np.array(scores2)   

      
    ave_scores1 /= itr
    ave_scores2 /= itr
    
    print('\nSource recognize Target data')
    for i in range(len(names)):
        print(names[i], '  ', ave_scores1[i]) 
        
    print('\nTarget recognize Source data')
    for i in range(len(names)):
        print(names[i], '  ', ave_scores2[i])
    return



def coarse2fine(test_name):
    source_f_indices, target_f_indices, feature_names = get_feature_indices(test_name)
    source_fft, source_diff, target_fft, target_diff = get_train_test_data(test_name)
    
    source_corr = get_motion_corr(source_fft)
    target_corr = get_motion_corr(target_fft)
    
    op = test_name != "CMU"
    s_features_new = get_motion_representation(source_f_indices,source_diff,op)
    t_features_new = get_motion_representation(target_f_indices,target_diff,op)
        
    X1_corr,Y1 = get_data(source_corr)    
    X2_corr,Y2 = get_data(target_corr)

    X1,X2 = s_features_new, t_features_new
    ave_scores1,ave_scores2 = [],[]
    class_lens1 = list(len(x) for x in source_corr)
    class_lens2 = list(len(x) for x in target_corr)
    num_classes = len(source_corr)
    
    itr = 100
    num1 = int(np.average(class_lens1) / 10) + 1
    num2 = int(np.average(class_lens2) / 10) + 1
    perc = 0.3
    for i in range(itr):
        X1_train, Y1_test, X1_corr_train, X1_corr_test = partition_data(X1, Y1, X1_corr, class_lens1, perc)    
        X2_train, Y2_test, X2_corr_train, X2_corr_test = partition_data(X2, Y2, X2_corr, class_lens2, perc)  

        motion_indices2 = motion_recognition(X1, Y1, X2_train, num2, num_classes)
        motion_indices1 = motion_recognition(X2, Y2, X1_train, num1, num_classes)
        
        X1_train_conf, Y1_train_conf, _ = get_confident_data(motion_indices1, X1_corr_train)
        X2_train_conf, Y2_train_conf, _ = get_confident_data(motion_indices2, X2_corr_train)
        
        if test_name == "CMU": 
            print('Use CMU data to recognize KIT data')
        else: 
            print('Use BKL data to recognize ACCE data')

        scores1, _ = test(X1_train_conf, Y1_train_conf, X1_corr_test, Y1_test, 0)

        if test_name == "CMU": 
            print('Use KIT data to recognize CMU data')
        else: 
            print('Use ACCE data to recognize BKL data')

        scores2, names = test(X2_train_conf, Y2_train_conf, X2_corr_test, Y2_test, 0)
        
        if not len(ave_scores1): 
            ave_scores1 = np.array(scores1)
        else: 
            ave_scores1 += np.array(scores1)
        if not len(ave_scores2): 
            ave_scores2 = np.array(scores2)
        else: 
            ave_scores2 += np.array(scores2)   
        print('scores1', scores1)
        print('scores2', scores2)
      
    ave_scores1 /= itr
    ave_scores2 /= itr
    if test_name == 'CMU': 
        f_name = "results_abs_" + str(perc) + "_CMU" + str(num1) +"_KIT" + str(num2)
    else: 
        f_name = "results_abs_"+ str(perc)  + "_BKL" + str(num1) +"_ACCE" + str(num2) + "_new"
    
    fptr = open(f_name,'w')
    fptr.write('Percentage of unknown data included in training set: ' + str(perc) + '\n')
    fptr.write('Number of target sample data we picked: ' + str(num1) + ' for Source and '+str(num2) + " for Target")
    fptr.write(f_name + str('\n'))
    fptr.write('Number of iterations: ' + str(itr) + '\n')
    
    print('\nSource recognize Target data')
    fptr.write('\nSource recognize Target data\n')
    for i in range(len(names)):
        print(names[i],'  ',ave_scores1[i])
        fptr.write(names[i] + ' ' + str(ave_scores1[i]) + '\n') 
        
    print('\nTarget recognize Source data')
    fptr.write('\nTarget recognize Source data\n')
    for i in range(len(names)):
        print(names[i],'  ',ave_scores2[i])
        fptr.write(names[i] + ' ' + str(ave_scores2[i]) + '\n') 
    fptr.close()
    return 



if __name__ == '__main__':
    options = sys.argv
    print(options)
    func = int(options[1])
    dataset = int(options[2])
    if dataset == 0:
        test_name = "CMU"
    elif dataset == 1:
        test_name = "BKL->Acce"
    elif dataset == 2:
        test_name = "CMU->SMS"
    elif dataset == 3:
        test_name = "KIT->SMS"
    
    if func == 0:
        result = baseline(test_name)
    elif func == 1:
        result = coarse_transfer(test_name)
    elif func == 2:
        result = coarse2fine(test_name)
    elif func == 3:
        result = augmented_coarse(test_name)




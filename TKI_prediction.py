# -*- coding: utf-8 -*-
# 产生热量图的数据npy文件

from __future__ import division, print_function
import time
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
import copy
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LassoCV

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# from skfeature.example.temp import datasave

from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import LCSI
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import trace_ratio
from skfeature.function.sparse_learning_based import ll_l21
from skfeature.function.sparse_learning_based import ls_l21
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import NDFS
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.sparse_learning_based import UDFS
from skfeature.function.statistical_based import CFS
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import low_variance
from skfeature.function.statistical_based import t_score
from skfeature.function.streaming import alpha_investing
from skfeature.function.structure import graph_fs,group_fs,tree_fs
from skfeature.function.wrapper import decision_tree_backward
from skfeature.function.wrapper import decision_tree_forward
from skfeature.function.wrapper import svm_backward
from skfeature.function.wrapper import svm_forward
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle

def load_data():
    data0 = pd.read_excel('D:\\projects\\Chunxia He\\project1\\TKI.xls')
    data = data0.values
    Data = data[:,:35]
    label_ORR = data[:,35]
    label_PFS_time = data[:,36]
    label_PFS = data[:,38]
    label_OS_time = data[:,39]
    label_OS = data[:,40]

    for i in range(len(label_PFS)):
        if label_PFS[i]==2:
            label_PFS[i]=0

    Data = np.array(Data).astype('float64')
    label_ORR = np.array(label_ORR).astype('int32')
    label_PFS_time = np.array(label_PFS_time).astype('float64')
    label_PFS = np.array(label_PFS).astype('int32')
    label_OS_time = np.array(label_OS_time).astype('float64')
    label_OS = np.array(label_OS).astype('int32')

    return Data,label_ORR,label_PFS_time,label_PFS,label_OS_time,label_OS

def functionoperator(operator, X, y, nfolds, clf, **kwargs):
    case_num = X.shape[0]
    case_idx2 = range(case_num)
    case_idx2 = list(case_idx2)
    np.random.shuffle(case_idx2)

    X = X[case_idx2]
    y = y[case_idx2]


    # information_theoretical_based feature selection
    if 'feanum' in kwargs.keys():
        num_fea = kwargs['feanum']

    if 'FLAG_DataBalance' in kwargs.keys():
        FLAG_DataBalance = kwargs['FLAG_DataBalance']
    else:
        FLAG_DataBalance = False

    tmp_auc = []
    tmp_acc = []
    tmp_sen = []
    tmp_spe = []

    # 分离数据
    # split data into nfolds folds
    SKF = StratifiedKFold(n_splits=nfolds, random_state=0, shuffle=True)
    finy_test = []
    finypredict_prob = []
    featuresidex = []

    for tr_idx,te_idx in SKF.split(X,y):
        # separate data into train group and test group
        X_train = X[tr_idx]
        y_train = y[tr_idx]
        X_test = X[te_idx]
        y_test = y[te_idx]

        # 数据归一化
        # data normalization
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # obtain the index of selected features on the training set
        if operator == "CIFE_main":  # 1
            idx = CIFE.cife(X_train, y_train, n_selected_features=num_fea)
        if operator == "CMIM_main":  # 2
            idx = CMIM.cmim(X_train, y_train, n_selected_features=num_fea)
        if operator == "DISR_main":  # 3
            idx = DISR.disr(X_train, y_train, n_selected_features=num_fea)
        if operator == "FCBF_main":  # 4
            idx= FCBF.fcbf(X_train, y_train, delta=0)
        if operator == "ICAP_main":  # 5
            idx= ICAP.icap(X_train, y_train, n_selected_features=num_fea)
        if operator == "JMI_main":  # 6
            idx = JMI.jmi(X_train, y_train, n_selected_features=num_fea)
        if operator =='LCSI_main':
            idx,_,_ = LCSI.lcsi(X_train,y_train,gamma=0,function_name = 'JMI',n_selected_features=num_fea)
        if operator == "MIFS_main":  # 7
            idx = MIFS.mifs(X_train, y_train, n_selected_features=num_fea)
        if operator == "MIM_main":  # 8
            idx = MIM.mim(X_train, y_train, n_selected_features=num_fea)
        if operator == "MRMR_main":  # 9
            idx = MRMR.mrmr(X_train, y_train, n_selected_features=num_fea)
        if operator == "fisher_score_main":  # 10
            idx = fisher_score.fisher_score(X_train, y_train)
        if operator == "lap_score_main":  # 11
            kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            idx = lap_score.lap_score(X_train, W=W)
        if operator == "reliefF_main":  # 12
            idx = reliefF.reliefF(X_train, y_train)
        if operator == "SPEC_main":  # 13
            # specify the second ranking function which uses all except the 1st eigenvalue
            kwargs_style = {'style': 0}
            # obtain the scores of features
            idx = SPEC.spec(X_train, **kwargs_style)
        if operator == "trace_ratio_main":  # 14
            idx = trace_ratio.trace_ratio(X_train, y_train, num_fea, style='fisher')
        if operator == "ll_l21_main":  # 15
            idx = ll_l21.proximal_gradient_descent(X_train, y_train, 0.1, verbose=False)
        if operator == "ls_l21_main":  # 16
            idx = ls_l21.proximal_gradient_descent(X_train, y_train, 0.1, verbose=False)
        if operator == "MCFS_main":  # 17
            # construct affinity matrix
            kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            num_cluster = 2  # specify the number of clusters, it is usually set as the number of classes in the ground truth
            # obtain the feature weight matrix
            idx = MCFS.mcfs(X_train, n_selected_features=num_fea, W=W, n_clusters=num_cluster)
        if operator == "NDFS_main":  # 18
            # construct affinity matrix
            kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            # obtain the feature weight matrix
            idx = NDFS.ndfs(X_train, W=W, n_clusters=2)
        if operator == "RFS_main":  # 19
            idx = RFS.rfs(X_train, y_train, gamma=0.1)
        if operator == "UDFS_main":  # 20
            # obtain the feature weight matrix
            idx = UDFS.udfs(X_train, gamma=0.1, n_clusters=2)
        if operator == "CFS_main":  # 21
            idx = CFS.cfs(X_train, y_train)
        if operator == "chi_square_main":  # 22
            # obtain the chi-square score of each feature
            idx = chi_square.chi_square(X_train, y_train)
        if operator == "f_score_main":  # 23
            # obtain the chi-square score of each feature
            idx = f_score.f_score(X_train, y_train,mode='index')
        if operator == "gini_index_main":  # 24
            # obtain the chi-square score of each feature
            idx = gini_index.gini_index(X_train, y_train)
        if operator =="low_variance_main":
            p = 0.1
            idx = low_variance.low_variance_feature_selection(X_train,p*(1-p))
        if operator == "t_score_main":  # 25
            idx = t_score.t_score(X_train, y_train)
        if operator == "graph_fs_main":
            w, obj, q = graph_fs.graph_fs(X_train,y_train)
            idx = graph_fs.feature_ranking(w)
        if operator == "alpha_investing_main":  # 26
            idx = alpha_investing.alpha_investing(X_train, y_train, 0.05, 0.05)
        if operator == "decision_tree_backward_main":  # 27
            idx = decision_tree_backward.decision_tree_backward(X_train, y_train, num_fea)
        if operator == "decision_tree_forward_main":  # 28
            idx = decision_tree_forward.decision_tree_forward(X_train, y_train, num_fea)
        if operator == "svm_backward_main":  # 29
            idx = svm_backward.svm_backward(X_train, y_train, num_fea)
        if operator == "svm_forward_main":  # 30
            idx = svm_forward.svm_forward(X_train, y_train, num_fea)

        # obtain the dataset on the selected features
        idx = np.array(idx).astype('int32')
        X_train_selected = X_train[:, idx[0:num_fea]]
        X_test_selected = X_test[:, idx[0:num_fea]]

        featuresidex.append(idx[0:num_fea])

        # 数据均衡
        # Data Balancing for the training dataset using SMOTE
        if FLAG_DataBalance:
            sm = SMOTE(kind='regular')  # kind = ['regular', 'borderline1', 'borderline2', 'svm']
            X_selected_resampled, y_selected_resampled = sm.fit_sample(X_train_selected, y_train)
            # train a classification model with the selected features on the training dataset
            clf.fit(X_selected_resampled, y_selected_resampled)
        else:
            # train a classification model with the selected features on the training dataset
            clf.fit(X_train_selected, y_train)

        # predict the class labels of test data
        y_predict = clf.predict(X_test_selected)
        # evaluation of the trained model on the test data
        sen_o, spe_o = cal_sen_spe(y_test, y_predict)
        acc_o = accuracy_score(y_test, y_predict)
        y_predict_prob = clf.predict_proba(X_test_selected)
        finy_test.extend(y_test)
        finypredict_prob.extend(y_predict_prob[:, 1])
        auc_o = roc_auc_score(y_test, y_predict_prob[:, 1])
        tmp_auc.append(auc_o)
        tmp_acc.append(acc_o)
        tmp_sen.append(sen_o)
        tmp_spe.append(spe_o)

    auc01 = roc_auc_score(finy_test, finypredict_prob)
    AUC = float(sum(tmp_auc)) / len(tmp_auc)
    ACC = float(sum(tmp_acc)) / len(tmp_acc)
    SEN = float(sum(tmp_sen)) / len(tmp_sen)
    SPE = float(sum(tmp_spe)) / len(tmp_spe)
    return AUC, ACC, SEN, SPE, finy_test, finypredict_prob, auc01, featuresidex

def model_save1(model,model_name,operator0,X_train,y_train,num_fea):
    operator = operator0+'_main'

    if operator == "CIFE_main":  # 1
        idx = CIFE.cife(X_train, y_train, n_selected_features=num_fea)
    if operator == "CMIM_main":  # 2
        idx = CMIM.cmim(X_train, y_train, n_selected_features=num_fea)
    if operator == "DISR_main":  # 3
        idx = DISR.disr(X_train, y_train, n_selected_features=num_fea)
    if operator == "FCBF_main":  # 4
        idx = FCBF.fcbf(X_train, y_train, delta=0)
    if operator == "ICAP_main":  # 5
        idx = ICAP.icap(X_train, y_train, n_selected_features=num_fea)
    if operator == "JMI_main":  # 6
        idx = JMI.jmi(X_train, y_train, n_selected_features=num_fea)
    if operator == 'LCSI_main':
        idx, _, _ = LCSI.lcsi(X_train, y_train, gamma=0, function_name='JMI', n_selected_features=num_fea)
    if operator == "MIFS_main":  # 7
        idx = MIFS.mifs(X_train, y_train, n_selected_features=num_fea)
    if operator == "MIM_main":  # 8
        idx = MIM.mim(X_train, y_train, n_selected_features=num_fea)
    if operator == "MRMR_main":  # 9
        idx = MRMR.mrmr(X_train, y_train, n_selected_features=num_fea)
    if operator == "fisher_score_main":  # 10
        idx = fisher_score.fisher_score(X_train, y_train)
    if operator == "lap_score_main":  # 11
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X_train, **kwargs_W)
        idx = lap_score.lap_score(X_train, W=W)
    if operator == "reliefF_main":  # 12
        idx = reliefF.reliefF(X_train, y_train)
    if operator == "SPEC_main":  # 13
        # specify the second ranking function which uses all except the 1st eigenvalue
        kwargs_style = {'style': 0}
        # obtain the scores of features
        idx = SPEC.spec(X_train, **kwargs_style)
    if operator == "trace_ratio_main":  # 14
        idx = trace_ratio.trace_ratio(X_train, y_train, num_fea, style='fisher')
    if operator == "ll_l21_main":  # 15
        idx = ll_l21.proximal_gradient_descent(X_train, y_train, 0.1, verbose=False)
    if operator == "ls_l21_main":  # 16
        idx = ls_l21.proximal_gradient_descent(X_train, y_train, 0.1, verbose=False)
    if operator == "MCFS_main":  # 17
        # construct affinity matrix
        kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X_train, **kwargs_W)
        num_cluster = 2  # specify the number of clusters, it is usually set as the number of classes in the ground truth
        # obtain the feature weight matrix
        idx = MCFS.mcfs(X_train, n_selected_features=num_fea, W=W, n_clusters=num_cluster)
    if operator == "NDFS_main":  # 18
        # construct affinity matrix
        kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X_train, **kwargs_W)
        # obtain the feature weight matrix
        idx = NDFS.ndfs(X_train, W=W, n_clusters=2)
    if operator == "RFS_main":  # 19
        idx = RFS.rfs(X_train, y_train, gamma=0.1)
    if operator == "UDFS_main":  # 20
        # obtain the feature weight matrix
        idx = UDFS.udfs(X_train, gamma=0.1, n_clusters=2)
    if operator == "CFS_main":  # 21
        idx = CFS.cfs(X_train, y_train)
    if operator == "chi_square_main":  # 22
        # obtain the chi-square score of each feature
        idx = chi_square.chi_square(X_train, y_train)
    if operator == "f_score_main":  # 23
        # obtain the chi-square score of each feature
        idx = f_score.f_score(X_train, y_train, mode='index')
    if operator == "gini_index_main":  # 24
        # obtain the chi-square score of each feature
        idx = gini_index.gini_index(X_train, y_train)
    if operator == "low_variance_main":
        p = 0.1
        idx = low_variance.low_variance_feature_selection(X_train, p * (1 - p))
    if operator == "t_score_main":  # 25
        idx = t_score.t_score(X_train, y_train)
    if operator == "graph_fs_main":
        w, obj, q = graph_fs.graph_fs(X_train, y_train)
        idx = graph_fs.feature_ranking(w)
    if operator == "alpha_investing_main":  # 26
        idx = alpha_investing.alpha_investing(X_train, y_train, 0.05, 0.05)
    if operator == "decision_tree_backward_main":  # 27
        idx = decision_tree_backward.decision_tree_backward(X_train, y_train, num_fea)
    if operator == "decision_tree_forward_main":  # 28
        idx = decision_tree_forward.decision_tree_forward(X_train, y_train, num_fea)
    if operator == "svm_backward_main":  # 29
        idx = svm_backward.svm_backward(X_train, y_train, num_fea)
    if operator == "svm_forward_main":  # 30
        idx = svm_forward.svm_forward(X_train, y_train, num_fea)

        # obtain the dataset on the selected features
    idx = np.array(idx).astype('int32')
    X_train_selected = X_train[:, idx[0:num_fea]]

    model.fit(X_train_selected,y_train)



    with open('G:\\projects\\Chunxia He\\project1\\best_10_models\\'+model_name+'_'+operator0+'.pkl', 'wb') as f:
        pickle.dump(model, f)

    np.save('G:\\projects\\Chunxia He\\project1\\best_10_models\\'+model_name+'_'+operator0+'.npy',idx)

def plot_figure(data,FeatureSelectors,classifiers,figure_name):
    f, ax = plt.subplots(figsize=(21, 6))  # figsize=(8, 6)
    plt.subplots_adjust(left=0.2, bottom=0.3, right=1.0, top=0.9, hspace=0.2, wspace=0.3)
    sns.heatmap(data, vmin=0.6, vmax=0.95, cmap='Blues', ax=ax, annot=True,
                fmt='.2f')  # , linewidths=.05, linecolor='white'

    plt.xticks(rotation=30)

    ax.set_xticklabels(FeatureSelectors, horizontalalignment='right', rotation=30)
    # ax.set_yticks(range(len(FeatureSelectors)))

    ax.set_yticklabels(classifiers, rotation=30)
    ax.set_xlabel('Feature Selection Methods', fontsize=20)
    ax.set_ylabel('Classifiers', fontsize=20)

    plt.savefig('D:\\projects\\Chunxia He\\project1\\paper\\revise\\f10f15f20\\predictive_'+figure_name+'.tif',format='tif', dpi=300, bbox_inches='tight')
    # plt.show()

def cal_sen_spe(y_actual, y_pred):
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
    for i in range(len(y_pred)):
        if y_actual[i] == 0 and y_actual[i] != y_pred[i]:
            FP += 1
    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
    for i in range(len(y_pred)):
        if y_actual[i] == 1 and y_actual[i] != y_pred[i]:
            FN += 1

    if ((y_actual == y_pred).all() and np.sum(y_actual) == len(y_actual)) or (
                (y_actual == y_pred).all() and np.sum(y_actual) == 0):  # all 1s or 0s
        Sensitivity = 1.0
        Specificity = 1.0
    elif (y_actual == y_pred).any() == False:
        Sensitivity = 0.0
        Specificity = 0.0
    else:
        Sensitivity = float(TP) / (float(TP + FN) + 0.000000001)
        Specificity = float(TN) / (float(TN + FP) + 0.000000001)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return (Sensitivity, Specificity)

def evaluation_main():
    Data,label_ORR,labelPFS_time,label_PFS,label_OS_time,label_OS = load_data()
    t0 = time.time()

    # 12 base classifiers
    classifiers = {"LR": LogisticRegression(random_state=2024),
                   "LDA": LinearDiscriminantAnalysis(),
                   "SVM": SVC(probability=True,random_state=2024),
                   "NB": GaussianNB(),
                   "KNN": KNeighborsClassifier(),
                   "DT": DecisionTreeClassifier(random_state=2024),
                   "Bagging": BaggingClassifier(random_state=2024),
                   "RF": RandomForestClassifier(random_state=2024),
                   "AdaB": AdaBoostClassifier(random_state=2024),
                   "XGB": XGBClassifier(),
                   "LGBM": LGBMClassifier(random_state=2024),
                   "CatB": CatBoostClassifier(random_state=2024, verbose=False)
                   }

    # 25 feature selection methods
    FeatureSelectors = ["CIFE",  # 1
                        "CMIM",  # 2
                        "DISR",  # 3
                        "FCBF",  # 4
                        "ICAP",  # 5
                        "JMI",  # 6
                        "LCSI", #7
                        "MIFS",  # 8
                        "MIM",  # 9
                        "MRMR",  # 10
                        "fisher_score",  # 11
                        "lap_score",  # 12
                        "reliefF",  # 13
                        "SPEC",  # 14
                        "trace_ratio",  # 15
                        "ll_l21",  # 16
                        "ls_l21",  # 17
                        "MCFS",  # 18
                        # "NDFS",  # 19
                        "RFS",  # 20
                        "UDFS",  # 21
                        "CFS",  # 22
                        "f_score",  # 23
                        "gini_index",  # 24
                        "t_score",  # 25
                        ]

    nfolds = 5  # nfolds of cross-validation
    num_fea = 20  # number of selected features

    total_AUC = []
    total_ACC = []
    total_SEN = []
    total_SPE = []
    total_finy_test = []
    total_finypredict_prob = []
    total_auc01 = []
    total_featuresidex = []
    for name1, clf in classifiers.items():
        t1 = time.time()
        print('---------------------%s---------------------' % name1)
        tmp_AUC = []
        tmp_ACC = []
        tmp_SEN = []
        tmp_SPE = []
        tmp_finy_test = []
        tmp_finypredict_prob = []
        tmp_auc01 = []
        tmp_featuresidex = []
        for name2 in FeatureSelectors:
            time1 = time.time()
            AUC, ACC, SEN, SPE, finy_test, finypredict_prob, auc01, featuresidex = functionoperator(name2 + "_main",
                                                                                                    Data, label_PFS, nfolds,
                                                                                                    clf,
                                                                                                    feanum=num_fea,
                                                                                                    FLAG_DataBalance=0)
            time2 = time.time()
            print("本次用时：", time2 - time1)
            print('AUC:%.3f, ACC:%.3f, SEN:%.3f, SPE:%.3f ---- %s' % (AUC, ACC, SEN, SPE, name2))
            tmp_AUC.append(AUC)
            tmp_ACC.append(ACC)
            tmp_SEN.append(SEN)
            tmp_SPE.append(SPE)
            tmp_finy_test.append(finy_test)
            tmp_finypredict_prob.append(finypredict_prob)
            tmp_auc01.append(auc01)
            tmp_featuresidex.append(featuresidex)

        total_finy_test.append(tmp_finy_test)
        total_finypredict_prob.append(tmp_finypredict_prob)
        total_auc01.append(tmp_auc01)
        total_featuresidex.append(tmp_featuresidex)
        total_AUC.append(tmp_AUC)
        total_ACC.append(tmp_ACC)
        total_SEN.append(tmp_SEN)
        total_SPE.append(tmp_SPE)
        t2 = time.time() - t1
        print("------------------------------%s-----time:" % name1, t2)
    matrix_AUC = np.array(total_AUC).reshape([len(classifiers), len(FeatureSelectors)])
    plot_figure(matrix_AUC,FeatureSelectors,classifiers,'AUC')
    matrix_ACC = np.array(total_ACC).reshape([len(classifiers), len(FeatureSelectors)])
    plot_figure(matrix_ACC,FeatureSelectors,classifiers,'ACC')
    matrix_SEN = np.array(total_SEN).reshape([len(classifiers), len(FeatureSelectors)])
    plot_figure(matrix_SEN,FeatureSelectors,classifiers,'SEN')
    matrix_SPE = np.array(total_SPE).reshape([len(classifiers), len(FeatureSelectors)])
    plot_figure(matrix_SPE,FeatureSelectors,classifiers,'SPE')
    matrix_add = matrix_AUC + matrix_ACC + matrix_SEN + matrix_SPE

    #save top-5 feature selection methods
    # total_featuresidex = np.array(total_featuresidex)
    # total_featuresidex1 = total_featuresidex[:,[5,6,13,15,22],:,:]
    # total_featuresidex2 = np.reshape(total_featuresidex1,(300,5))
    # df = pd.DataFrame(total_featuresidex2)
    # df.to_excel('G:\\projects\\Chunxia He\\project1\\results\\feature_selection_index.xlsx')

    # plan1 save the best ten models
    # top_indices = np.argsort(matrix_add.flatten())[::-1][:10]
    #
    # rows = top_indices // matrix_add.shape[1]
    # cols = top_indices % matrix_add.shape[1]
    #
    # classifiers_list=list(classifiers)
    #
    # for i in range(len(cols)):
    #     fs_method = FeatureSelectors[cols[i]]
    #     classifier_name = classifiers_list[rows[i]]
    #     classifier_model = classifiers[classifier_name]
    #     model_save1(classifier_model,classifier_name,fs_method,Data,label_PFS,num_fea)

    #plan2 using the top-5 features to train models
    # Data_sel = Data[:,[18,16,33,1,0]]
    # for name1, clf in classifiers.items():
    #     clf.fit(Data_sel,label_PFS)
    #     with open('G:\\projects\\Chunxia He\\project1\\top_5_feature_models\\' + name1 + '.pkl',
    #               'wb') as f:
    #         pickle.dump(clf, f)



    t3 = time.time() - t0
    print("total time:", t3)


if __name__ == '__main__':
    evaluation_main()

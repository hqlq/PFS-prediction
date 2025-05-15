import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score,roc_auc_score
from lifelines.utils import concordance_index

def load_data():
    data0 = pd.read_excel('D:\\projects\\Chunxia He\\project1\\Validation_TKI_3.xlsx')
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

def load_best_models(model_name,data,y_test):
    model_path ='D:\\projects\\Chunxia He\\project1\\best_10_models\\'+model_name+'.pkl'
    idx_path = 'D:\\projects\\Chunxia He\\project1\\best_10_models\\'+model_name+'.npy'
    with open(model_path,'rb') as f:
        model = pickle.load(f)
    idx = np.load(idx_path)
    test_data = data[:,idx[:5]]
    y_predict = model.predict(test_data)
    sen_o, spe_o = cal_sen_spe(y_test, y_predict)
    acc_o = accuracy_score(y_test, y_predict)
    y_predict_prob = model.predict_proba(test_data)
    auc_o = roc_auc_score(y_test, y_predict_prob[:, 1])
    cindex = concordance_index(y_test, y_predict)
    return [acc_o,auc_o,sen_o,spe_o,cindex],y_predict_prob

def load_top_models(model_name,data,y_test):
    model_path = 'D:\\projects\\Chunxia He\\project1\\top_5_feature_models\\' + model_name + '.pkl'
    test_data = data[:,[18,16,33,1,0]]
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    y_predict = model.predict(test_data)
    sen_o, spe_o = cal_sen_spe(y_test, y_predict)
    acc_o = accuracy_score(y_test, y_predict)
    y_predict_prob = model.predict_proba(test_data)
    auc_o = roc_auc_score(y_test, y_predict_prob[:, 1])
    cindex = concordance_index(y_test,y_predict)
    return [acc_o, auc_o, sen_o, spe_o,cindex], y_predict_prob




if __name__ == '__main__':
    Data, label_ORR, labelPFS_time, label_PFS, label_OS_time, label_OS = load_data()

    '''testing on saved model1'''
    # print('testing on best models')
    # model_name1=['CB_SPEC','LDA_ll_l21','LGBM_f_score','LGBM_LCSI','LR_ll_l21','LR_SPEC','SVM_JMI','SVM_LCSI','SVM_ll_l21','XGB_ll_l21']
    # for i in range(len(model_name1)):
    #     performence,prob = load_best_models(model_name1[i],Data,label_PFS)
    #     print(performence)

    '''testing on saved model2'''
    print('testing on best features')
    model_name2=["LR","LDA","SVM","NB","KNN","Decision Tree","Bagging","Random Forest","AdaBoosting","XGB","LGBM","CB"]
    for i in range(len(model_name2)):
        performence, prob = load_top_models(model_name2[i], Data, label_PFS)
        print(performence)

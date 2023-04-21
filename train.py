from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np
import os
import random
import utils
import joblib
import cv2
import xml.etree.ElementTree as ET
def load_data(data_path,shuffle=True):
    N=4000
    #load the positive and negative data
    p_list = []
    n_list = []
    folder_list = os.listdir(data_path)
    for i in range(len(folder_list)):
        folder = folder_list[i]
        if os.path.exists(os.path.join(data_path,folder,'p')):
            p_sub = os.listdir(os.path.join(data_path, folder, 'p'))
        else:
            p_sub = []
        n_sub = os.listdir(os.path.join(data_path,folder,'n'))
        for j in range(len(p_sub)):
            p_list.append(os.path.join(folder,'p',p_sub[j]))
        for j in range(len(n_sub)):
            n_list.append(os.path.join(folder,'n',n_sub[j]))
    ng_p = []
    ng_n = []
    if shuffle:
        random.shuffle(p_list)
        random.shuffle(n_list)
    for i in range(len(p_list)):
        ng_p.append(np.load(os.path.join(data_path,p_list[i])))
    for i in range(len(n_list)):
        ng_n.append(np.load(os.path.join(data_path,n_list[i])))
    p_data = np.stack(ng_p,axis=0) #dimension: (num, h, w)
    n_data = np.stack(ng_n,axis=0) #dimension: (num, h, w)
    #we need reshape the data to (num, h*w)
    p_data = p_data.reshape(p_data.shape[0],-1)
    n_data = n_data.reshape(n_data.shape[0],-1)
    if p_data.shape[0] > N:
        class_size = N
    else:
        class_size = p_data.shape[0]
    ng = np.concatenate((p_data[:class_size,:],n_data[:class_size,:])) #dimension: (2*num, h*w)
    #create the labels
    y = np.concatenate((np.ones(class_size),-np.ones(class_size)))
    print(ng.shape)
    print(y.shape)
    return ng,y

def load_2nd_data(NG_path,anno_path, clf_one, width, height):

    path = os.path.join(NG_path,str(width)+'_'+str(height))
    NG_list = os.listdir(path)
    score_list = []
    label_list = []
    #looping over each image's NG
    for i in range(len(NG_list)):
        ng = np.load(os.path.join(path,NG_list[i]))
        scores = np.zeros((ng.shape[0]-7,ng.shape[1]-7))
        labels = np.zeros((ng.shape[0]-7,ng.shape[1]-7))
        #sliding over the NG
        for y in range(ng.shape[0]-7):
            for x in range(ng.shape[1]-7):
                scores[x,y] = clf_one.decision_function(ng[y:y+8,x:x+8].reshape(1,-1))
                labels[x,y] = utils.check_label(anno_path,NG_list[i].split('.')[0],x,y,width,height) #we need to check the label
        #we need to do the non-maximum suppression
        #the nms is done within each quantized size
        scores_nms,labels_nms = utils.non_maximum_suppression(scores,labels)
        score_list.append(scores_nms)
        label_list.append(labels_nms)
    #making the data balanced
    score_list = np.concatenate(score_list,axis=0)
    label_list = np.concatenate(label_list,axis=0)
    positive_number = np.sum(label_list==1)
    positive_index = np.where(label_list==1)
    negative_index = np.where(label_list==-1)
    negative_index = random.sample(list(negative_index[0]),positive_number)
    index = np.concatenate((positive_index[0],negative_index),axis=0)
    score_list = score_list[index]
    score_list = score_list.reshape(-1,1)
    label_list = label_list[index]
    # print out how many positive and negative samples we have
    print("Number of data samples loaded: ", score_list.shape[0])
    return score_list,label_list

def train_stage_I():
    #Training Stage I

    #define the Stage I Model
    clf_one = make_pipeline(StandardScaler(),
                        LinearSVC(fit_intercept=False, random_state=0, tol=1e-3, max_iter=5000))
    #prepare the training data
    data_path = "/media/zbh/Second/BING/VOC2007/trainval/sample"
    ng,y = load_data(data_path)
    print("load data done")
    #Fitting the ng feature to objectness labels
    clf_one.fit(ng,y)
    #saving the weights of stage I model
    Stage_I_model = "/media/zbh/Second/BING/models/stage_I.pkl"
    joblib.dump(clf_one, Stage_I_model)



def train_stage_II():
    #Training Stage II
    height = [10, 20, 40, 80, 160, 320]
    width = [10, 20, 40, 80, 160, 320]
    #Load the Stage I Model
    Stage_I_model = "/media/zbh/Second/BING/models/stage_I.pkl"
    clf_one = joblib.load(Stage_I_model)
    #load the training data
    NG_path = "/media/zbh/Second/BING/VOC2007/trainval/NG"
    anno_path = "/media/zbh/Second/BING/VOC2007/trainval/Annotations"
    for i in range(len(width)):
        for j in range(len(height)):
            score,y = load_2nd_data(NG_path, anno_path, clf_one, width[i], height[j])
            # define the Stage II Model
            clf_two = make_pipeline(StandardScaler(),
                                    LinearSVC(fit_intercept=True, random_state=0, tol=1e-4,max_iter=5000))
            clf_two.fit(score,y)
            if score.shape[0] <100:
                continue
            #saving the weights of stage II model
            Stage_II_model = "/media/zbh/Second/BING/models/II/"+str(width[i])+"_"+str(height[j])+".pkl"
            joblib.dump(clf_two, Stage_II_model)


if __name__ == '__main__':
    # train_stage_I()
    # print("Stage I training finished!")
    train_stage_II()
    print("Stage II training finished!")
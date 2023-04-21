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

img_path = ''
img = cv2.imread(img_path)
available_size = []
model_I_path = '/media/zbh/Second/BING/models/I/stage_I.pkl'
model_II_path = '/media/zbh/Second/BING/models/II'
II_models = os.listdir(model_II_path)

anno_file = os.path.join(anno_path,img_name+'.xml')
tree = ET.parse(anno_file)
root = tree.getroot()
size = root.find('size')
w = int(size.find('width').text)
h = int(size.find('height').text)

#load the stage I model
model_I = joblib.load(model_I_path)
#getting the available resize sizes
for i in range(len(II_models)):
    width = II_models[i].split('_')[0]
    height = II_models[i].split('_')[1]
    available_size.append((int(width),int(height)))

#resize the image and get the ng
resized_imgs = []
for i in range(len(available_size)):
    resized_imgs.append(cv2.resize(img,available_size[i]))
ng = []
for i in range(len(resized_imgs)):
    ng.append(utils.get_ng(resized_imgs[i]))


boxes = []
#looping over each ng
for i in range(len(ng)):
#looping over each resized image

    #load the stage II model
    model_II = joblib.load(os.path.join(model_II_path,II_models[i]))
    #sliding over the ng
    scores = np.zeros((ng.shape[0] - 7, ng.shape[1] - 7))
    for y in range(ng.shape[0] - 7):
        for x in range(ng.shape[1] - 7):
            #get the 8*8 ng
            ng_8 = ng[y:y+8,x:x+8]
            #reshape the ng to 1D
            ng_8 = ng_8.reshape(1,-1)
            #predict the ng
            filter_score = model_I.decision_function(ng_8)
            score = model_II.predict(filter_score)
            scores[x,y] = score
    # non-maximum suppression
    nms_index = utils.NMS(scores)
    #saving the position of bounding boxes in original image
    for j in range(len(nms_index)):
        x = nms_index[j][0]
        y = nms_index[j][1]
        n = available_size[i][0]
        m = available_size[i][1]
        x_o = int(w * x / n)
        y_o = int(h * y / m)
        x_op = int(w * (x + 7) / n)
        y_op = int(h * (y + 7) / m)
        boxes.append((x_o,y_o,x_op,y_op,scores[x,y]))


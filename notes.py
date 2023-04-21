# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_classification
# import numpy as np
#
#
# clf_one = make_pipeline(StandardScaler(),
#                         LinearSVC(fit_intercept=False, random_state=0, tol=1e-4))
# x = np.array([[0, 0], [1, 1],[2,2],[3,3]])
# y = np.array([1, 1, -1, -1])
# clf_one.fit(x, y)
# print(clf_one[1].coef_)
# print(clf_one[1].intercept_)
# print(clf_one.predict([[0.1, 0.1]]))
# print(clf_one.decision_function([[0.1, 0.1]]))
#
#
# import joblib
#
# # save
# joblib.dump(clf, "model.pkl")
#
# # load
# clf2 = joblib.load("model.pkl")
#
# clf2.predict(X[0:1])
import random

# import xml.etree.ElementTree as ET
# import os
# anno_path = '/media/zbh/Second/BING/VOC2007/trainval/Annotations'
# anno_list = os.listdir(anno_path)
# count = 0
# for i in range(len(anno_list)):
#     anno_file = os.path.join(anno_path,anno_list[i])
#     tree = ET.parse(anno_file)
#     root = tree.getroot()
#     for obj in root.findall('object'):
#         name = obj.find('name').text
#         if name == 'person':
#             count+=1
#             break
# print(count/len(anno_list))


# import utils
# xmin = 20
# ymin = 20
# xmax = 40
# ymax = 40
# x_o = 30
# y_o = 30
# x_op = 60
# y_op = 60
# iou = utils.compute_iou(int(xmin), int(ymin), int(xmax), int(ymax), int(x_o), int(y_o), int(x_op), int(y_op))
# print(iou)
# print(int(xmin)/int(xmax))

# import numpy as np
# import random
#
# dets = np.zeros((10, 5))
#
# for i in range(10):
#     dets[i, 4] = random.random()
# scores = dets[:, 4]
# order = scores.argsort()[::-1]
# print(scores)
# print(order)
# print(type(order))
# print(order.size)

import numpy as np
import random
score_list=[]
a= np.array([[1,2],[3,4],[5,6],[7,8]])
scores = a[:, 1]
print(scores.shape)
score_list.append(scores)
score_list.append(a[:,0])
score_list = np.concatenate(score_list,axis=0)
print(score_list.reshape(-1,1).shape)
print(35//8)
# print(score_list.shape)
# index = np.where(score_list>3)
# print(index)
# r = random.sample(list(index[0]),2)
# print(r)
# ir = np.concatenate((index[0],r),axis=0)
# print(ir)
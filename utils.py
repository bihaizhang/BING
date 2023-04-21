import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET

def rgb_gradient(img):
    img = img.astype(float)

    h, w, nch = img.shape

    gradientX = np.zeros((h, w))
    gradientY = np.zeros((h, w))

    d1 = np.abs(img[:, 1, :] - img[:, 0, :])
    gradientX[:, 0] = np.max(d1, axis=1) * 2
    d2 = np.abs(img[:, -1, :] - img[:, -2, :])
    gradientX[:, -1] = np.max(d2, axis=1) * 2
    d3 = np.abs(img[:, 2:w, :] - img[:, 0:w - 2, :])
    gradientX[:, 1:w - 1] = np.max(d3, axis=2)

    d1 = np.abs(img[1, :, :] - img[0, :, :])
    gradientY[0, :] = np.max(d1, axis=1) * 2
    d2 = np.abs(img[-1, :, :] - img[-2, :, :])
    gradientY[-1, :] = np.max(d2, axis=1) * 2
    d3 = np.abs(img[2:h, :, :] - img[0:h - 2, :, :])
    gradientY[1:h - 1, :] = np.max(d3, axis=2)

    mag = gradientX+gradientY

    mag[mag < 0] = 0
    mag[mag > 255] = 255
    return mag.astype(np.uint8)
def get_ng(img):
    ng = rgb_gradient(img)
    return ng

#compute the intersection over union
def compute_iou(xmin, ymin, xmax, ymax, x_o, y_o, x_op, y_op):
    #compute the intersection
    x1 = max(xmin, x_o)
    y1 = max(ymin, y_o)
    x2 = min(xmax, x_op)
    y2 = min(ymax, y_op)
    w = max(0, x2-x1)
    h = max(0, y2-y1)
    inter = w*h
    #compute the union
    union = (xmax-xmin)*(ymax-ymin)+(x_op-x_o)*(y_op-y_o)-inter
    iou = inter/union
    return iou

def check_label(anno_path,img_name,x,y,width,height):
    anno_file = os.path.join(anno_path,img_name+'.xml')
    tree = ET.parse(anno_file)
    root = tree.getroot()
    size = root.find('size')
    n = width
    m = height
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    x_o = int(w * x / n)
    y_o = int(h * y / m)
    x_op = int(w * (x + 7) / n)
    y_op = int(h * (y + 7) / m)
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'person':
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            iou = compute_iou(xmin,ymin,xmax,ymax,x_o,y_o,x_op,y_op)
            if iou > 0.5:
                return 1
    return -1

def non_maximum_suppression(filter_scores,labels,iou_th=0.4):
    #scores: each x,y representing the filter score of the 8x8 window at x,y position
    #labels: each x,y representing the label of the 8x8 window at x,y position
    #return: the scores and labels after non maximum suppression
    """Pure Python NMS baseline."""
    candidates = []
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            candidates.append([x,y,filter_scores[x,y],labels[x,y]])
    dets = np.zeros((len(candidates), 6))
    for i in range(len(candidates)):
        dets[i,0] = candidates[i][0]
        dets[i,1] = candidates[i][1]
        dets[i,2] = candidates[i][0]+7
        dets[i,3] = candidates[i][1]+7
        dets[i,4] = candidates[i][2]
        dets[i,5] = candidates[i][3]
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score置信度降序排序
    order = scores.argsort()[::-1]

    keep = []  # 保留的结果框集合, index of the boxes represented in candidates
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        inds = np.where(ovr <= iou_th)[0]
        order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    nms_scores = dets[keep, 4]
    nms_labels = dets[keep, 5]
    return nms_scores,nms_labels

#return the index of scores after non-maximum suppression
def NMS(scores,iou_th=0.4):
    candidates = []
    for x in range(scores.shape[0]):
        for y in range(scores.shape[1]):
            candidates.append([x, y, scores[x, y]])
    dets = np.zeros((len(candidates), 5))
    for i in range(len(candidates)):
        dets[i, 0] = candidates[i][0]
        dets[i, 1] = candidates[i][1]
        dets[i, 2] = candidates[i][0] + 7
        dets[i, 3] = candidates[i][1] + 7
        dets[i, 4] = candidates[i][2]
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score置信度降序排序
    order = scores.argsort()[::-1]

    keep = []  # 保留的结果框集合, index of the boxes represented in candidates
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        inds = np.where(ovr <= iou_th)[0]
        order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    #converting the index of candidates to the index of scores
    index = []
    for i in range(len(keep)):
        index.append((keep[i]//8,keep[i]%8))
    return index





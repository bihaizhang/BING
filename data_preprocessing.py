import os
import numpy as np
import cv2
import utils
import xml.etree.ElementTree as ET
#resizing the original images
height  = [10,20,40,80,160,320]
width = [10,20,40,80,160,320]
# img_path = '/media/zbh/Second/BING/VOC2007/trainval/JPEGImages'
# img_list = os.listdir(img_path)
# for i in range(len(width)):
#     for j in range(len(height)):
#         resize_path = os.path.join("/media/zbh/Second/BING/VOC2007/trainval/resize",str(width[i])+'_'+str(height[j]))
#         if not os.path.exists(resize_path):
#             os.makedirs(resize_path)
#         for k in range(len(img_list)):
#             if img_list[k].endswith('.jpg'):
#                 img = cv2.imread(os.path.join(img_path, img_list[k]))
#                 img = cv2.resize(img, (width[i], height[j]))
#                 cv2.imwrite(os.path.join(resize_path, img_list[k]), img)

#generating the normed gradient features
# ng_path = '/media/zbh/Second/BING/VOC2007/trainval/NG'
# for i in range(len(width)):
#     for j in range(len(height)):
#         save_path = os.path.join("/media/zbh/Second/BING/VOC2007/trainval/NG",str(width[i])+'_'+str(height[j]))
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         resize_path = os.path.join("/media/zbh/Second/BING/VOC2007/trainval/resize",str(width[i])+'_'+str(height[j]))
#         img_list = os.listdir(resize_path)
#         for k in range(len(img_list)):
#             if img_list[k].endswith('.jpg'):
#                 img = cv2.imread(os.path.join(resize_path, img_list[k]))
#                 ng = utils.get_ng(img)
#                 np.save(os.path.join(save_path, img_list[k].split('.')[0]), ng)

#labeling the normed gradient features

'''
looping over all NGs
for each NG, we slide the 8x8 window over it
for each window position, we check if it is a positive or negative sample
if it is a positive sample, we label it as 1, and mark the position of the corresponding original bounding box
if it is a negative sample, we label it as -1
how to save the labels?
each image corresponds to a folder with two sub-folders ("p" and "n"), containing all the 8x8 windows of this image
'''

ng_path = '/media/zbh/Second/BING/VOC2007/trainval/NG'
anno_path = '/media/zbh/Second/BING/VOC2007/trainval/Annotations'
sample_path = '/media/zbh/Second/BING/VOC2007/trainval/sample'
for i in range(len(width)):
    for j in range(len(height)):
        save_path = os.path.join("/media/zbh/Second/BING/VOC2007/trainval/NG",str(width[i])+'_'+str(height[j]))
        ng_list = os.listdir(save_path)
        n = width[i]
        m = height[j]
        for k in range(len(ng_list)):
            if ng_list[k].endswith('.npy'):
                img_name = ng_list[k].split('.')[0]
                # read the annotation file
                anno_file = os.path.join(anno_path, img_name + '.xml')
                tree = ET.parse(anno_file)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)
                ng = np.load(os.path.join(save_path, ng_list[k]))
                #looping over positions
                for y in range(ng.shape[0]-7):
                    for x in range(ng.shape[1]-7):
                        # converting back to original box position
                        x_o = int(w*x/n)
                        y_o = int(h*y/m)
                        x_op = int(w*(x+7)/n)
                        y_op = int(h*(y+7)/m)
                        #check if it is a positive sample
                        flag = 0 #flag=0 means it is a negative sample
                        for obj in root.iter('object'):
                            if obj.find('name').text == 'person':
                                bndbox = obj.find('bndbox')
                                xmin = int(bndbox.find('xmin').text)
                                ymin = int(bndbox.find('ymin').text)
                                xmax = int(bndbox.find('xmax').text)
                                ymax = int(bndbox.find('ymax').text)
                                #compute the intersection over union
                                iou = utils.compute_iou(xmin, ymin, xmax, ymax, x_o, y_o, x_op, y_op)
                                if iou > 0.7:
                                    flag = 1
                                    #save the label
                                    sample_file = os.path.join(sample_path, img_name, 'p', str(x_o)+'_'+str(y_o)+'_'+str(x_op)+"_"+str(y_op)+'.npy')
                                    temp_path = os.path.join(sample_path, img_name, 'p')
                                    if not os.path.exists(temp_path):
                                        os.makedirs(temp_path)
                                    np.save(sample_file, ng[y:y+8, x:x+8])
                                    break
                        if flag == 0:
                            if os.path.exists(os.path.join(sample_path, img_name, 'n')) and len(os.listdir(os.path.join(sample_path, img_name, 'n'))) >= 20:
                                continue
                            else:
                                # save the label
                                sample_file = os.path.join(sample_path, img_name, 'n',
                                                           str(x_o) + '_' + str(y_o) + '_' + str(x_op) + "_" + str(
                                                               y_op) + '.npy')
                                temp_path = os.path.join(sample_path, img_name, 'n')
                                if not os.path.exists(temp_path):
                                    os.makedirs(temp_path)
                                np.save(sample_file, ng[y:y + 8, x:x + 8])



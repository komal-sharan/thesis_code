from __future__ import division
import os
import numpy as np
import cv2
import pickle
import os, h5py, sys, argparse
import numpy as np
import cv2
import os.path
import sklearn.preprocessing
import collections
import pickle
import matplotlib.pyplot as plt
import numpy as np
import model_prediction

dirpath="/home/ksharan1/visualization/san-vqa-tensorflow/vqa-hat/vqahat_train/"

img_path="/home/ksharan1/visualization/san-vqa-tensorflow/vqa-hat/vqahat_train/2773461_1.png"
threshold = 0.003

def convertTobinary_newImage(threshold, img_path):
    print img_path
#     threshold = 0.0025
    x=cv2.imread(img_path)

    avg=np.average(x, axis=2)

#     normalized_array = sklearn.preprocessing.normalize(avg, norm='l1')
    normalized_array = avg / np.sum(avg)

    init_array = np.copy(normalized_array)
    rows,col=normalized_array.shape
    np.set_printoptions(threshold=np.nan)
    for i in range(rows):
        for j in range(col):
            if normalized_array[i][j] > threshold:
                normalized_array[i][j] = 1
            else:
                normalized_array[i][j] = 0

    return normalized_array, init_array, avg


def intersectionOverUnion(original,generated):
    A=np.array(original)
    B=np.array(generated)

    intersection=np.logical_and(A,B)
    union=np.logical_or(A,B)


    difference=((intersection == True).sum())/((union == True).sum())
    return difference

def convertTobinary_generated(mat, threshold):
    print threshold
    avg=np.average(mat, axis=2)
    normalized_array = avg / np.sum(avg)

    init_array = np.copy(normalized_array)
    rows,col=normalized_array.shape
    np.set_printoptions(threshold=np.nan)
    print threshold

    for i in range(rows):
        for j in range(col):
            if normalized_array[i][j] > threshold:
                normalized_array[i][j] = 1
            else:
                normalized_array[i][j] = 0
    return normalized_array, init_array, avg

# threshold = 4.10258372e-06
# z, b, c = convertTobinary_generated(data, 7.10258372e-06)


if __name__ == '__main__':
    modelpath='vqahat_model/san_lstm_att_finetune/model-50000'
    path_for_base='/home/ksharan1/visualization/san-vqa-tensorflow/model/san_lstm_att5/model-75000'
    path_for_new='/home/ksharan1/visualization/san-vqa-tensorflow/vqahat_model/san_lstm_att_multitask_new/model-75000'
    #model_prediction.test("2773461_1.png")

    u=open("gen.pkl",'rb')
    data=pickle.load(u)
    i=open("../train/save_vqahat_test.pkl")
    p=pickle.load(i)
    #model=model_prediction.restoremodel(path_for_base)

    acc7=[]
    acc8=[]
    acc9=[]

    counter=0


    genmaps=open("../san_vis_base_dec1.pkl")
    quesids=open("../san_vis_base_quesid_list_dec1.pkl")
    genmapdata=pickle.load(genmaps)
    quesidlist=pickle.load(quesids)

    for x in range(len(quesidlist)):
        print x

        filename=str(quesidlist[x])+"_1.png"
        filename2=str(quesidlist[x])+"_2.png"
        filename3=str(quesidlist[x])+"_3.png"

        if os.path.isfile(dirpath+filename2):
            filename=filename2


        #a, b, c = convertTobinary_newImage(9e-06, dirpath+filename)
        z7, b, c = convertTobinary_generated(genmapdata[x], 7.10258372e-06)
        #z8, b, c = convertTobinary_generated(genmapdata[x], 8.10258372e-06)
        #z9, b, c = convertTobinary_generated(genmapdata[x], 9.10258372e-06)


        #accuracy7=intersectionOverUnion(a,z7)*100
        #accuracy8=intersectionOverUnion(a,z8)*100
        #accuracy9=intersectionOverUnion(a,z9)*100



        #acc7.append(accuracy7)
        #acc8.append(accuracy8)
        #acc9.append(accuracy9)

    #pickle.dump(acc6,open("visual_accuracy_new_6_today_NOV_corrected.pkl","wb"))
    #pickle.dump(acc7,open("visual_accuracy_new_7_today_NOV_corrected.pkl","wb"))
    #pickle.dump(acc7,open("visual_acc_base_7_ocd.pkl","wb"))
    #pickle.dump(acc8,open("visual_acc_base_8_ocd.pkl","wb"))
    #pickle.dump(acc9,open("visual_acc_base_9_ocd.pkl","wb"))

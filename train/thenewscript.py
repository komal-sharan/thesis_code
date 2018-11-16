import pickle
import os, h5py, sys, argparse
import numpy as np
import cv2
import os.path
import sklearn.preprocessing
listforme=[]
train_data = {}
vqa_data={}
data=[]

input_ques_h5 = '/home/ksharan1/visualization/san-vqa-tensorflow/data_prepro.h5'

pathdir = "/home/ksharan1/visualization/san-vqa-tensorflow/vqa-hat/vqahat_train"
count = 0
path={}
pathlist={}
def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

def get_attention_vqa(name):
    new_array_list=[]
    flag=False
    pathdir = "/home/ksharan1/visualization/san-vqa-tensorflow/vqa-hat/vqahat_train"
    input_path_1 = os.path.join(pathdir,str(name))
    attention_array=cv2.imread(input_path_1).astype(np.float32)
    attention=np.average(attention_array, axis=2)
    new_array=cv2.resize(attention,(7,7))
    new_array = np.reshape(new_array,(49,))

    normalized_array=sklearn.preprocessing.normalize(np.expand_dims(new_array, axis=0), norm='l1')
    #print(sum(normalized_array))
    row,col=normalized_array.shape
    summ=0
    #for i in range(row):
    #    for j in range(col):
    #        summ=summ+normalized_array[i][j]
    #print summ
    print np.sum(normalized_array)
    normalized_array_new=np.reshape(normalized_array,(49,))
    return np.array(normalized_array_new)


def get_data():

    dataset = {}
    train_data = {}
    # load json file

    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_train')
        # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

        tem = hf.get('question_id_train')
        train_data['ques_id'] = np.array(tem)
    print len(train_data['ques_id'])


    for image_path in os.listdir(pathdir):
        question_id=image_path.split('_')[0]
        path[question_id]=image_path

    newdic={}
    newtest={}
    trainVQAdata={}
    vqa=[]
    newdatavqa=[]
    ids=[]
    count=0
    quesids=[]

    for x in range(len(train_data['ques_id'])):
        if str(train_data['ques_id'][x]) in path.keys():
            ids.append(x)
            quesids.append(train_data['ques_id'][x])
            y=train_data['ques_id'][x]
            newdatavqa.append(get_attention_vqa(str(path[str(y)])))
            count=count+1

    #pickle.dump(newdatavqa, open("newdatavqa.pkl", "wb"))

    end=len(newdatavqa)
    print end
    trainend=end-3000
    print trainend
    train_newdatavqa=newdatavqa[:trainend]
    print len(train_newdatavqa)

    trainids=ids[:trainend]
    print len(trainids)



    #VQA HAT TRAIN DATA
    newdic['vqahat']=np.array(train_newdatavqa)
    newdic['question']=train_data['question'][trainids,:]
    newdic['img_list']=train_data['img_list'][trainids]
    newdic['length_q']=train_data['length_q'][trainids]
    newdic['answers']=train_data['answers'][trainids]
    newdic['ques_id']=train_data['ques_id'][trainids]
    print('question aligning')
    newdic['question'] = right_align(newdic['question'], newdic['length_q'])
    print len(newdic['question'])
    pickle.dump(newdic, open("save_vqahat_train1.pkl", "wb"))

    #VQA HAT TEST DATA
    test_newdatavqa=newdatavqa[trainend:]
    testids=ids[trainend:]
    newtest['vqahat']=np.array(test_newdatavqa)
    newtest['question']=train_data['question'][testids,:]
    newtest['img_list']=train_data['img_list'][testids]
    newtest['length_q']=train_data['length_q'][testids]
    newtest['answers']=train_data['answers'][testids]
    newtest['ques_id']=train_data['ques_id'][testids]
    print('question aligning')
    newtest['question'] = right_align(newtest['question'], newtest['length_q'])
    pickle.dump(newtest, open("save_vqahat_test1.pkl", "wb"))

    #VQA TRAIN DATA NOT HAVING VQAHAT TEST Data
    #trainVQAid = [x for x in range(len(train_data['ques_id'])) if x not in ids]

    #trainVQAdata['question']=train_data['question'][trainVQAid,:]
    #trainVQAdata['img_list']=train_data['img_list'][trainVQAid]
    #trainVQAdata['length_q']=train_data['length_q'][trainVQAid]
    #trainVQAdata['answers']=train_data['answers'][trainVQAid]
    #trainVQAdata['ques_id']=train_data['ques_id'][trainVQAid]
    #print('question aligning')
    #trainVQAdata['question'] = right_align(trainVQAdata['question'], trainVQAdata['length_q'])
    #pickle.dump(trainVQAdata, open("vqa_data.pkl", "wb"))


get_data()
#get_attention_vqa("2773341_1.png")

import numpy as np
import cv2
import os, pdb
import caffe
import h5py, json
from scipy.misc import imresize


[IMG_HEIGHT, IMG_WIDTH] = [224, 224]

input_json = 'data_prepro_test.json'
image_root = 'goodata'
# vgg19
cnn_proto = 'vgg19/backup.prototxt'
cnn_model = 'vgg19/VGG_ILSVRC_19_layers.caffemodel'
gpuid = 0
batch_size = 40
out_name = 'data_img_pool5_keeptesting.h5'

def extract_feat(imlist, dname):
    dataLen = len(imlist)

    # vgg19
    f.create_dataset(dname, (dataLen, 512, 7, 7), dtype='f4') # pool5
    #f.create_dataset(dname, (dataLen, 4096), dtype='f4') # fc7
    batch = zip(range(0, dataLen, batch_size), range(batch_size, dataLen+1, batch_size))
    batch.append(((dataLen//batch_size)*batch_size, dataLen))
    count=0
    for start, end in batch:
        batch_image = np.zeros([batch_size, 3, IMG_HEIGHT, IMG_WIDTH])
        batch_imname = imlist[start:end]
        for b in xrange(end-start):
            imname = "data/"+batch_imname[b].encode('utf-8')
            print imname
            x=cv2.imread(imname)
            I = imresize(x, (IMG_HEIGHT, IMG_WIDTH))-mean
            I = np.transpose(I, (2, 0, 1))
            batch_image[b, ...] = I

        net.blobs['data'].data[:] = batch_image
        net.forward()
        # vgg19
        #batch_feat = net.blobs['pool5'].data[...].copy()
        batch_feat = net.blobs['pool5'].data[...].copy()
        f[dname][start:end] = batch_feat[:end-start, ...]




caffe.set_device(gpuid)
caffe.set_mode_gpu()

net = caffe.Net(cnn_proto, cnn_model, caffe.TEST)

mean = np.array((103.939, 116.779, 123.680), dtype=np.float32)

with open(input_json) as data_file:
    data = json.load(data_file)

f = h5py.File(out_name, "w")

extract_feat(data['unique_img_train'], 'images_train')
extract_feat(data['unique_img_test'], 'images_test')

f.close()

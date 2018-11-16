# coding: utf-8

import sys
dataDir = '../../VQA'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqa import VQA
#import matplotlib
from vqaEvaluation.vqaEval import VQAEval
#import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os
import pickle
#matplotlib.use('Agg')

# set up file names and paths
versionType ='' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='train2014'
annFile     = '../Annotations/mscoco_val2014_annotations.json'
quesFile    = '../Questions/OpenEnded_mscoco_val2014_questions.json'
imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
resultType  ='lstm'
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

# An example result json file has been provided in './Results' folder.

[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s%s_%s_%s_%s.json'%(dataDir, versionType, taskType, dataType, \
resultType, fileType) for fileType in fileTypes]
accuracyFile='/home/ksharan1/visualization/san-vqa-tensorflow/VQA/Results/OpenEnded_results_new.json'
resFile='/home/ksharan1/visualization/san-vqa-tensorflow/OpenEnded_mscoco_lstm_results_aftermoving_new.json'
# create vqa object and vqaRes object
print annFile
print quesFile
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)


# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""


vqaEval.evaluate()

# print accuracies
print "\n"
print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
print "Per Question Type Accuracy is the following:"
for quesType in vqaEval.accuracy['perQuestionType']:
	print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "\n"
print "\n\n\n"
print "for statistical ana"
print vqaEval.typevslist

print "\n\n\n"
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
	print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
print "\n"
# demo how to use evalQA to retrieve low score result
evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
if len(evals) > 0:
	print 'ground truth answers'
	randomEval = random.choice(evals)
	randomAnn = vqa.loadQA(randomEval)
	vqa.showQA(randomAnn)

	print '\n'
	print 'generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval])
	ann = vqaRes.loadQA(randomEval)[0]
	print "Answer:   %s\n" %(ann['answer'])

	imgId = randomAnn[0]['image_id']
	imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
	if os.path.isfile(imgDir + imgFilename):
		#I = io.imread(imgDir + imgFilename)
		#plt.imshow(I)
		#plt.axis('off')
		#plt.show()
		print "reaches here"
# plot accuracy for various question types


#plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
#plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='0',fontsize=10)
#plt.title('Per Question Type Accuracy', fontsize=10)
#plt.xlabel('Question Types', fontsize=10)
#plt.ylabel('Accuracy', fontsize=10)
#plt.show()

# save evaluation results to ./Results folder
pickle.dump(vqaEval.typevslist, open("stastiscal_for_model_new.pkl",'w'))
json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))

import pickle
from numpy.random import normal
import random
res1=open("stastiscal_for_model_base.pkl")
res2=open("stastiscal_for_model_new.pkl")
results1=pickle.load(res1)
results2=pickle.load(res2)
#print results
category_list=['what room is','what number is','what sport is','what is the person','is there a','is this an','where are the','what time','how many people are in','do you',]
for x in category_list:
    res1list=results1[x]
    res2list=results2[x]
    a = res1list
    b = res2list
    print a
    print b
    diff = sum(x - y for x, y in zip(a, b))
    n = 100000
    dist = []
    for _ in range(n):
        d = 0
        for x, y in zip(a, b):
            if random.uniform(0, 1) < 0.5:
                d += x - y
            else:
                d += y - x
            dist.append(d)
    

    print('Original difference is larger than %.4f%% of samples' % (sum(map(lambda x: diff > x, dist)) * 100 / n))

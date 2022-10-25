import pandas as pd
import numpy as np
import sys
import pandas as pd

filename = sys.argv[1]
out = sys.argv[2]

print('reading in data')
dataset =pd.read_csv(filename)

print('dataset size: '+str(len(dataset)))

allinf = np.where(np.all(dataset.iloc[:,8:]==-np.inf,axis=1))[0]
dataset = dataset.drop(allinf)

print('dataset size: '+str(len(dataset)))

dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

print('Replaced infinities with nan in dataset')

print('Dividing training and test data')

train_dataset = dataset.sample(frac=0.9, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.iloc[:,:8]
train_labels = train_dataset.iloc[:,8:]

test_features = test_dataset.iloc[:,:8]
test_labels = test_dataset.iloc[:,8:]

print('saving csvs')
train_features.to_csv(out+'_trainfeatures.csv',index=False)
train_labels.to_csv(out+'_trainlabels.csv',index=False,na_rep=np.nan)
test_features.to_csv(out+'_testfeatures.csv',index=False)
test_labels.to_csv(out+'_testlabels.csv',index=False,na_rep=np.nan)

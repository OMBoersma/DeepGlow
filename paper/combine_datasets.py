import sys
import pandas as pd

nfiles = len(sys.argv[1:])
files = sys.argv[1:-1]
out = sys.argv[-1]
total = []

for i,filename in enumerate(files):
    print('reading in file '+str(i))
    ext = filename[-3:]
    if ext =='hdf':
        filedata = pd.read_hdf(filename,key='data')        
    else:
        filedata = pd.read_csv(filename)
    total.append(filedata)
dataset = pd.concat(total,ignore_index=True)
dataset.to_csv(out+'_total.csv',index=False)




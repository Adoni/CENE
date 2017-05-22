import sys
import numpy
fname=sys.argv[1]
f=open(fname)
f.readline()
eids=[]
es=[]
indexes=dict()
for line in f:
    line=line.strip().split(' ')
    index=len(eids)
    indexes[line[0]]=index
    eids.append(line[0])
    es.append(map(lambda x:float(x), line[1:]))
es=numpy.array(es)

while 1:
    a=raw_input('Word: ')
    try:
        index=indexes[a]
    except:
        print('No this word')
        continue
    d = numpy.linalg.norm(es - es[index], axis=1)
    d=zip(eids, d)
    d=sorted(d, key=lambda x:x[1])
    print('\n')
    for dd in d[:10]:
        print('%s   %f'%dd)
    print('\n\n\n')

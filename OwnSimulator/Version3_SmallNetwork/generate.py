import random
random.seed()

n=10
nVehAtInst = 400
timeForOneTrip = 255
i = 0
while i<nVehAtInst:
    x = random.randint(1,n)
    y = random.randint(1,n)
    if(x!=y):
        print 0,x,y
        i+=1
ns = 3600 #no. of seconds
remainder = nVehAtInst % timeForOneTrip
for i in xrange(ns):
    tmp = random.randint(0,timeForOneTrip)
    #nv=no of vehicles per second : 1 with prob 1/3 and 2 with prob 2/3
    nv = int(nVehAtInst/timeForOneTrip)
    if(tmp <= remainder):
        nv +=1
    j = 1
    while(j<=nv):
        x = random.randint(1,n)
        y = random.randint(1,n)
        if(x!=y):
            print i,x,y
            j+=1

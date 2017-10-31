from __future__ import print_function 
import matplotlib.pyplot as plt
import heapq as hp
import math
import numpy as np
import random 

n = 36 #No. of nodes 
nIter = 100
nEdges = 60
threshold = 1800
links = []
edges = []
drivers = []
adjList = [[] for i in xrange(n+1)]
edgeCount = [[0 for i in xrange(n+1)] for i in xrange(n+1)]
freeTime = 20

'''Hyper Parameters'''
beta = float(90)/float(130)
epsilonG,alphaG,RG = 1,0.9,0.9
tollAtFreeTime = 40
''' '''

def VOT(t):
	return 100/(1+math.exp(-0.07*(t-40)))



class LinkManager(object):
	epsilon,alpha,R = 1,0.9,0.9
	initialQval = 125000
	def __init__(self,a,b,time=20):
		self.a = a
		self.b = b
		self.time = time
		self.toll = random.randint(1,10)*10
		self.qVal = np.array([self.initialQval for i in xrange(11)])

	def chooseToll(self):
		# print("Choosing toll with epsilon as ", self.epsilon)
		x = random.uniform(0, 1)
		if(x < self.epsilon):
			self.toll = random.randint(1,10)*10
		else :
			self.toll = np.argmax(self.qVal)*10
		# print("New toll value is ",self.toll)

	def calCost(self,p):
		vot = VOT(self.time)
		return p*vot + (1-p)*self.toll

class Vehicle:
	def __init__(self,time,originx,originy,destx,desty):
		self.time = time
		self.origin = 6*(originy-1) + originx 
		self.dest = 6*(desty-1) + destx 
		self.p = self.assignP()

	def assignP(self):
		return 0.5

	def shortestPath(self):
		# print(self.origin,self.dest)
		level = [0 for i in xrange(37)]
		h,cost,timeCost,tollCost = [],0,0,0
		cur = self.origin
		while(cur!=self.dest):
			links = adjList[cur]
			for link in links:
				if(level[link.b]==0):
					linkCost = link.calCost(self.p)
					hp.heappush(h,[cost+linkCost, link.b, timeCost+link.time, tollCost+link.toll])
					level[link.b]=cur
			tmp = hp.heappop(h)
			cost,timeCost,tollCost = tmp[0],tmp[2],tmp[3]
			cur = tmp[1]
		return int(timeCost)


f1 = open('edges','r')
for line in f1 :
	words = line.split()
	link = LinkManager(int(words[0]),int(words[1]))
	links.append(link)
	adjList[int(words[0])].append(link)
	edges.append([int(words[0]),int(words[1])])
f1.close()

f1 = open('load','r')
totalVeh = 0
for line in f1 :
	words = line.split()
	for i,x in enumerate(words):
		words[i]=int(words[i])
	tmp = Vehicle(words[0],words[1],words[2],words[3],words[4])
	drivers.append(tmp)
	totalVeh += 1 
f1.close()

vehCount = [0 for i in xrange(4000)]
for driver in drivers :
	vehCount[driver.time]+=1
	endTime = driver.shortestPath()
	vehCount[driver.time+endTime]-=1

totalCnt = []
currentCnt = 0
for i in xrange(4000):
	currentCnt += vehCount[i]
	totalCnt.append(currentCnt)

plt.plot(totalCnt)
plt.title("No. of vehicles at any time")
plt.show()
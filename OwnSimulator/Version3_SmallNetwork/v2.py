from __future__ import print_function 
import heapq as hp
import math
import numpy as np
import random 

n = 36 #No. of nodes 
nIter = 200
nEdges = 60
threshold = 1800
links = []
edges = []
drivers = []
adjList = [[] for i in xrange(n+1)]
edgeCount = [[0 for i in xrange(n+1)] for i in xrange(n+1)]

def VOT(t):
	return 100/(1+math.exp(-0.07*(t-40)))

def calcVariance(meanVal):
	total = 0 
	for link in links : 
		nVeh = edgeCount[link.a][link.b]
		total += (nVeh-meanVal)**2 ;
	return total/nEdges 

def updateTime():
	for link in links : 
		nVeh = edgeCount[link.a][link.b]
		# print("from ",link.a," to ",link.b," we have ",nVeh," vehicles.")
		# print("old time was ",link.time,end=" ")
		extra = max(0,nVeh-threshold)
		if(extra < 1800):
			link.time = 20 + 3.33*(math.exp((float(extra)/900)) - 1)
		elif extra < 2700 :
			link.time = 40 + 11.11*(math.exp((float(extra-1800)/900)) - 1)
		else :
			link.time = 60 + 32.1*(math.log((float(extra-2700)/1800)+1))
		link.time = int(link.time)
		# print("and new time is ",link.time)
	return 

# def updateTime():
# 	for link in links :
# 		nVeh = edgeCount[link.a][link.b]
# 		# print("from ",link.a," to ",link.b," we have ",nVeh," vehicles.")
# 		extra = max(0,nVeh-threshold)
# 		# print("old time was ",link.time,end=" ")
# 		link.time = 20 + extra*130/(totalVeh-threshold)
# 		# print("and new time is ",link.time)

def updateToll():
	for link in links : 
		nVeh = edgeCount[link.a][link.b]
		# print("from ",link.a," to ",link.b," we have ",nVeh," vehicles.")
		reward = nVeh*link.toll
		# print("Old reward for this link is ",link.qVal[link.toll/10],end=" ")
		link.qVal[link.toll/10] = (1-link.alpha)*link.qVal[link.toll/10] + (link.alpha)*reward
		link.updateAlpha()
		# print("and new reward is ",link.qVal[link.toll/10])
		link.chooseToll() 


class LinkManager(object):
	epsilon,alpha = 1,0.9
	initialQval = 125000
	@classmethod
	def updateEpsilon(cls):
		e0,ef = 1,0.01
		multiplier = (ef/e0)**(float(1)/max(nIter,1000)) 
		cls.epsilon *= multiplier

	@classmethod
	def updateAlpha(cls):
		a0,af = 0.9,0.9	
		multiplier = (af/a0)**(float(1)/max(nIter,1000))  #used max(nIter,100) as of now during debugging 
		cls.alpha *= multiplier

	def __init__(self,a,b,time=20,toll=50):
		self.a = a
		self.b = b
		self.time = time
		self.toll = toll
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
		# return 0.5
		return random.uniform(0, 1)	

	def shortestPath(self):
		# print(self.origin,self.dest)
		level = [0 for i in xrange(37)]
		h,cost = [],0
		cur = self.origin
		while(cur!=self.dest):
			links = adjList[cur]
			for link in links:
				if(level[link.b]==0):
					linkCost = link.calCost(self.p)
					hp.heappush(h,[cost+linkCost, link.b])
					level[link.b]=cur
			tmp = hp.heappop(h)
			cost = tmp[0]
			cur = tmp[1]
		pathCnt = 0	
		while(cur!=self.origin):
			edgeCount[level[cur]][cur]+=1
			cur = level[cur]
			pathCnt+=1
		return pathCnt

""" This is to print the edgeCount matrix for the edges in the graph """
def printEdgeCnt():
	total,cnt = 0,0
	for edge in edges :
		total += edgeCount[edge[0]][edge[1]]
		print(edgeCount[edge[0]][edge[1]],end=" ")
		cnt+=1
		if(cnt==5):
			print()
			cnt = 0
	return total
""" ends """


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

for i in xrange(nIter):
	# edgeCount = [[-edgeCount[i][j] for j in xrange(n+1)] for i in xrange(n+1)]
	# edgeCount = [[0 for j in xrange(n+1)] for i in xrange(n+1)]
	edgeCount =  np.zeros((n+1,n+1),dtype=int)
	dist = 0
	for driver in drivers : 
		dist += driver.shortestPath()
	printEdgeCnt()
	print("Total distance covered by drivers is : ",dist," and thus the meanValue is ",dist/nEdges)
	print(calcVariance(dist/nEdges))
	updateTime()
	updateToll()
	LinkManager.updateEpsilon()

# for link in links : 
# 	print(edgeCount[link.a][link.b],link.time)
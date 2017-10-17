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

beta = float(90)/float(130)
freeTime = 20

def VOT(t):
	return 100/(1+math.exp(-0.07*(t-40)))

def calcVariance(meanVal):
	total = 0 
	for link in links : 
		nVeh = edgeCount[link.a][link.b]
		total += (nVeh-meanVal)**2 ;
	return total/nEdges 

# def updateTime():
# 	for link in links : 
# 		nVeh = edgeCount[link.a][link.b]
# 		# print("from ",link.a," to ",link.b," we have ",nVeh," vehicles.")
# 		# print("old time was ",link.time,end=" ")
# 		extra = max(0,nVeh-threshold)
# 		if(extra < 1800):
# 			link.time = 20 + 3.33*(math.exp((float(extra)/900)) - 1)
# 		elif extra < 2700 :
# 			link.time = 40 + 11.11*(math.exp((float(extra-1800)/900)) - 1)
# 		else :
# 			link.time = 60 + 32.1*(math.log((float(extra-2700)/1800)+1))
# 		link.time = int(link.time)
# 		# print("and new time is ",link.time)
# 	return 

def updateTime():
	for link in links :
		nVeh = edgeCount[link.a][link.b]
		# print("from ",link.a," to ",link.b," we have ",nVeh," vehicles.")
		extra = max(0,nVeh-threshold)
		# print("old time was ",link.time,end=" ")
		link.time = 20 + extra*130/(totalVeh-threshold)
		# print("and new time is ",link.time)

# def updateToll():
# 	for link in links : 
# 		nVeh = edgeCount[link.a][link.b]
# 		# print("from ",link.a," to ",link.b," we have ",nVeh," vehicles.")
# 		reward = nVeh*link.toll
# 		# print("Old reward for this link is ",link.qVal[link.toll/10],end=" ")
# 		link.qVal[link.toll/10] = (1-link.alpha)*link.qVal[link.toll/10] + (link.alpha)*reward
# 		link.updateAlpha()
# 		# print("and new reward is ",link.qVal[link.toll/10])
# 		link.chooseToll() 

def updateToll():
	for link in links : 
		delta = beta*(link.time-freeTime) + 10
		link.toll = link.R*delta + (1-link.R)*link.toll

class LinkManager(object):
	epsilon,alpha,R = 1,0.9,0.9
	initialQval = 125000
	@classmethod
	def updateEpsilon(cls):
		e0,ef = 1,0.01
		multiplier = (ef/e0)**(float(1)/max(nIter,1000)) 
		cls.epsilon *= multiplier

	@classmethod
	def updateAlpha(cls):
		a0,af = 0.9,0.4	
		multiplier = (af/a0)**(float(1)/max(nIter,1000))  #used max(nIter,100) as of now during debugging 
		cls.alpha *= multiplier

	@classmethod
	def updateR(cls):
		r0,rf = 0.9,0.3	
		multiplier = (rf/r0)**(float(1)/max(nIter,1000))  #used max(nIter,100) as of now during debugging 
		cls.R *= multiplier


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
		pathCnt = 0
		while(cur!=self.origin):
			edgeCount[level[cur]][cur]+=1
			cur = level[cur]
			pathCnt+=1
		return pathCnt,cost,timeCost,tollCost

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

def totalRevenue():
	revenues = []
	for link in links : 
		revenues.append(edgeCount[link.a][link.b]*link.toll)
	revenues = np.array(revenues)
	return np.sum(revenues),np.mean(revenues),np.var(revenues)


def printMetrics(dist,totalcost,timecost,tollcost):
	print("Total Distance : ",dist,"\t Mean Distance : ",dist/nEdges)
	print("Variance in no. of vehicles/link : ",calcVariance(dist/nEdges))
	print("Total cost : ",totalcost,"\t Mean Cost : ",totalcost/totalVeh)
	print("Total time : ",timecost,"\t Mean Time : ",timecost/totalVeh)
	print("Total toll : ",tollcost,"\t Mean Toll : ",tollcost/totalVeh)
	a,b,c = totalRevenue()
	print("Total revenue : ", a, "Mean revenue : ", b, "Variance revenue : ", c)	
	return 

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
	dist,totalcost,timecost,tollcost = 0,0,0,0
	for driver in drivers : 
		 a,b,c,d = driver.shortestPath()
		 dist+=a 
		 totalcost+=b 
		 timecost+=c 
		 tollcost+=d
	printEdgeCnt()
	printMetrics(dist,totalcost,timecost,tollcost)
	# print("Total distance covered by drivers is : ",dist," and thus the meanValue is ",dist/nEdges)
	# print(calcVariance(dist/nEdges))
	updateTime()
	updateToll()			
	LinkManager.updateEpsilon()
	LinkManager.updateR()

# for link in links : 
# 	print(edgeCount[link.a][link.b],link.time)
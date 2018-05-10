from __future__ import print_function 
import matplotlib.pyplot as plt
import heapq as hp
import math
import numpy as np
import random 

n = 16 #No. of nodes 
nIter = 500
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
	return (1000/(1+math.exp(-0.001*(t-200)))) - 450

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
		# print("old time was ",link.time,end=" ")
		if(nVeh < 500):
			link.time = 20 + (80*nVeh/500)
		else : 
			link.time = (nVeh**2)/5000 + 30
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

# def updateToll():
# 	for link in links : 
# 		delta = beta*(link.time-freeTime) + tollAtFreeTime
# 		link.toll = link.R*delta + (1-link.R)*link.toll
# 		link.updateR()

class LinkManager(object):
	epsilon,alpha,R = epsilonG,alphaG,RG 
	initialQval = 0
	@classmethod
	def updateEpsilon(cls):
		e0,ef = 1,0.01
		multiplier = (ef/e0)**(float(1)/nIter) 
		cls.epsilon *= multiplier

	@classmethod
	def updateAlpha(cls):
		a0,af = 0.9,0.4	
		multiplier = (af/a0)**(float(1)/nIter)  #used max(nIter,100) as of now during debugging 
		cls.alpha *= multiplier

	@classmethod
	def updateR(cls):
		r0,rf = 0.9,0.3	
		multiplier = (rf/r0)**(float(1)/nIter)  #used max(nIter,100) as of now during debugging 
		cls.R *= multiplier


	def __init__(self,a,b,time=200):
		self.a = a
		self.b = b
		self.time = time
		self.toll = random.randint(1,10)*10
		# self.toll = 70
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
	def __init__(self,time,origin,dest):
		self.time = time
		self.origin = origin
		self.dest = dest 
		self.p = self.assignP()

	def assignP(self):
		# return random.uniform(0,1)
		return 0.5

	def shortestPath(self):
		# print(self.origin,self.dest)
		level = [0 for i in xrange(n+1)]
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
		return np.array([pathCnt,cost,timeCost,tollCost])

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
	revenues,tolls = [],[]
	for link in links : 
		revenues.append(edgeCount[link.a][link.b]*link.toll)
		tolls.append(link.toll)
	revenues,tolls = np.array(revenues),np.array(tolls)
	return tolls,np.sum(revenues),np.mean(revenues),np.var(revenues)

varVeh,totCost,tolCost,timCost,totRev,meanTol = [],[],[],[],[],[]
def printMetrics(metrics):
	dist,totalcost,timecost,tollcost = metrics[0],metrics[1],metrics[2],metrics[3]
	print("Total Distance : ",dist,"\t Mean Distance : ",dist/nEdges, "\t Mean distance/vehicle : ",dist/totalVeh)
	print("Variance in no. of vehicles/link : ",calcVariance(dist/nEdges))
	print("Total cost : ",totalcost,"\t Mean Cost : ",totalcost/totalVeh)
	print("Total time : ",timecost,"\t Mean Time : ",timecost/totalVeh)
	print("Total toll : ",tollcost,"\t Mean Toll : ",tollcost/totalVeh)
	t,a,b,c = totalRevenue()
	varVeh.append(calcVariance(dist/nEdges))
	totCost.append(totalcost)
	tolCost.append(tollcost)
	timCost.append(timecost)
	totRev.append(a)
	meanTol.append(np.mean(t))
	print("Total revenue : ", a, "Mean revenue : ", b, "Variance revenue : ", c)
	# print("Tolls for all agents are : ",t)
	print("Mean Toll : ",np.mean(t))
	return 

def showPlot(inp,titleName):
	fig = plt.figure()
	plt.plot(inp)
	plt.title(titleName)
	# plt.show()
	fig.savefig("./Graphs/Version1/v1_"+str(titleName)+"_"+str(nIter)+".png")
	plt.close(fig)

def insertLink(a,b):
	link = LinkManager(a,b)
	links.append(link)
	adjList[a].append(link)
	edges.append([a,b])

f1 = open('edges','r')
for line in f1 :
	words = line.split()
	insertLink(int(words[0]),int(words[1]))
	insertLink(int(words[1]),int(words[0]))	
f1.close()

f1 = open('load','r')
totalVeh = 0
for line in f1 :
	words = line.split()
	for i,x in enumerate(words):
		words[i]=int(words[i])
	tmp = Vehicle(words[0],words[1],words[2])
	drivers.append(tmp)
	totalVeh += 1 
f1.close()

for i in xrange(nIter):
	# edgeCount = [[-edgeCount[i][j] for j in xrange(n+1)] for i in xrange(n+1)]
	# edgeCount = [[0 for j in xrange(n+1)] for i in xrange(n+1)]
	edgeCount =  np.zeros((n+1,n+1),dtype=int)
	# dist,totalcost,timecost,tollcost = 0,0,0,0
	metrics = np.zeros(4,dtype=int)
	for driver in drivers : 
		 tmp = driver.shortestPath()
		 metrics = metrics+tmp
	# printEdgeCnt()
	if i!=0 : 
		printMetrics(metrics)
	# print("Total distance covered by drivers is : ",dist," and thus the meanValue is ",dist/nEdges)
	# print(calcVariance(dist/nEdges))
	updateTime()
	updateToll()
	for link in links : 
		print(np.argmax(link.qVal), end=" ")
	print()
	for link in links : 
		print(link.a, link.b, link.toll, sep=" ")
	LinkManager.updateEpsilon()
	LinkManager.updateR()

showPlot(varVeh,"Variance in vehicle per link")
showPlot(totCost,"Total Cost incurred by Drivers")
showPlot(tolCost,"Total Toll Cost given by Drivers")
showPlot(timCost,"Total Time taken by all Drivers")
showPlot(totRev,"Total revenue generated")
showPlot(meanTol,"Mean Toll value of each link")
# for link in links : 
# 	print(edgeCount[link.a][link.b],link.time)
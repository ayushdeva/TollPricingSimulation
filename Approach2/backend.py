from __future__ import print_function 
import matplotlib.pyplot as plt
import heapq as hp
import math
import numpy as np
import random 

# threshold = 1800
# edges = []
# edgeCount = [[0 for i in xrange(n+1)] for i in xrange(n+1)]
# freeTime = 20

# '''Hyper Parameters'''
# beta = float(90)/float(130)
# epsilonG,alphaG,RG = 1,0.9,0.9
# tollAtFreeTime = 40
# ''' '''

class Simulator(object):
	
	adjList = [[] for i in xrange(37)] #Adjacency List of the Network. Will scan it later from the load file.
	n = 36 #no. of nodes
	nIter = 200 # no. of iterations
	nEdges = 60 # no. of edges (uni-directional)
	nVehicles = 0 #will update when we loadVehicles()
	freeTime = 20
	links = {} # key : "<origin>to<destination>" ; value : corresponding LinkManager object
	drivers = [] # list of all Vehicle objects
	totalTime = [] # totalTime taken by all vehicles to complete their trips in one iteration
	varianceInSpeed = [] # variance in avg. speed of all vehicles in each iteration
	varianceVehDes = [] # variance in vehicle density on each edge in each iteration
	totalRevenue = [] # total revenue collected in each iteration
	tollVariance = [] # variance in toll values of each edge per iteration. 
	totalTripsCompleted = []

	beta = 4

	epsilon = 1 
	@classmethod
	def updateEpsilon(cls):
		e0,ef = 1,0.01
		multiplier = (ef/e0)**(float(1)/cls.nIter) 
		cls.epsilon *= multiplier

	alpha = 0.9 
	@classmethod
	def updateAlpha(cls):
		a0,af = 0.9,0.4	
		multiplier = (af/a0)**(float(1)/cls.nIter)  #used max(nIter,100) as of now during debugging 
		cls.alpha *= multiplier

	R = 0.9 
	@classmethod
	def updateR(cls):
		r0,rf = 0.9,0.3	
		multiplier = (rf/r0)**(float(1)/cls.nIter)  #used max(nIter,100) as of now during debugging 
		cls.R *= multiplier

	@classmethod
	def VOT(cls,t):
		return (1000/(1+math.exp(-0.001*(t-200)))) - 450

	def __init__(self):
		pass

	def initNetwork(self):
		self.loadNetwork() 
		self.loadVehicles()	

	def loadNetwork(self):
		f1 = open('edges','r')
		for line in f1 :
			words = line.split()
			link = LinkManager(int(words[0]),int(words[1]))
			linkKey = words[0]+"to"+words[1]
			self.links[linkKey] = link
			self.adjList[int(words[0])].append(link)
		f1.close()

	def loadVehicles(self):
		f1 = open('load','r')
		for line in f1 :
			words = line.split()
			for i,x in enumerate(words):
				words[i]=int(words[i])
			veh = Vehicle(words[0],words[1],words[2],words[3],words[4])
			self.drivers.append(veh)
			self.nVehicles += 1
		f1.close()

	@classmethod
	def showPlot(cls,inp,titleName):
		fig = plt.figure()
		plt.plot(inp)
		plt.title(titleName)
		# plt.show()
		fig.savefig("./Graphs/"+str(titleName)+".png")
		plt.close(fig)

	@classmethod
	def plotStats(cls):
		cls.showPlot(cls.totalTripsCompleted,"Total Trips Completed")
		cls.showPlot(cls.totalTime,"Total Time Taken For All Trips")
		cls.showPlot(cls.varianceInSpeed,"Variance in avg speed of each driver")
		cls.showPlot(cls.varianceVehDes,"Variance in vehicular density on each edge")
		cls.showPlot(cls.totalRevenue,"Total Revenue Collected")
		cls.showPlot(cls.tollVariance,"Variance in all Toll Prices")

class LinkManager(Simulator):
	initialQval = 0 #the initial qValue for each toll value (to be the same)
	intitalTime = 1 #the initial knowledge of time taken to cross the road	
	
	def __init__(self,a,b):
		self.endA = a
		self.endB = b
		self.time = self.intitalTime
		self.toll = random.uniform(0,100)
		self.qVal = np.array([self.initialQval for i in xrange(11)])

	def chooseToll(self):
		delta = self.beta*(self.time-self.freeTime)
		self.toll = self.R*delta + (1-self.R)*self.toll

	def updateState(self,rev,newTime):
		self.time = newTime
		# self.qVal[self.toll/10] = (1-self.alpha)*self.qVal[self.toll/10] + (self.alpha)*rev
		self.chooseToll()

	def calCost(self,p):
		vot = self.VOT(self.time)
		return p*vot + (1-p)*self.toll

class Vehicle(Simulator):
	
	def __init__(self,time,originx,originy,destx,desty):
		# Simulator.__init__(self)
		self.time = time
		self.origin = 6*(originy-1) + originx 
		self.dest = 6*(desty-1) + destx 
		self.p = self.assignP()
		self.route = []

	def assignP(self):
		# return random.uniform(0,1)
		return 0.5

	def shortestPath(self):
		# print(self.origin,self.dest)
		level = [0 for i in xrange(37)]
		h,cost,timeCost,tollCost = [],0,0,0
		cur = self.origin
		while(cur!=self.dest):
			adjEdges = self.adjList[cur] #adjacent Edges from cur node
			for link in adjEdges:
				if(level[link.endB]==0):
					linkCost = link.calCost(self.p)
					hp.heappush(h,[cost+linkCost, link.endB, timeCost+link.time, tollCost+link.toll])
					#Redundancy : timeCost and tollCost may be reduntant. Not removing now, but for future. 
					level[link.endB]=cur
			# print(h)
			tmp = hp.heappop(h)
			cost,timeCost,tollCost = tmp[0],tmp[2],tmp[3]
			cur = tmp[1]
		path = []
		while(cur!=self.origin):
			# edgeCount[level[cur]][cur]+=1
			path.append(cur)
			cur = level[cur]
		path.append(self.origin)
		path = np.flip(np.array(path),0) #reverse the path array - in the order of nodes visited
		self.route = path
		return path


# sim = Simulator()
# print(sim.nVehicles)
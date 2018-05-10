from __future__ import print_function
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import sys
reload(sys)

sys.setdefaultencoding('ascii')
freeFlowTime = 0

def updateMaxTime():
	doc = minidom.parse("./data/edgeInfo")
	maxTime = float((doc.getElementsByTagName("interval")[0]).getAttribute("end"))
	maxTime = int(maxTime)+10
	return maxTime

def VehPerInst(endTime):
	arrDep = [0 for i in xrange(endTime)] # +=1 for each departure (new trip) and -=1 for each arrival (end trip)
	print(len(arrDep))
	doc = minidom.parse("tripinfo.xml")
	vehs = doc.getElementsByTagName("tripinfo")
	nVeh = len(vehs)
	time = np.zeros(nVeh)
	minTime = 1e5
	for i,veh in enumerate(vehs) :
		depart = int(float(veh.getAttribute("depart")))
		arrival = int(float(veh.getAttribute("arrival")))
		time[i] = arrival - depart
		minTime = min(minTime,time[i])
		# if(arrival > maxTime)
		arrDep[depart] += 1
		arrDep[arrival] -= 1
	avgTime = np.sum(time)/nVeh
	print("Average Time for a vehicle is", avgTime)
	vehiclesAtInst = [0 for i in xrange(endTime)]
	curr = 0 
	print("at t=0 : ",arrDep[0])
	for i in xrange(endTime):
		curr += arrDep[i]
		vehiclesAtInst[i] = curr
	plt.plot(vehiclesAtInst)
	plt.title("No. of vehicles at each instance")
	plt.show()
	print(minTime)

def calcVariance(arr):
	arr = np.array(arr)
	return np.var(arr)

def VehStats(glTotTime,glVarSpeed,glTotTrips):
	doc = minidom.parse("tripinfo.xml")
	vehs = doc.getElementsByTagName("tripinfo")
	totalTime, avgSpeed = 0,[]
	nVeh = len(vehs)
	for veh in vehs : 
		totalTime += float(veh.getAttribute("duration"))
		avgSpeed.append(float(veh.getAttribute("routeLength"))/float(veh.getAttribute("duration")))
	glTotTime.append(totalTime)
	glVarSpeed.append(calcVariance(avgSpeed))
	glTotTrips.append(nVeh)

def EdgeStats(edgeObj,vehDen,totRev,tollVar):
	# parse extraInfo.
	# update time of each edge
	# update toll of each edge
	doc = minidom.parse("./data/edgeInfo")
	edges = doc.getElementsByTagName("edge")
	vehDensity,tolls,totalRev = [],[],0
	# print(len(edges))
	# print("Edge\tNVeh\tRevenue\tMeanTime")
	for edge in edges : 
		edgeId = str(edge.getAttribute("id"))
		# print(edge.getAttribute("traveltime"))
		newTime = edge.getAttribute("traveltime")
		if(str(newTime)==""):
			newTime = freeFlowTime
		# print(edge.getAttribute("arrived"), type(str(edge.getAttribute("arrived"))))
		totalVehs = int(edge.getAttribute("left")) + int(edge.getAttribute("arrived"))
		vehDensity.append(totalVehs)
		tolls.append(edgeObj[edgeId].toll)
		revenue = edgeObj[edgeId].toll * totalVehs
		totalRev += revenue
		# print(edgeId+"\t"+str(totalVehs)+"\t"+str(revenue)+"\t"+str(newTime))
		edgeObj[edgeId].updateState(float(revenue),float(newTime))
	totRev.append(totalRev)
	vehDen.append(np.var(vehDensity))
	tollVar.append(np.var(tolls))
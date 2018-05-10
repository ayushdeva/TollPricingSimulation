#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from backend import *
from stats import *

import os
import sys
import optparse
import subprocess
import heapq as hp
import random
from xml.dom import minidom

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", '..', "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci 

def conv2string(route):
    #route is a list of nodes traversed.
    #eg [1,2,3,4,5,6] ---> "1to2 2to3 3to4 4to5 5to6"
    temp_str = ""
    for i, val in enumerate(route[:-1]):
        temp_str += str(val) + "to" + str(route[i+1]) + " "
    return temp_str

def generate_routefile(trips):
    with open("data/cross.rou.xml", "w") as netRoutes:
        print("""<routes>
                 <vType id="generic" accel="0.8" decel="4.5" sigma="0.5" length="3" minGap="2.1" maxSpeed="20" guiShape="passenger"/>""", file=netRoutes)
        vid = 1
        for trip in trips : 
        	path = conv2string(trip.route)
        	print("""<route id="""+'"'+str(vid)+'"' + """ edges=""" + '"' + path + '"' + """/>""", file=netRoutes)
        	print("""<vehicle id="""+'"'+str(vid)+'"'+""" type="generic" route=""" + '"' + str(vid) + '"' + """ depart="""+'"'+str(trip.time)+'"'+ """ color="1,0,0"/>""", file=netRoutes)
        	vid+=1
        print("</routes>", file=netRoutes)
        # printCustom(globalCount)
    # sys.exit(0)        

def run():
    """execute the TraCI control loop"""
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("1", 2)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        # if traci.trafficlights.getPhase("1") == 2:
            # # we are not already switching
            # if traci.inductionloop.getLastStepVehicleNumber("1") > 0:
            #     # there is a vehicle from the north, switch
            #     traci.trafficlights.setPhase("1", 3)
            # else:
            #     # otherwise try to keep green for EW
            #     traci.trafficlights.setPhase("1", 2)
        step += 1
        if(step==3600):
            break
    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    # sumoBinary = checkBinary('sumo-gui')
    sumoBinary = checkBinary('sumo')
    sim = Simulator()
    sim.initNetwork()
    for i in xrange(sim.nIter):
        print("Iteration No. : ",i+1) 
    	for trip in sim.drivers :
    		trip.shortestPath()
    	generate_routefile(sim.drivers)
        traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                                 "--tripinfo-output", "tripinfo.xml", ])
        run()
        # if i==0 :
            # VehPerInst()
        endTime = updateMaxTime()
        # VehPerInst(endTime)
        VehStats(Simulator.totalTime,Simulator.varianceInSpeed,Simulator.totalTripsCompleted)
        EdgeStats(sim.links,Simulator.varianceVehDes,Simulator.totalRevenue,Simulator.tollVariance)
        # Simulator.updateEpsilon()
        # Simulator.updateAlpha()
        Simulator.updateR()
    Simulator.plotStats()
        # print("New Epsilon : ",sim.epsilon)
        # print("New Alpha : ",sim.alpha)
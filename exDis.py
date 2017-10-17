from __future__ import print_function
import heapq as hp

n = 36
adjList = [[] for i in xrange(n+1)]
f1 = open('edges','r')
for line in f1 :
	words = line.split()
	adjList[int(words[0])].append(int(words[1]))
f1.close()


def shortestPath(s,t):
	level = [0 for i in xrange(n+1)]
	h,cost = [],0
	cur = s
	while(cur!=t):
		# print("curr on ",cur,"with cost ",cost)
		nodes = adjList[cur]
		for node in nodes:
			if(level[node]==0):
				hp.heappush(h,[cost+1, node])
				level[node]=cur
		tmp = hp.heappop(h)
		cost = tmp[0]
		cur = tmp[1]
	return cost

dist = 0
for i in xrange(1,37):
	for j in xrange(i+1,37):
		tmp = shortestPath(i,j)
		dist += tmp

total = float(n*(n-1))/2
print(float(dist)/total)
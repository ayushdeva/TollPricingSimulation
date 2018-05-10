""" This is to print the edgeCount matrix for the edges in the graph """
def printEdgeCnt():
	total,cnt = 0,0
	f2 = open('graph','r')
	for line in f2 :
		words = line.split()
		cnt+=1
		total += edgeCount[int(words[0])][int(words[1])]
		print(edgeCount[int(words[0])][int(words[1])],end=" ")
		if(cnt==5):
			print()
			cnt = 0
	f2.close()
	return total
""" ends """
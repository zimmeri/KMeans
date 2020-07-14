
import numpy as np 
import sys
import pandas as pd
import random as r
import math as math






class KMeans(object):
	def __init__(self, args):
		self.k = args[0]
		self.option = args[1]
		self.data = np.array(args[2])
		if self.option == 2:
			#log transform review count and checkins
			for datapt in range(len(self.data)) :
				self.data[datapt][6] = math.log(self.data[datapt][6])
				self.data[datapt][7] = math.log(self.data[datapt][7])

		self.m = np.zeros(len(self.data))
		self.m = self.m.astype(int)
		self.clusterpoints = np.zeros(self.k * 5).reshape(self.k,5) #cluster representatives; 
		#clusterpoints[j] is a vector of the means of each feature in that cluster
		self.prevWC = 0

	def cluster(self):
		#latitude, longitude, reviewCount, checkins
		for cluster in range(self.k):
			point = r.randint(0,len(self.data)+1)
			latitude = self.data[point][3]
			longitude = self.data[point][4]
			reviewCount = self.data[point][6]
			checkins = self.data[point][7]
			self.clusterpoints[cluster] = [cluster, latitude,longitude,reviewCount,checkins]
		#nump this up b
		while(True):
			#for cluster in self.clusterpoints:
			#	for dataindex in range(len(self.data)):
			#		point = self.data[dataindex]
			#		dist = self.distance(mid,point)
			#		prevDist = self.distance(self.clusterpoints[self.m[dataindex]],point)
			#		if dist < prevDist:
			#			self.m[dataindex] = mid[0]
			for dataindex in range(len(self.data)):
				distances = []
				#find all the distances from the mean of each cluster for the point
				for clusterID in range(self.k):
					if self.option == 4:
						distances = np.append(distances, self.manDistance(self.clusterpoints[clusterID], self.data[dataindex]))
					else:
						distances = np.append(distances, self.distance(self.clusterpoints[clusterID], self.data[dataindex]))
				minID = np.argmin(distances)
				#find the minimum of these distances and reassign
				self.m[dataindex] = minID


			self.calcMean()
			wc = self.getWC()
			print("wc", wc)
			print("prevWC", self.prevWC)
			if wc == self.prevWC:
				return [wc, self.clusterpoints]	
			else:
				self.prevWC = wc
			

	def distance(self, mid, point):
		latDist = math.pow(point[3] - mid[1], 2)
		longDist = math.pow(point[4] - mid[2], 2)
		revDist = math.pow(point[6] - mid[3], 2)
		checkDist = math.pow(point[7] - mid[4],2)
		dist = math.sqrt(latDist + longDist + revDist + checkDist)
		return dist

	def manDistance(self, mid, point):
		latDist = abs(point[3] - mid[1])
		longDist = abs(point[4] - mid[2])
		revDist = abs(point[6] - mid[3])
		checkDist = abs(point[7] - mid[4])
		dist = latDist + longDist + revDist + checkDist
		return dist

	def calcMean(self):
		for x in range(self.k):
			karray = np.argwhere(self.m == x)
			latMean = 0
			longMean = 0
			revMean = 0
			checkMean = 0
			for y in karray:
				latMean = latMean + self.data[y[0]][3]
				longMean = longMean + self.data[y[0]][4]
				revMean = revMean + self.data[y[0]][6]
				checkMean = checkMean + self.data[y[0]][7]
			latMean = latMean / len(karray)
			longMean = longMean / len(karray)
			revMean = revMean / len(karray)
			checkMean = checkMean / len(karray)
			self.clusterpoints[x] = [x, latMean,longMean,revMean,checkMean]

	def getWC(self):
		wc = 0
		for x in range(self.k):
			karray = np.argwhere(self.m == x)
			for y in karray:
				point = self.data[y[0]]
				if self.option == 4:
					wc += math.pow(self.manDistance(self.clusterpoints[x], point), 2)
				else:
					wc += math.pow(self.distance(self.clusterpoints[x], point), 2)
		return wc



if __name__ == "__main__" :
	k = int(sys.argv[2])
	option = int(sys.argv[3])
	name = sys.argv[1]
	train_data = pd.read_csv("../data/given/" + name, delimiter = ',', index_col=None, engine='python')
	keens = KMeans([k,option,train_data])
	k = keens.cluster()
	print("WC-SSE=" + str(k[0]))
	for x in range(len(k[1])):
		print("Centroid" + str(x) +"=[" + str(k[1][x][1]) + "," + str(k[1][x][2]) + "," + str(k[1][x][3]) + "," +str(k[1][x][4]) + "]")



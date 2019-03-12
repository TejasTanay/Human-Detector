import cv2
import numpy as np 
import math


def HOG_Vec(name):
	img = cv2.imread(name,cv2.IMREAD_COLOR) #Reeading the image
	size=img.shape;
	n=img.shape[0]; # Assigning row and column values
	m=img.shape[1]; 

	

	#Changing RGB to Grayscale
	imgG=np.zeros((n,m))

	for i in range(0,n):
		for j in range(0,m):
			imgG[i][j]=round(0.299*img[i][j][2]+0.587*img[i][j][1]+0.114*img[i][j][0])



	#Calculating Gx

	mP=m-2
	nP=n-2

	Px=np.zeros((nP,mP))
	for i in range(1,n-1):
		for j in range(1,m-1):
			Px[i-1][j-1]=((imgG[i-1][j-1]*-1)+(imgG[i-1][j]*0)+(imgG[i-1][j+1]*1)+(imgG[i][j-1]*-1)+(imgG[i][j]*0)+(imgG[i][j+1]*1)+(imgG[i+1][j-1]*-1)+(imgG[i+1][j]*0)+(imgG[i+1][j+1]*1))
			
			Px[i-1][j-1]=Px[i-1][j-1]/3

	#Calculating Gy

	Py=np.zeros((nP,mP))
	for i in range(1,n-1):
		for j in range(1,m-1):
			Py[i-1][j-1]=((imgG[i-1][j-1]*1)+(imgG[i-1][j]*1)+(imgG[i-1][j+1]*1)+(imgG[i][j-1]*0)+(imgG[i][j]*0)+(imgG[i][j+1]*0)+(imgG[i+1][j-1]*-1)+(imgG[i+1][j]*-1)+(imgG[i+1][j+1]*-1))
			
			Py[i-1][j-1]=Py[i-1][j-1]/3

	

	#Calculating Magnitude

	mag=np.zeros((nP,mP))
	for i in range(nP):
		for j in range(mP):
			mag[i][j]=math.sqrt((Px[i][j]*Px[i][j])+(Py[i][j]*Py[i][j]))
			mag[i][j]=mag[i][j]/np.sqrt(2)
	c1=np.pad(mag, ((1,1),(1,1)), 'constant')
	#Padding Back to get the original number of pixel count
	mag=c1
	

	#Calculating Degree Matrix

	deg=np.zeros((nP,mP))
	for i in range(nP):
		for j in range(mP):
			if (Px[i][j]==0 and Py[i][j]==0):
				deg[i][j]=0
			if (Px[i][j]!=0):
				deg[i][j]= (math.degrees(np.arctan(Py[i][j]/Px[i][j])))
			else:
				if(Py[i][j]>0):
					deg[i][j]=90
				else:
					deg[i][j]=-90

			if(deg[i][j]<0):
				deg[i][j]=180+deg[i][j]

			if(deg[i][j]==-0):
				deg[i][j]=0

	#Padding Back to get the original number of pixel count
	b1=np.pad(deg, ((1,1),(1,1)), 'constant')
	deg=b1


	
    #Calculating Histograms for all cells
	row=math.floor(n/8)
	col=math.floor(m/8)
	#print(row,col)
	row1=0
	col1=0
	count=0


    #Storing all the Histograms in this 3D matrix
	cellHistStrct=np.zeros((row,col,9))



	for r in range(0,n,8):
		for c in range(0,m,8):
			i=r
			lim_i=i+8
			
			hist = [0, 0, 0, 0, 0, 0, 0, 0, 0]
			for i in range(i,lim_i):
				j=c
				lim_j=j+8
				for j in range(j,lim_j):
					
					
					if(deg[i][j] == 0 or deg[i][j] == 180):
						hist[0] += mag[i][j]
						continue
					if(deg[i][j] > 0 and deg[i][j] < 20):
						hist[0] += ((20-deg[i][j])/20)*mag[i][j]
						hist[1] += ((deg[i][j]-0)/20)*mag[i][j]
						continue
					if(deg[i][j] == 20):
						hist[1] += mag[i][j]
						continue
					if(deg[i][j] > 20 and deg[i][j] < 40):
						hist[1] += ((40-deg[i][j])/20)*mag[i][j]
						hist[2] += ((deg[i][j]-20)/20)*mag[i][j]
						continue
					if(deg[i][j] == 40):
						hist[2] += mag[i][j]
						continue
					if(deg[i][j] > 40 and deg[i][j] < 60):
						hist[2] += ((60-deg[i][j])/20)*mag[i][j]
						hist[3] += ((deg[i][j]-40)/20)*mag[i][j]
						continue
					if(deg[i][j] == 60):
						hist[3] += mag[i][j]
						continue
					if(deg[i][j] > 60 and deg[i][j] < 80):
						hist[3] += ((80-deg[i][j])/20)*mag[i][j]
						hist[4] += ((deg[i][j]-60)/20)*mag[i][j]
						continue
					if(deg[i][j] == 80):
						hist[4] += mag[i][j]
						continue
					if(deg[i][j] > 80 and deg[i][j] < 100):
						hist[4] += ((100-deg[i][j])/20)*mag[i][j]
						hist[5] += ((deg[i][j]-80)/20)*mag[i][j]
						continue
					if(deg[i][j] == 100):
						hist[5] += mag[i][j]
						continue
					if(deg[i][j] > 100 and deg[i][j] < 120):
						hist[5] += ((120-deg[i][j])/20)*mag[i][j]
						hist[6] += ((deg[i][j]-100)/20)*mag[i][j]
						continue
					if(deg[i][j] == 120):
						hist[6] += mag[i][j]
						continue
					if(deg[i][j] > 120 and deg[i][j] < 140):
						hist[6] += ((140-deg[i][j])/20)*mag[i][j]
						hist[7] += ((deg[i][j]-120)/20)*mag[i][j]
						continue
					if(deg[i][j] == 140):
						hist[7] += mag[i][j]
						continue
					if(deg[i][j] > 140 and deg[i][j] < 160):
						hist[7] += ((160-deg[i][j])/20)*mag[i][j]
						hist[8] += ((deg[i][j]-140)/20)*mag[i][j]
						continue
					if(deg[i][j] == 160):
						hist[8] += mag[i][j]
						continue
					if(deg[i][j] > 160):
						hist[8] += ((180-deg[i][j])/20)*mag[i][j]
						hist[0] += ((deg[i][j]-160)/20)*mag[i][j]
						continue
			
			count=count+1
					
			
			cellHistStrct[row1][col1]=hist
			col1=col1+1
		row1=row1+1
		col1=0


	#Now that all the histograms of cells are stored in the matrix we calculate histograms for blocks
	
	#CALCULATING BLOCKS

	value=(row-1)*(col-1)*4*9
	
	sum=0.0

	Vector=np.zeros(1)
	for i in range(0,row-1):
		for j in range(0,col-1):
			sum=0.0
			Vec=np.zeros(1)
			Vec=np.append(Vec,cellHistStrct[i][j])
			Vec=np.append(Vec,cellHistStrct[i][j+1])
			Vec=np.append(Vec,cellHistStrct[i+1][j])
			Vec=np.append(Vec,cellHistStrct[i+1][j+1])
			Vec1=Vec[1:]
			# performing L2 Norm
			for k in range(0,36):
				sum=sum+Vec1[k]*Vec1[k]
			sum=np.sqrt(sum)
			for k in range (0,36):
				if (sum==0):
					continue
				Vec1[k]=Vec1[k]/sum
			Vector=np.append(Vector,Vec1)



	Vector1=Vector[1:]
	
	
	
	return Vector1




HOG_Vec("pos10.bmp")











	










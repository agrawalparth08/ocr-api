from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json

import io
import cv2
import numpy as np


def evaluateQuestion(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)

	# apply Canny Edge Detection
	edged = cv2.Canny(blurred, 0, 50)

	(_,contours,_) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	# thresh = 100
	# im_bw = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)[1]  
	# cv2.imwrite("im_bw.jpg",im_bw)
	# height,width= im_bw.shape
	# im_bw = cv2.resize(im_bw,dsize = (width*5,height*4),interpolation = cv2.INTER_CUBIC)

	# cv2.imshow("Binary Image",im_bw)
	# cv2.waitKey(0)

	# ret,thresh = cv2.threshold(im_bw,127,255,cv2.THRESH_BINARY_INV)

	# im2,ctrs,hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	m = list()
	targetvec = list()
	for c in contours:
		p = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * p, True)
		
		if len(approx) == 4 and cv2.contourArea(approx)>1000 and cv2.contourArea(approx) <2000: #parameter which needs to be tuned for separate area size

			print("Area:",cv2.contourArea(approx))
			targetvec.append(approx)
	#cv2.drawContours(image,targetvec, -1,(0,255,0),1)
	
	m = list()
	
   


	point_list = []
	for c in targetvec:
		x1, y1, width1, height1 = cv2.boundingRect(c)
		point_list.append([x1,y1,width1,height1])


	#filter necessary so that the big outer contour is not detected
	point_array = [point for point in point_list]
	duplicate_array = []
	same_pt = []
	point_array = sorted(point_array,key=lambda x: (x[1]))
	for i in point_array:
		print ("Point Array :",i)
		
	for i in range(len(point_array)):
		for j in range(i+1,len(point_array)):
			#nearby contour points to remove
			if point_array[i][1]+ 10 > point_array[j][1]:
				point_array[j][1] = point_array[i][1]


	point_array = sorted(point_array,key=lambda x: (x[1],x[0]))

	for i in range(len(point_array)):
		for j in range(i+1,len(point_array)):
			if point_array[i][0]+ 10 > point_array[j][0] and  point_array[i][1]+ 10 > point_array[j][1] and point_array[i][2]+ 10 > point_array[j][2] and point_array[i][3]+ 10 > point_array[j][3] :
				duplicate_array.append(j)

	#deleting from reverse based on index to avoid out of index issue 
	duplicate_array = sorted(list(set(duplicate_array)),reverse=True)
	print("Points detected:",len(point_array),"Duplicate Points to be removed:",len(list(set(duplicate_array))))
	#print(duplicate_array)
	for i in duplicate_array:
		print ("Deleted",i)

	for i in duplicate_array:
		del point_array[i]

	for i in point_array:
		print("final points",i   )
	roilist = []
	 # if not os.path.isdir('rois'):
	 # 	 os.makedirs('rois') 
	for i  in range(0,len(point_array)):

			x, y, width, height = point_array[i][0],point_array[i][1],point_array[i][2],point_array[i][3]
			#if y < 720:
			#cropping some padding which contains box lines
			roi = image[y+2:y+height-1, x+3:x+width-3]
			
			cv2.rectangle(image,(x,y),(x+width,y+height),(0,255,0),1)
			#print(roi.shape)
			# print("height - width {}".format(abs(height-width)))   
   #      	cv2.imwrite("imbw.png"), roi) 
			roilist.append(roi)
			
			


	json_file = open("model.json","r")
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights("model.h5")

	model = loaded_model
	characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

	#image = cv2.imread("response_2.jpg")
	responselist = []
	for roi in roilist:
		thresh = 170    
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

		kernel1 = np.ones((3,3),np.uint8)
		cv2.imshow("roi1", roi)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		
		# cv2.imshow("dst", dst)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		im_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,51,12)


		#_,im_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		

		#bn_im_bw_cl = cv2.morphologyEx(bn_im_bw, cv2.MORPH_CLOSE, kernel,iterations=2)
		
		# cv2.imshow("imdilate", imdilate)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# cv2.imshow("im_bw1", im_bw)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		
		# kernels = np.ones((1,5), np.uint8)  # note this is a horizontal kernel
		# e_im = cv2.erode(im_bw, kernels, iterations=1)
		# d_im = cv2.dilate(e_im, kernels, iterations=1)
		
		im_bw = cv2.erode(im_bw, kernel, iterations=1)
		# blur = cv2.GaussianBlur(gray,(3,3),0)
		# smooth = cv2.addWeighted(blur,1.5,gray,-0.5,0)
		# cv2.imshow("smooth",smooth)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		_,im_bw = cv2.threshold(im_bw, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		cv2.imshow("im_bw2",im_bw)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		 
		cv2.imwrite("im_bw.jpg",im_bw)
		height,width = im_bw.shape
		im_bw = cv2.resize(im_bw,dsize = (width*5,height*4),interpolation = cv2.INTER_CUBIC)

		

		ret,thresh = cv2.threshold(im_bw,127,255,cv2.THRESH_BINARY_INV)

		im2,ctrs,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
		m = list()
		sorted_ctrs = sorted(ctrs, key = lambda ctr: cv2.boundingRect(ctr)[0])
		
		
		pchl = list()

		dp = im_bw.copy()

		response = []
		old_point_x = 0
		old_point_y = 0

		for i,ctr in enumerate(sorted_ctrs):
			x,y,w,h = cv2.boundingRect(ctr)
			
			print("Height, Weight, W/h, X , Y ->",h,w,float(w)/h,x,y)
		# 	cv2.rectangle(dp,(x-10,y-10),(x+w+10,y+h+10),(90,0,255),3)

		# cv2.imshow("final",dp)   
		# cv2.waitKey(0)
			
			if float (w/h) < 3 and x>5 and y>10:
				print("Contour:",(np.amax(ctr,axis = 0)))
				max_ctr = np.amax(ctr,axis = 0)

				if max_ctr[0][0] > (old_point_x + 20): #for characters detected too close

					old_point_x = max_ctr[0][0]
					old_point_y = max_ctr[0][1]
					roi = im_bw[y-10:y+h+10, x-5:x+w+10]
				
					
					roi = cv2.resize(roi,dsize = (28,28), interpolation = cv2.INTER_AREA)
					kernel = np.ones((2,2),np.uint8)

					#roi = cv2.erode(roi,kernel,iterations = 1)
					
					#roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
					cv2.imshow("roi2", roi)
					cv2.waitKey(0)
					cv2.destroyAllWindows()

					roi = np.array(roi)
					t = np.copy(roi)
					t = t /255.0
					t = 1 - t
					t = t.reshape(1,784)
					

					prob = model.predict_proba(t)
					prob_list = prob[0].argsort()[-3:][::-1]

					top_response = {
					"t1":[characters[prob_list[0]],prob[0][prob_list[0]]],
					"t2":[characters[prob_list[1]],prob[0][prob_list[1]]],
					"t3":[characters[prob_list[2]],prob[0][prob_list[2]]]}
					#print(top_response)
					print(characters[prob_list[0]],characters[prob_list[1]])


					response.append(characters[prob_list[0]])
		responselist.append(response)
		print(response)
	return(responselist)
	

#print(evaluateQuestion(cv2.imread("image.png")))


 

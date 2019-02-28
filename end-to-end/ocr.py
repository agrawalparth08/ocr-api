from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json

import io
import cv2
import numpy as np


def evaluateQuestion(image):

	json_file = open("model.json","r")
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights("model.h5")

	model = loaded_model
	characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

	#image = cv2.imread("response_2.jpg")
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	thresh = 127
	im_bw = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)[1]  
	cv2.imwrite("im_bw.jpg",im_bw)
	height,width = im_bw.shape
	im_bw = cv2.resize(im_bw,dsize = (width*5,height*4),interpolation = cv2.INTER_CUBIC)

	cv2.imshow("Binary Image",im_bw)
	cv2.waitKey(0)

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
			
				cv2.imshow("roi",roi)
				cv2.waitKey(0)
				roi = cv2.resize(roi,dsize = (28,28), interpolation = cv2.INTER_AREA)
				kernel = np.ones((2,2),np.uint8)
				roi = cv2.erode(roi,kernel,iterations = 1)
				#roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
				# cv2.imshow("roi",roi)
				# cv2.waitKey(0)

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
				print(characters[prob_list[0]])


				response.append(characters[prob_list[0]])
	print(response)
	return(response)
	

#print(evaluateQuestion(cv2.imread("roi9.png")))




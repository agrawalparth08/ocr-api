# # import the necessary packages
import numpy as np
import cv2


def getBlob(im):

	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 0
	params.maxThreshold = 100

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 100

	# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0.3
	# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0.5
	    
	# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.3

	# Create a detector with the parameters
	ver = (cv2.__version__).split('.')
	if int(ver[0]) < 3 :
		detector = cv2.SimpleBlobDetector(params)
	else : 
		detector = cv2.SimpleBlobDetector_create(params)


	# Detect blobs.
	keypoints = detector.detect(im)

	def getKey(item):
		return item[1]

	im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	li = []
	for i in range(len(keypoints)):
		#print(i,"x:",keypoints[i].pt)
		li.append(keypoints[i].pt)


	keypoint_sorted = sorted(li, key=getKey)
	#for i in range(len(keypoint_sorted)):
		#print(keypoint_sorted[i])
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.imwrite("partial_OMR_detected.jpg",im_with_keypoints)
	cv2.waitKey(0)

	return keypoint_sorted


def getCircles(image):

	output = image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT, 1, 20,
	              param1=45,
	              param2=22,
	              minRadius=0,
	              maxRadius=55)


	point_list = []
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")

		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			#print(x,y)
			point_list.append([x,y])
			# corresponding to the center of the circle
			cv2.circle(output, (x, y), r, (0, 255, 0), 1)
			#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

		# show the output image
		cv2.imshow("output", np.hstack([output]))
		cv2.waitKey(0)

	#sort by x coord because we will get row_count in metadata
	point_list = sorted(point_list,key=lambda x: x[0])

	return point_list

def evaluateQuestion(image,row_count=2,x_response = ["A","B","C","D"],y_response= ["Vertebrate","Invertebrate"]):

#if __name__ == "__main__":

	#load image
	#image = cv2.imread("roi5.png")

	#get all circles
	# print("shape of OMR: ", image.shape)
	# point_list = getCircles(image)

	# #setting the x-y range based on circles
	# x_range = []
	# y_range = sorted([point_list[i][1] for i in range(row_count)])

	# print(y_range)
	# for i in range(0,len(point_list),row_count):
	# 	row_group = point_list[i:i+row_count]

	# 	x_range.append(min([row[0] for row in row_group ]))
	# print(x_range)

	x_range = [60,110,160,210]

	#Detecting blob points
	blob_points = getBlob(image)
	print ("blob_points ", blob_points)

	# #final response list
	responses = []

	for point in blob_points: 
		print(point) 
		found = False 
		for i in range(len(x_range)): 
			if int(point[0]) < x_range[i]: 
				responses.append(i + 1)
				#print(responses) 
				found = True 
			if found:break
		
	print("Final responses")
	print(responses)
	return responses
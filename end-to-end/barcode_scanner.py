# import the necessary packages
# sudo apt-install libzbar0
# pip install pyzbar

from pyzbar import pyzbar
import argparse
import cv2
 
# construct the argument parser and parse the arguments
def decoder(image):
	# load the input image
	#image = cv2.imread(args["image"])
	barcode_image = image.copy()
	# find the barcodes in the image and decode each of the barcodes
	barcodes = pyzbar.decode(barcode_image)
	 
	# loop over the detected barcodes
	for barcode in barcodes:
		# extract the bounding box location of the barcode and draw the
		# bounding box surrounding the barcode on the image
		(x, y, w, h) = barcode.rect
		cv2.rectangle(barcode_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	 
		# the barcode data is a bytes object so if we want to draw it on
		# our output image we need to convert it to a string first
		barcodeData = barcode.data.decode("utf-8")
		barcodeType = barcode.type
	 
		# draw the barcode data and barcode type on the image
		text = "{} ({})".format(barcodeData, barcodeType)
		cv2.putText(barcode_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)
	 
		# print the barcode type and data to the terminal
		print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
		cv2.imwrite("barcoded.jpg",barcode_image)
	 
	# show the output image
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	return barcodeData
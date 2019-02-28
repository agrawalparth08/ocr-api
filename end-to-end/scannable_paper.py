import cv2
import numpy as np
import matplotlib.pyplot as plt
from barcode_scanner import decoder
import pandas as pd
import omr
import ocr


def rectify(h):
    h = h.reshape((4,2))
    print (h)
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    print (add)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1) 
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def outerRectangle(image):


    height, width, channels = image.shape
    if width > height:
        image = cv2.transpose(image) 
        image = cv2.flip(image,1)

    # resize image so it can be processed
    image = cv2.resize(image, (1600, 1200))  

    # creating copy of original image
    orig = image.copy()

    # convert to grayscale and blur to smooth
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #blurred = cv2.medianBlur(gray, 5)

    # apply Canny Edge Detection
    edged = cv2.Canny(blurred, 0, 50)
    orig_edged = edged.copy()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (_,contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get approximate contour
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break

    # mapping target points to 800x800 quadrilateral
    approx = rectify(target)
    pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

    M = cv2.getPerspectiveTransform(approx,pts2)

    dst = cv2.warpPerspective(orig,M,(800,800))
    cv2.imshow("first",dst)
    cv2.waitKey(0)
    return dst



def innerRectangles(dst):

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #cv2.imshow("blurred", blurred)

    # apply Canny Edge Detection
    edged = cv2.Canny(blurred, 0, 50)

    (_,contours,_) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    targetvec = list()
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        #print("Area:",cv2.contourArea(approx))
        
        if len(approx) == 4 and cv2.contourArea(approx) >4000: #parameter which needs to be tuned for separate area size

            # print(cv2.contourArea(approx) == cv2.contourArea(contours))
            #print("Area:",cv2.contourArea(approx))
            targetvec.append(approx)

    #cv2.drawContours(dst, targetvec, -1, (0, 255, 0), 2)
    #cv2.imshow("image",dst)
    #cv2.waitKey(0)

    point_list = []
    for c in targetvec:
        x1, y1, width1, height1 = cv2.boundingRect(c)
        point_list.append([x1,y1,width1,height1])
        #print("x, y , height, width .... {} {} {} {}",x1,y1,width1,height1)

    #filter necessary so that the big outer contour is not detected
    point_array = [point for point in point_list if point[0] > 10]

    #print(point_array)
    duplicate_array = []
    same_pt = []
    point_array = sorted(point_array,key=lambda x: (x[1],x[0]))

    #Actual Points
    #for i in point_array:
        #print ("P:",i)

    #Duplicate point indices encountered are maintained in the list
    for i in range(len(point_array)):
        for j in range(i+1,len(point_array)):
            #nearby contour points to remove
            if point_array[i][0]+ 10 > point_array[j][0] and point_array[i][1]+ 10 > point_array[j][1] and point_array[i][2]+ 10 > point_array[j][2] and point_array[i][3]+ 10 > point_array[j][3] :
                #print("T:",point_array[i],point_array[j])
                duplicate_array.append(j)

    #deleting from reverse based on index to avoid out of index issue 
    duplicate_array = sorted(list(set(duplicate_array)),reverse=True)
    print("Points detected:",len(point_array),"Duplicate Points to be removed:",len(list(set(duplicate_array))))
    #print(duplicate_array)

    for i in duplicate_array:
        del point_array[i]

    for i in point_array:
        print ("Final:",i)

    for i  in range(0,len(point_array)):

        x, y, width, height = point_array[i][0],point_array[i][1],point_array[i][2],point_array[i][3]
        #if y < 720:
        #cropping some padding which contains box lines
        roi = dst[y+3:y+height-3, x+5:x+width-5]
        #cv2.rectangle(dst,(x,y),(x+width,y+height),(0,255,0),1)
        #print(roi.shape)
        #cv2.imshow("ROI",roi)
        #cv2.waitKey(0)
        cv2.imwrite("roi"+str(i+1)+".png", roi)
        #cv2.destroyAllWindows()

    return len(point_array)


if __name__ == "__main__":

    '''

    Step 1 
    Get the image and contour area (black box surrounding the image)

    '''

    image = cv2.imread('test_multi.jpg') 
    dst = outerRectangle(image)


    '''

    Step 2 
    Get the barcode string from the cropped image dst

    '''

    # x,y,w,h = 600,650,160,120 # will always be at the bottom of the page, coordinates need to be hardcoded
    # crop= dst[y:h+y,x:w+x]

    # cv2.imshow("snip - barcode",crop )
    # cv2.waitKey(0)

    # barcode_string = decoder(crop)
    # print(barcode_string)


    '''

    Step 3
    Fetch all the co-ordinates and question data from the csv using barcode string as id

    '''

    qpts_data = pd.read_csv("question_data.csv")


    '''

    Step 4 Detecting all questions/boxes and saving all those images


    '''

    regions_detected = innerRectangles(dst)
    print("Detected regions :",regions_detected)

    '''

    Step 5 Evaluating questions/boxes


    '''

    responses = []
    q_types = ["ocr","rough","omr","rough","omr","rough","ocr","rough","ocr"]


    for i in range(regions_detected):
        q_img = "roi"+str(i+1)+".png"
        if q_types[i] == "omr":
            img = cv2.imread(q_img)
            responses.append(omr.evaluateQuestion(img))

        if q_types[i] =="ocr":
            img = cv2.imread(q_img)
            responses.append(ocr.evaluateQuestion(img))

    print(responses)









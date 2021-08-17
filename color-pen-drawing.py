import cv2 as cv
import numpy as np

widthImage = 640
heightImage = 480
capture = cv.VideoCapture(2)
capture.set(3, widthImage)
capture.set(4, heightImage)
capture.set(10, 150)

def stackImages(scale,imgArray):

    rows = len(imgArray)

    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]

    height = imgArray[0][0].shape[0]

    if rowsAvailable:

        for x in range ( 0, rows):

            for y in range(0, cols):

                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:

                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)

                else:

                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)

                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)

        hor = [imageBlank]*rows

        hor_con = [imageBlank]*rows

        for x in range(0, rows):

            hor[x] = np.hstack(imgArray[x])

        ver = np.vstack(hor)

    else:

        for x in range(0, rows):

            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:

                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)

            else:

                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)

            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)

        hor= np.hstack(imgArray)

        ver = hor

    return ver

def getCont(frames):
    biggest = np.array([])
    maxArea = 0
    contours, hierachy = cv.findContours(frames, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1200:
            # cv.drawContours(copy, cnt, -1, (0,255,0), 1)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4:
                biggest = approx
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv.drawContours(copy, biggest, -1, (0,255,0), 20)
    return biggest

def process(frames):
    gray = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 1)
    canny = cv.Canny(blur, 200, 200)
    kernel = np.ones((5,5))
    dilation = cv.dilate(canny, kernel=kernel, iterations=2)
    thresh = cv.erode(dilation, kernel=kernel, iterations=1)

    return thresh

def orderer (points):
    points = points.reshape((4,2))
    pointsnew = np.zeros((4,1,2), np.int32)
    add = points.sum(1)
    
    pointsnew[0]= points[np.argmin(add)]
    pointsnew[3]= points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsnew [1] = points[np.argmin(diff)]
    pointsnew [2] = points[np.argmax(diff)]

    return pointsnew

def warper(frames, biggest):
    orderer(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImage,0],[0, heightImage],[widthImage, heightImage]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_out= cv.warpPerspective(frames, matrix, (widthImage, heightImage))
    
    cropper = img_out[20:img_out.shape[0]-20, 20:img_out.shape[1]-20]
    imgcropper = cv.resize(cropper, (widthImage, heightImage))
    return img_out
while True:

    isTrue, frames = capture.read()
    frames = cv.resize(frames,(widthImage, heightImage)) 
    copy = frames.copy()
    thresh = process(frames)
    biggest = getCont(thresh)
    if biggest.size != 0:  
        imagewarper = warper(frames, biggest)
        imageArray = ([frames, copy], [thresh, imagewarper])
    else:
        imageArray = ([frames, copy], [frames, frames])
    stackedImages = stackImages(0.5, imageArray)
    cv.imshow('video', stackedImages)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows() 

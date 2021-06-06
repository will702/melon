
import cv2
import numpy as np

def find_blob(img_ori):
	#n = 6
	#img_ori = cv2.imread('test/sick/roi_f{}.jpg'.format(n))
	res = cv2.resize(img_ori,(240,240))
	gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	#gray = 255-img_ori
	blur = cv2.GaussianBlur(gray, (13,13),0)

	#thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 67,-1)
	_,thresh1 = cv2.threshold(blur,165,255,cv2.THRESH_BINARY)

	im_thr = cv2.bitwise_and(res,res, mask= thresh1)
	im_blur = cv2.blur(blur, (3,3))

	kernel=np.ones((5,5),np.uint8)
	img_erosion = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernel) 

	kernel=np.ones((4,4),np.uint8)
	img_delate = cv2.morphologyEx(blur, cv2.MORPH_ERODE, kernel) 

	#opening = cv2.morphologyEx(img_erosion, cv2.MORPH_CLOSE, kernel)
	#opening2 = cv2.morphologyEx(img_erosion, cv2.MORPH_OPEN, kernel)
	gradiant = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

	op2 = img_delate + img_erosion 
	op = op2 + gradiant
	#_,thresh2 = cv2.threshold(op,75,255,cv2.THRESH_BINARY)
	thresh2 = cv2.adaptiveThreshold(op, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21,8)

	colored = cv2.cvtColor(op2, cv2.COLOR_GRAY2BGR)
	colored[np.all(colored == [0,0,0], axis=-1)] = [0,0,255]
	#colored[np.all(colored == [255,255,255], axis=2)] = [0,255,0]
	# Set up the detector with default parameters.
	#detector = cv2.SimpleBlobDetector()

	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 10
	params.maxThreshold = 255

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 10
	params.maxArea = 9000000

	# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0.01
	params.maxCircularity = 1

	# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0.1
	params.maxConvexity = 1

	# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.1
	params.maxInertiaRatio = 1

	
	# Create a detector with the parameters
	ver = (cv2.__version__).split('.')
	if int(ver[0]) < 3 :
		detector = cv2.SimpleBlobDetector(params)
	else : 
		detector = cv2.SimpleBlobDetector_create(params)

	# Detect blobs.
	keypoints = detector.detect(thresh2)
	#blobs = cv2.drawKeypoints(res, keypoints, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
	#for kp in keypoints:
	#	x, y = kp.pt
	#	cv2.circle(res, (int(x), int(y)), 15, (100,155,0))	# Show keypoints
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	#blods_img = cv2.drawKeypoints(res, keypoints, 20, (0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	b = len(keypoints)

	# Show keypoints
	""" 
	cv2.imshow("thr1", thresh1)
	cv2.imwrite('op21.png', op2)
	cv2.imshow("erosion",img_erosion)
	cv2.imshow("delate",img_delate)
	cv2.imshow("gradiant",gradiant)
	cv2.imshow("op",op)

	cv2.imshow("blur", blur)
	
	cv2.imshow("thr2", thresh2)


	cv2.waitKey(0)
	"""
	return b
"""
n = 5
img = cv2.imread('test/sick/sick2.jpg'.format(n))
x = find_blob(img)
#print(type(x))
print(x)
# Read image
	#img_ori = cv2.imread("test/rois4.jpg")
"""

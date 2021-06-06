import numpy as np
import cv2


def find_sample(roi_img):
	#im = cv2.imread(roi_img)
	img = cv2.resize(roi_img, (640,480))

	imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask_green1 = cv2.inRange(imghsv, np.array([30, 100, 170]), np.array([179, 255, 255]))
	mask_green2 =cv2.inRange(imghsv, np.array([0, 88, 120]), np.array([38, 150, 255]))
	mask_green = cv2.max(mask_green1, mask_green2)

	mask1 = cv2.inRange(imghsv, np.array([0, 110, 88]), np.array([30, 255, 255]))

	mask_yallow1 = cv2.inRange(imghsv, np.array([0, 110, 88]), np.array([30, 255, 255]))
	mask_yallow2 = cv2.inRange(imghsv, np.array([0, 67, 185]), np.array([35,255, 255]))
	mask_yallow3 = cv2.inRange(imghsv, np.array([0, 0, 0]), np.array([100,70, 214]))

	mask_yallow = cv2.max(mask_yallow2, mask_yallow3)

	mask_orange0 = cv2.inRange(imghsv, np.array([0, 84, 178]), np.array([179, 255, 255]))
	mask_orange = cv2.max(mask_yallow1, mask_orange0)
	#use mask with the highes number of white pixel 
	val1=np.sum(mask_green==255)
	val2=np.sum(mask_yallow==255)
	val3=np.sum(mask_orange==255)

	if(val1>val2 and val1>val3):
		mask=mask_green
	elif(val2>val1 and val2>val3):
		mask=mask_yallow
	else:
		mask=mask_orange

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	masked_result = cv2.bitwise_and(img, img, mask=mask)

	for cnt in contours:
		cnt = max(contours, key = cv2.contourArea)
		x,y,w,h = cv2.boundingRect(cnt)
		roi = img[y:y+h, x:x+w]
		roi = cv2.resize(roi, (500,500))
		(roi_h, roi_w) = roi.shape[:2] #w:image-width and h:image-height

		c_c_x = roi_w//2
		c_c_y = roi_h//2
		r = 100
		#cv2.circle(roi, (c_c_x, c_c_y), r, (255, 255, 255), 1)
		roi_sample = roi[c_c_y - r:c_c_y + r, c_c_x - r:c_c_x + r]
	#print(np.sum(cnt))

	return masked_result, roi_sample
"""

def extract_rgb(roi):
	(B, G, R) =  cv2.split(roi.astype("float"))

	b, g, r = round(np.mean(B),2), round(np.mean(G),2), round(np.mean(R),2)
	#print("Blue: "+str(b))
	print(b)
	print(g)
	print(r)


image = 'test/melon.jpg'
mask, roi = find_sample(image)
#extract_rgb(roi)
#cv2.imwrite('test/sick/roi_fin1.jpg', roi)
cv2.imshow('Mask', mask)
cv2.imshow('ROI', roi)

cv2.waitKey()
cv2.destroyAllWindows()
"""

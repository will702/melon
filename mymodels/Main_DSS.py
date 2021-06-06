import numpy as np
import cv2
import os

from mymodels import fuzzy_bgr
from mymodels import segment_roi
from mymodels import sick_melon

classNames = {1: 'Melon', 2: 'not'}
def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

def extract_rgb(roi):
	#img = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
	#(B, G, R) =  cv2.split(roi.astype("float"))
	B = roi[:, :, 0]
	G = roi[:, :, 1]
	R = roi[:, :, 2]

	b = np.mean(B)
	g = np.mean(G)
	r = np.mean(R)
	# Norm process
	total = b+g+r
	
	b_norm = round(b/total*255)
	g_norm = round(g/total*255)
	r_norm = round(r/total*255)
	return b_norm, g_norm, r_norm

def detect_melon(cvNet, image, im_path):
	#im = cv2.imread(image)
	img = cv2.resize(image, (480,360))
	#img = cv2.cvtColor(img_re, cv2.COLOR_RGB2BGR)

	#cvNet = cv2.dnn.readNetFromTensorflow('./ssd_melon_model_18853/frozen_inference_graph.pb', 
	#										'./ssd_melon_model_18853/graph_ori.pbtxt')
	rows = img.shape[0]
	cols = img.shape[1]

	cvNet.setInput(cv2.dnn.blobFromImage(img, size=(480, 360), swapRB=True, crop=False))
	cvOut = cvNet.forward()

	for detection in cvOut[0,0,:,:]:
		score = float(detection[2])
		if score > 0.7:
			class_id = detection[1]
			class_name=id_class_name(class_id,classNames)
			if class_name != 'not':
				left = detection[3] * cols
				top = detection[4] * rows
				right = detection[5] * cols
				bottom = detection[6] * rows
				Confidence = str(round(detection[2], 5))

				melon = img[int(top):int(bottom), int(left):int(right)]
	try:
		identify_melon = 1
		melon_ready = melon.copy()

		melon_mask, roi_fin = segment_roi.find_sample(melon_ready)
		blobs = sick_melon.find_blob(roi_fin)

		b, g, r = extract_rgb(roi_fin)
		try:
			ind = fuzzy_bgr.fuzzy_ripe_index(b, g, r, im_path)
			index = str(round(ind,  2))
			blue = str(b)
			green = str(g)
			red = str(r)
		except ValueError:
			index = -1
			blue = str(b)
			green = str(g)
			red = str(r)

		fin = img.copy()
		cv2.rectangle(fin, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
		if float(index) > 0:
			if float(index) < 3.5:
				Ripness = "Under Ripe"
			elif float(index) >= 3.5 and float(index) < 6.5:
				Ripness = "About to Ripe"
			elif float(index) >= 6.5:
				Ripness = "Ripe"
		elif float(index) == -1:
			Ripness = "False positive"

		cv2.putText(fin,  "{}".format(Ripness), (int(left), int(top)-10),
				cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,200), 2)

		if blobs > 20:
			cv2.putText(fin, "Sick Melon", (10 , 40),
				cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,200), 2)
				
		cv2.imwrite(os.path.join(im_path, "mymodels/output_images/",'melon.png'),melon)
		cv2.imwrite(os.path.join(im_path, "mymodels/output_images/",'Final.png'),fin)
		cv2.imwrite(os.path.join(im_path, "mymodels/output_images/",'melon_mask.png'),melon_mask)
		cv2.imwrite(os.path.join(im_path, "mymodels/output_images/",'roi_fin.png'),roi_fin)

	except UnboundLocalError:
		identify_melon = 0
	return identify_melon
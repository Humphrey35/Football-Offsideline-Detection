# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
args = vars(ap.parse_args())

lower = {'white':(165, 2, 100)} #assign new item lower['blue'] = (93, 10, 0)
upper = {'white':(330,1,100)}
kernel = np.ones((5,15),np.uint8)

if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the HOG descriptor/person detector
winSize = (48,96)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
#derivAperture = 1
#winSigma = 4.
#histogramNormType = 0
#L2HysThreshold = 2.0000000000000001e-01
#gammaCorrection = 0
#nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)#,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#hog = cv2.HOGDescriptor(cv2.Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1,-1, cv::HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

# loop over the image paths
#imagePaths = list(paths.list_images(args["images"]))

#for imagePath in imagePaths:
while True:
	(grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	#image = cv2.imread(imagePath)
	#image = imutils.resize(image, width=min(400, image.shape[1]))
	t = int(frame.shape[1] * 1.4)
	image = imutils.resize(frame, width=t)
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4))

	# draw the original bounding boxes
	#for (x, y, w, h) in rects:
	for i in range(len(weights)):
		if (weights[i] > 1.1):
			(x, y, w, h) = rects[i]
			
			# color dect
			crop = orig[y:y + h, x:x + w]
			#blurred = cv2.GaussianBlur(crop, (11, 11), 0)
			hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, (0,0,180), (255, 40, 255))
			mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
			mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
			
			print((mask>254).sum() / (len(mask)*len(mask[0])))
			#print(len(mask)*len(mask[0]))
			cv2.imshow("test",mask)
			cv2.imshow("original", crop)
			key = cv2.waitKey(0)
			#exit();
			
			# 
			#cv2.putText(orig, str(np.round(weights[i], 2)),  (x, y), 3, 2, (255, 255, 255))
			#cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	#rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	#pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	#for (xA, yA, xB, yB) in pick:
	#	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	#filename = imagePath[imagePath.rfind("/") + 1:]
	#print("[INFO] {}: {} original boxes, {} after suppression".format(
	#	filename, len(rects), len(pick)))

	# show the output images
	#image = imutils.resize(image, width=720)
	orig = imutils.resize(orig, width=720)
	cv2.imshow("Before NMS", orig)
	#cv2.imshow("After NMS", image)
	key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
		
camera.release()
cv2.destroyAllWindows()
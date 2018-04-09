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

lower = {'team1':(0,0,180), 'green':(35, 50, 50), 'team2':(120, 40, 85)} #team1 = white
upper = {'team1':(255, 40, 255), 'green':(60, 255, 255), 'team2':(186,255,255)}
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
	gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
	blur_gray = cv2.GaussianBlur(gray, (15, 15), 0)
	
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	
	rho = 1  # distance resolution in pixels of the Hough grid
	theta = np.pi / 90  # angular resolution in radians of the Hough grid
	threshold = 15  # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 200  # minimum number of pixels making up a line
	max_line_gap = 10  # maximum gap in pixels between connectable line segments
	line_image = np.copy(image) * 0  # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
	offsideLine = np.array([])
	
	print(len(lines))
	for line in lines:
		for x1,y1,x2,y2 in line:
			if (((x2-x1) < (y2-y1))):
				offsideLine = np.append(offsideLine, line)
				break
				#cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

	#image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
	
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4))

	# draw the original bounding boxes
	#for (x, y, w, h) in rects:
	
	team1Player = np.array([])
	team2Player = np.array([])
	team1PlayerPositionX = 0
	team2PlayerPositionX = 0
	team1PlayerPositionY = 0
	team2PlayerPositionY = 0
	
	for i in range(len(weights)):
		if (weights[i] > 0.7):
			(x, y, w, h) = rects[i]
			
			# color dect
			crop = orig[y:y + h, x:x + w]
			#blurred = cv2.GaussianBlur(crop, (11, 11), 0)
			hsvTeam1 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
			maskTeam1 = cv2.inRange(hsvTeam1, lower['team1'], upper['team1'])
			maskTeam1 = cv2.morphologyEx(maskTeam1, cv2.MORPH_OPEN, kernel)
			maskTeam1 = cv2.morphologyEx(maskTeam1, cv2.MORPH_CLOSE, kernel)
			
			percTeam1 = (maskTeam1>254).sum() / (len(maskTeam1)*len(maskTeam1[0]))
			
			print("Team1/weiß " + str((maskTeam1>254).sum() / (len(maskTeam1)*len(maskTeam1[0]))))
			
			hsvGreen = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
			maskGreen = cv2.inRange(hsvGreen, lower['green'], upper['green'])
			maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_OPEN, kernel)
			maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_CLOSE, kernel)
			
			percGreen = (maskGreen>254).sum() / (len(maskGreen)*len(maskGreen[0]))
			
			print("Grün " + str((maskGreen>254).sum() / (len(maskGreen)*len(maskGreen[0]))))
			
			hsvTeam2 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
			maskTeam2 = cv2.inRange(hsvTeam2, lower['team2'], upper['team2'])
			maskTeam2 = cv2.morphologyEx(maskTeam2, cv2.MORPH_OPEN, kernel)
			maskTeam2 = cv2.morphologyEx(maskTeam2, cv2.MORPH_CLOSE, kernel)
			
			percTeam2 = (maskTeam2>254).sum() / (len(maskTeam2)*len(maskTeam2[0]))
			
			print("Team2/rot " + str((maskTeam2>254).sum() / (len(maskTeam2)*len(maskTeam2[0]))))
			
			print(" ")
			
			#print(len(mask)*len(mask[0]))
			
			maskTeam2 = imutils.resize(maskTeam2, width=300)
			crop = imutils.resize(crop, width=300)
			
			#cv2.imshow("test",maskTeam2)
			#cv2.imshow("original", crop)
			#key = cv2.waitKey(0)
			# if the 'q' key is pressed, stop the loop
			#if key == ord("q"):
			#	break
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
			
				
			if (percGreen < 0.7 and percGreen > 0.1):
				#cv2.imshow("original", crop)
				if (percTeam1 > 0.03):
					team2Player = np.append(team2Player, rects[i])
					cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
					if (team1PlayerPositionX < (x+w)):
						team1PlayerPositionX = x+w
						team1PlayerPositionY = y+h
				else:
					team1Player = np.append(team1Player, rects[i])
					cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
					if (team2PlayerPositionX < (x+w)):
						team2PlayerPositionX = x+w
						team2PlayerPositionY = y+h
				#key = cv2.waitKey(0)
				#if key == ord("q"):
				#	break
			#exit();
			
			# 
			#cv2.putText(orig, str(np.round(weights[i], 2)),  (x, y), 3, 2, (255, 255, 255))

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
	
	x1,y1,x2,y2
	lineX = (offsideLine[3] - offsideLine[1]) / (offsideLine[2] - offsideLine[0])
	lineB =  offsideLine[1] - lineX*offsideLine[0]
	
	t1y = 0
	t1x = int(lineB)
	t2y = int(image.shape[1])
	t2x = int(lineX*image.shape[1] + lineB)
	
	xtmp = offsideLine[2]-offsideLine[0]
	ytmp = offsideLine[3]-offsideLine[1]
	
	lineOffsideX = lineX
	lineOffsideB =  team1PlayerPositionY - lineOffsideX*team1PlayerPositionX
	
	o1y = 0
	o1x = int(lineOffsideB)
	o2y = int(image.shape[1])
	o2x = int(lineOffsideX*image.shape[1] + lineOffsideB)
	
	cv2.line(image, (t1y, t1x), (t2y,t2x), (255,0,0), 5)
	
	cv2.line(image, (o1y, o1x), (o2y,o2x), (255,0,0), 5)
	
	image = imutils.resize(image, width=720)
	cv2.imshow("Before NMS", image)
	#cv2.imshow("After NMS", image)
	key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
		
camera.release()
cv2.destroyAllWindows()
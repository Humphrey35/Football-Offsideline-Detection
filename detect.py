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

# define shirt colors of the teams and the pitch
lower = {'team1':(0,0,180), 'green':(35, 50, 50), 'team2':(120, 40, 85)} #team1 = white
upper = {'team1':(255, 40, 255), 'green':(60, 255, 255), 'team2':(186,255,255)}
kernel = np.ones((5,15),np.uint8)

if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the HOG descriptor/person detector
# interesstingly most variables only support one parameter right now
winSize = (48,96)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

# use Daimler detector because he is trained to recognise people 96x48
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

offsideLine = np.array([])

# loop over the video
while True:
	(grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
	# load the image and resize it to improve detection accuracy
	# since the most of our source material is 720p and players are about
	# 1/5th of the picture we need to scale it larger to get to the 96px height
	t = int(frame.shape[1] * 1.4)
	image = imutils.resize(frame, width=t)
	orig = image.copy()
	
	# line detection: gray image / fill lines / canny / HoughLinesP
	gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
	blur_gray = cv2.GaussianBlur(gray, (15, 15), 0)
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)	
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

	
	rho = 1  # distance resolution in pixels of the Hough grid
	theta = np.pi / 90  # angular resolution in radians of the Hough grid
	threshold = 15  # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 200  # minimum number of pixels making up a line
	max_line_gap = 10  # maximum gap in pixels between connectable line segments
	line_image = np.copy(image) * 0  # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
	longestLine = 0
	
	print(len(lines))
	if (len(lines) > 0):
		for line in lines:
			for x1,y1,x2,y2 in line:
				# we only need one line. but it has be a vertical line.
				if (((x2-x1)+10 < (y2-y1))):
					if (longestLine < np.sqrt((x2-x1)**2 + (y2-y1)**2)):
						longestLine = np.sqrt((x2-x1)**2 + (y2-y1)**2)
						offsideLine = (x1,y1,x2,y2)

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4))

	# init arrays to store the players after it was made sure that it is indeed a player
	team1Player = np.array([])
	team2Player = np.array([])
	# store the position of the last man of each team on the pitch
	team1PlayerPositionX = 0
	team2PlayerPositionX = 0
	team1PlayerPositionY = 0
	team2PlayerPositionY = 0
	
	# iterate over the found players
	for i in range(len(weights)):
		(x, y, w, h) = rects[i]
		cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
		# throw all results under a threshold away
		if (weights[i] > 0.7):
			
			# crop an rectangle of the result
			crop = orig[y:y + h, x:x + w]
			
			# find the percentage of the definded colors
			hsvTeam1 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
			maskTeam1 = cv2.inRange(hsvTeam1, lower['team1'], upper['team1'])
			maskTeam1 = cv2.morphologyEx(maskTeam1, cv2.MORPH_OPEN, kernel)
			maskTeam1 = cv2.morphologyEx(maskTeam1, cv2.MORPH_CLOSE, kernel)
			percTeam1 = (maskTeam1>254).sum() / (len(maskTeam1)*len(maskTeam1[0]))
			
			print("Team1/white " + str((maskTeam1>254).sum() / (len(maskTeam1)*len(maskTeam1[0]))))
			
			hsvGreen = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
			maskGreen = cv2.inRange(hsvGreen, lower['green'], upper['green'])
			maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_OPEN, kernel)
			maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_CLOSE, kernel)
			percGreen = (maskGreen>254).sum() / (len(maskGreen)*len(maskGreen[0]))
			
			print("Green " + str((maskGreen>254).sum() / (len(maskGreen)*len(maskGreen[0]))))
			
			hsvTeam2 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
			maskTeam2 = cv2.inRange(hsvTeam2, lower['team2'], upper['team2'])
			maskTeam2 = cv2.morphologyEx(maskTeam2, cv2.MORPH_OPEN, kernel)
			maskTeam2 = cv2.morphologyEx(maskTeam2, cv2.MORPH_CLOSE, kernel)
			percTeam2 = (maskTeam2>254).sum() / (len(maskTeam2)*len(maskTeam2[0]))
			
			print("Team2/red " + str((maskTeam2>254).sum() / (len(maskTeam2)*len(maskTeam2[0]))))
			
			print(" ")
			
			# apply a blue rectange on all found spots
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
			
			# use the green to filter out fans
			if (percGreen < 0.7 and percGreen > 0.1):
				# sort players into the teams
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

	# calculate line parameters of the Offside / Origin Line from the two dots given from the HoughLinesP
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
	
	# draw the Origin if the Offside Line
	cv2.line(image, (t1y, t1x), (t2y,t2x), (100,240,80), 5)
	
	# draw Offside Line
	cv2.line(image, (o1y, o1x), (o2y,o2x), (30,120,220), 5)
	
	# show the result
	image = imutils.resize(image, height=720)
	cv2.imshow("Players and Offside", image)
	
	key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
		
camera.release()
cv2.destroyAllWindows()
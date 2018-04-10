# USAGE
# python detect.py -v/--video video

# Import the necessary packages
from scipy.optimize import fsolve
import numpy as np
import argparse
import imutils
import cv2


# Defined function for later intersection calculation
def f(xy, funcargs):
    x, y = xy
    z = np.array([y - funcargs[0][0]*x - funcargs[0][1], y - funcargs[0][2]*x - funcargs[0][3]])
    return z


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# Define shirt colors of the teams and the pitch, hsv color model
lower = {'team1': (0, 0, 180), 'green': (35, 50, 50), 'team2': (120, 40, 85)}  # team1 = white, team2 = red
upper = {'team1': (255, 40, 255), 'green': (60, 255, 255), 'team2': (186, 255, 255)}
# Thresholds for line detection
lowGrayThreshold = 50
highGrayThreshold = 150

# Do colorfiltering and morphing for detected players
def colorFiltering(image, lower_threshold, upper_threshold):
    hsvTeam = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    maskTeam = cv2.inRange(hsvTeam, lower_threshold, upper_threshold)
    maskTeam = cv2.morphologyEx(maskTeam, cv2.MORPH_OPEN, kernel)
    maskTeam = cv2.morphologyEx(maskTeam, cv2.MORPH_CLOSE, kernel)
    percTeam  = (maskTeam > 254).sum() / (len(maskTeam) * len(maskTeam[0]))
    return percTeam


# Initialize the HOG descriptor/person detector
# Interesting that most variables only support one parameter right now
winSize = (48, 96)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
# Use Daimler detector because he is trained to recognise people 48x96 pixels
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

# Initialize morph-kernel
kernel = np.ones((5,15),np.uint8)

# Parameters for HoughLinesP function
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 90  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 200  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments

# Blank arrays for off-side lane calculation
offsideLine = np.array([])
box5Line = np.array([])

# Read in video
vid_cap = cv2.VideoCapture(args["video"])

# Loop over the video, processing frames
while True:

    # Read frame from video
    (grabbed, frame) = vid_cap.read()
    ''' If we are viewing a video and we did not grab a frame,
    '   then we have reached the end of the video '''
    if args.get("video") and not grabbed:
        break

    ''' Load the image and resize it to improve detection accuracy
    '   Since the most of our source material is 720p and players are about
    '   1/5th of the picture we need to scale it larger to get to the 96px height.
    '   Copy image for later drawings '''
    t = int(frame.shape[1] * 1.4)
    image = imutils.resize(frame, width=t)
    orig = image.copy()

    # Line detection: gray image / fill lines / canny / HoughLinesP
    grayImg = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
    blurredGrayImg = cv2.GaussianBlur(grayImg, (15, 15), 0)
    cannyEdges = cv2.Canny(blurredGrayImg, lowGrayThreshold, highGrayThreshold)
    cannyEdges = cv2.morphologyEx(cannyEdges, cv2.MORPH_CLOSE, kernel)

    # Creating a blank to draw lines on
    line_image = np.copy(image) * 0

    ''' Run Hough on edge detected image
    '   Output "lines" is an array containing endpoints of detected line segments '''
    lines = cv2.HoughLinesP(cannyEdges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    box16Line = 0

    ''' Loop for initializing the offside line. '''
    if (lines is not None):
        for line in lines:
            for x1,y1,x2,y2 in line:
                # we only need one line. but it has be a vertical line.
                if (((x2-x1)+10 < (y2-y1))):
                    if (box16Line < np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)):
                        box16Line = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        offsideLine = (x1,y1,x2,y2)
                if (cannyEdges.shape[1] - x2 < 20):
                        box5Line = (x1, y1, x2, y2)

    # Calculate line parameters of the Offside / Origin Line from the two dots given from the HoughLinesP
    lineX = (offsideLine[3] - offsideLine[1]) / (offsideLine[2] - offsideLine[0])
    lineB =  offsideLine[1] - lineX*offsideLine[0]
    t1y = 0
    t1x = int(lineB)
    t2y = int(image.shape[1])
    t2x = int(lineX*image.shape[1] + lineB)

    ''' Calculate intersection of 16m box and 5m box. 
    '   Calculated point is used for turning offsideLine into right angle '''
    testX = (box5Line[3] - box5Line[1]) / (box5Line[2] - box5Line[0])
    testB = box5Line[1] - testX * box5Line[0]
    z = (lineX, lineB, testX, testB)

    intersect = fsolve(f, [1, 2], args=[z])

    # Init arrays to store the players after it was made sure that it is indeed a player
    team1Player = np.array([])
    team2Player = np.array([])

    # Store the position of the last man of each team on the pitch
    team1PlayerPositionX = 0
    team2PlayerPositionX = 100000000
    team1PlayerPositionY = 0
    team2PlayerPositionY = 0

    # Detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4))

    # Iterate over the found players
    for i in range(len(weights)):

        (x, y, w, h) = rects[i]

        # DEBUG: Show all detected pedestrians
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Throw all results under a threshold away
        if (weights[i] > 0.7):

            # Crop an rectangle of the result
            crop = orig[y:y + h, x:x + w]

            # Find the percentage of the definded colors
            # Team 1
            percTeam1 = colorFiltering(crop, lower['team1'], upper['team1'])

            # DEBUG:
            # print("Team1/white " + str((maskTeam1>254).sum() / (len(maskTeam1)*len(maskTeam1[0]))))

            # Grass
            percGreen = colorFiltering(crop, lower['green'], upper['green'])

            # DEBUG:
            # print("Green " + str((maskGreen>254).sum() / (len(maskGreen)*len(maskGreen[0]))))

            # Team2
            percTeam2 = colorFiltering(crop, lower['team2'], upper['team2'])

            # DEBUG:
            # print("Team2/red " + str((maskTeam2>254).sum() / (len(maskTeam2)*len(maskTeam2[0]))))

            # DEBUG: apply a blue rectange on all found spots
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Use the green to filter out fans
            if (percGreen < 0.7 and percGreen > 0.1):
                # Sort players into the teams
                if (percTeam1 > 0.03):
                    team2Player = np.append(team2Player, rects[i])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Calculate offside line from found player
                    lineX = (intersect[1] - (y + h)) / (intersect[0] - (w + x))
                    lineB = (y + h) - lineX*(x + w)
                    ro1y = 0
                    ro1x = int(lineB)
                    ro2y = int(image.shape[1])
                    ro2x = int(lineX*image.shape[1] + lineB)

                    if (team2PlayerPositionX > ro2x and ro2x > 0):
                        team2PlayerPositionX = ro2x
                        team2PlayerPositionY = ro1x
                else:
                    team1Player = np.append(team1Player, rects[i])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # draw the Origin if the Offside Line

    # DEBUG:
    # cv2.line(image, (t1y, t1x), (t2y,t2x), (100,240,80), 5)

    cv2.line(image, (0, team2PlayerPositionY), (int(image.shape[1]),team2PlayerPositionX), (120,120,220), 5)

    # DEBUG:
    # draw line on the 5 box
    #if (len(box5Line) > 0):
    #	cv2.line(image, (box5Line[0], box5Line[1]), (box5Line[2],box5Line[3]), (30,120,120), 5)

    # Show the result
    image = imutils.resize(image, height=720)
    cv2.imshow("Players and Offside", image)

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# Close Video
vid_cap.release()
cv2.destroyAllWindows()

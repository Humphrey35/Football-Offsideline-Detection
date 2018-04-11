import cv2
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
    help="path to the (optional) image file")
args = vars(ap.parse_args())

x_co = 0
y_co = 0
def on_mouse(event,x,y,flag,param):
  global x_co
  global y_co
  if(event==cv2.EVENT_MOUSEMOVE):
    x_co=x
    y_co=y

cv2.namedWindow('camera')

#capture = cv.CaptureFromCAM(0)
while True:
#    src = cv.QueryFrame(capture)
    src = cv2.imread(args["image"])
    #src = cv2.GaussianBlur(src,(5,5),0)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cv2.setMouseCallback("camera",on_mouse);
    s=hsv[y_co,x_co]
    print("H:",s[0],"      S:",s[1],"       V:",s[2])
    #cv2.PutText(src,str(s[0])+","+str(s[1])+","+str(s[2]), (x_co,y_co),font, (55,25,255))
    cv2.imshow("camera", hsv)
    if cv2.waitKey(10) == 27:
        break
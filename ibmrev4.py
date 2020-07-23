# import the necessary packages
from imutils import contours
import numpy as np
import imutils
import cv2
import datetime

def show_image(name, img):
    disp = cv2.resize(img, (800, 600))
    cv2.imshow(name, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

loc = ["cs1", "cs2", "pd1", "pd2"]
agl = [4.467, -6.949, 4.289, -1.253]
y = [1078, 1098, 685, 1000]
itcp = [1.7, 1.7, 1.7, 1.7]

boundaries = (30, 255)


lower = np.array(boundaries[0], dtype = "uint8")
upper = np.array(boundaries[1], dtype = "uint8")

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

num = 0
width = 1
    

def ibm(key, yb, angle, intercept):
    filename = ("pengukuran/" + key + "-2020-7-12.jpg")
    img = cv2.imread(filename)
    img = cv2.fastNlMeansDenoisingColored(img,None,15,15,7,21)
    b, g, r = cv2.split(img)
    if key != "pd2":
        b = cv2.multiply(b, 0.9)
    gray = cv2.subtract(g, b)
    mask = cv2.inRange(gray, lower, upper)
    gray = cv2.bitwise_and(gray, gray, mask = mask)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.multiply(gray, 2)
    final = gray
    final = imutils.rotate(final, angle)
    final = cv2.rectangle(final, (0,yb), (1600, 1200), (0,0,0), -1)
    if key == "cs1":
        final = cv2.rectangle(final, (1450,0), (1600, 1200), (0,0,0), -1)
    elif  key == "cs2":
        final = cv2.rectangle(final, (0,0), (200, 1200), (0,0,0), -1)
    elif  key == "pd2":
        final = cv2.rectangle(final, (1450,0), (1600, 1200), (0,0,0), -1)
    edged = cv2.Canny(final, 0, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    ldim = 0
    for c in cnts:
        if cv2.contourArea(c) < 900:
            continue
        orig = img.copy()
        orig = imutils.rotate(orig, angle)
        x, y, w, h = cv2.boundingRect(c)
        box = ((x,y+h),(x,y),(x+w,y),(x+w,y+h))
        cv2.rectangle(orig, (int(x),int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        tlbl = (tlblX, tlblY)
        trbr = (trbrX, trbrY)
        (cx, cy) = midpoint(tlbl, trbr)
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)
        dimA = h * 0.0227 + intercept
        dt = datetime.datetime.now()
        dts = str(dt.year) + "/" + str(dt.month) + "/" + str(dt.day) + " " + str(dt.hour) + ":" + str(dt.minute) + ":" + str(dt.second)
        cv2.putText(orig, "H = {:.1f}cm".format(dimA),
                    (int(cx) + 10, int(cy) - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        orig = imutils.rotate(orig, -angle)
        orig = orig[80:orig.shape[0]-80, 60:orig.shape[1]-60]
        orig = cv2.resize(orig, (1600, 1200))
        cv2.putText(orig, dts,
                    (0, orig.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 2)
        if dimA < ldim:
            continue
        ldim = dimA
    savefile = (key + "-display.jpg")
    cv2.imwrite(savefile, orig)
    print("{} cm".format(ldim))
            
for location in loc:
    ibm(location, y[num], agl[num], itcp[num])
    num = num + 1

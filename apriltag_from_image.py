import cv2
from pupil_apriltags import Detector

img = cv2.imread("apriltag-real.png")
grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


at_detector = Detector(families="tagCircle21h7")

tags = at_detector.detect(grey_image)

print(f"FOUND {len(tags)} TAGS!")

for detection in tags:
    print(detection)
    p1 = tuple(map(int, detection.corners[3]))
    p2 = tuple(map(int, detection.corners[2]))
    p3 = tuple(map(int, detection.corners[1]))
    p4 = tuple(map(int, detection.corners[0]))
    cv2.line(img, p1, p2, (0,255,0), 3)
    cv2.line(img, p2, p3, (0,255,0), 3)
    cv2.line(img, p3, p4, (0,255,0), 3)
    cv2.line(img, p4, p1, (0,255,0), 3)
    print("----------------------------------------------")

cv2.imshow("Highlighted Apriltag", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# print(at_detector.detect(grey_image))


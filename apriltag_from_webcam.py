import cv2
from pupil_apriltags import Detector

CONFIDENCE_THRESHOLD = 10

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

# UNCOMMENT THESE LINES TO INCREASE RESOLUTION
# DESIRED_WIDTH = 1920
# DESIRED_HEIGHT = 1080
# vc.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
# vc.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

# Setup a detector - we use the circle21h7 family of tags, other than that default settings
at_detector = Detector(families="tagCircle21h7")


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()

    # UNCOMMENT THESE LINES TO INCREASE RESOLUTION
    # actual_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    # actual_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"Requested resolution: {DESIRED_WIDTH}x{DESIRED_HEIGHT}, camera resolution: {actual_width}x{actual_height}")
else:
    rval = False

while rval:
    # Take the frame and make it greyscale - no need to use all the colours
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass the frame into the detector
    tags = at_detector.detect(grey_image)

    # Remove all tags that are below a certain confidence value
    tags = [tag for tag in tags if tag.decision_margin > CONFIDENCE_THRESHOLD]

    for detection in tags:
        print(detection)
        # Convert the pixel values it an integer
        p1 = tuple(map(int, detection.corners[3]))
        p2 = tuple(map(int, detection.corners[2]))
        p3 = tuple(map(int, detection.corners[1]))
        p4 = tuple(map(int, detection.corners[0]))

        # Draw lines around the apriltag - note we are using the frame and not the greyscale image here since we want to see locations on the colour preview
        cv2.line(frame, p1, p2, (0,255,0), 3)
        cv2.line(frame, p2, p3, (0,255,0), 3)
        cv2.line(frame, p3, p4, (0,255,0), 3)
        cv2.line(frame, p4, p1, (0,255,0), 3)
        cv2.putText(frame, str(detection.tag_id), p2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the colour preview
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
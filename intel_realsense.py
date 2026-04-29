import pyrealsense2 as rs
import cv2
import numpy as np
from pupil_apriltags import Detector

CONFIDENCE_THRESHOLD = 10

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

at_detector = Detector(families="tagCircle21h7", nthreads=8)

def draw_identifiers(frame, corners, box_colour=(0, 255, 0), text_colour=(255, 0, 0), centre_colour=(0, 0, 255)):
    p1 = tuple(map(int, corners[3]))
    p2 = tuple(map(int, corners[2]))
    p3 = tuple(map(int, corners[1]))
    p4 = tuple(map(int, corners[0]))
    
    # Calculate the center of the tag using numpy
    centroid = np.mean(corners, axis=0).astype(int)

    cv2.line(frame, p1, p2, box_colour, 3)
    cv2.line(frame, p2, p3, box_colour, 3)
    cv2.line(frame, p3, p4, box_colour, 3)
    cv2.line(frame, p4, p1, box_colour, 3)
    cv2.putText(frame, str(detection.tag_id), p2, cv2.FONT_HERSHEY_SIMPLEX, 1, text_colour, 2)
    cv2.circle(frame, centroid, 2, centre_colour, 3)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # Get color frame
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Normalize depth image to 0-255 using a FIXED range
        # Adjust the max value (5000) based on your camera's typical working distance
        depth_clipped = np.clip(depth_image, 0, 5000)
        depth_colormap = (depth_clipped / 5000 * 255).astype(np.uint8)

        # Apply a heatmap colormap
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

        # Pass the frame into the detector
        tags = at_detector.detect(grey_image)

        # Remove all tags that are below a certain confidence value
        tags = [tag for tag in tags if tag.decision_margin > CONFIDENCE_THRESHOLD]

        for detection in tags:            
            # Calculate the center of the tag using numpy
            centroid = np.mean(detection.corners, axis=0).astype(int)
            
            # Find the distance of the center of the apriltag
            dist = depth_frame.get_distance(centroid[0], centroid[1])
            print(f"The tag with id {detection.tag_id} is {dist:.3f} meters away")


            draw_identifiers(color_image, detection.corners)
            draw_identifiers(depth_colormap, detection.corners)
            
        cv2.imshow('Camera Feed', color_image)
        cv2.imshow('Depth Feed', depth_colormap)
        
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
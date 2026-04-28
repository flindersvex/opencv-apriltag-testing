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

at_detector = Detector(families="tagCircle21h7")

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

        # Normalize depth image to 0-255 for colormap
        # Clip depth values (e.g., 0-5000mm) and scale to 8-bit
        depth_colormap = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = np.uint8(depth_colormap)

        # Apply a heatmap colormap
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

        # Pass the frame into the detector
        tags = at_detector.detect(grey_image)

        # Remove all tags that are below a certain confidence value
        tags = [tag for tag in tags if tag.decision_margin > CONFIDENCE_THRESHOLD]

        for detection in tags:
            # print(detection)
            # Convert the pixel values it an integer
            p1 = tuple(map(int, detection.corners[3]))
            p2 = tuple(map(int, detection.corners[2]))
            p3 = tuple(map(int, detection.corners[1]))
            p4 = tuple(map(int, detection.corners[0]))
            
            # Calculate the center of the tag using numpy
            centroid = np.mean(detection.corners, axis=0).astype(int)
            
            dist = depth_frame.get_distance(centroid[0], centroid[1])
            print(f"The tag with id {detection.tag_id} is {dist:.3f} meters away")


            # Draw lines around the apriltag - note we are using the colour and not the greyscale image here since we want to see locations on the colour preview
            cv2.line(color_image, p1, p2, (0,255,0), 3)
            cv2.line(color_image, p2, p3, (0,255,0), 3)
            cv2.line(color_image, p3, p4, (0,255,0), 3)
            cv2.line(color_image, p4, p1, (0,255,0), 3)
            cv2.putText(color_image, str(detection.tag_id), p2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw lines around the apriltag - note we are using the depth and not the greyscale image here since we want to see locations on the depth preview
            cv2.line(depth_colormap, p1, p2, (0,255,0), 3)
            cv2.line(depth_colormap, p2, p3, (0,255,0), 3)
            cv2.line(depth_colormap, p3, p4, (0,255,0), 3)
            cv2.line(depth_colormap, p4, p1, (0,255,0), 3)
            cv2.putText(depth_colormap, str(detection.tag_id), p2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(color_image, centroid, 2, (0, 0, 255), 3)
            cv2.circle(depth_colormap, centroid, 2, (0, 0, 255), 3)
            
        cv2.imshow('Camera Feed', color_image)
        cv2.imshow('Depth Feed', depth_colormap)
        
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
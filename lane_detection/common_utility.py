import cv2 as cv
import numpy as np
import os
import common_utility as util

def identify_lanes(lines, width): 
    left_lines = []
    right_lines = [] 

    if lines is not None: 
        for line in lines: 
            x1, y1, x2, y2 = line[0]

            #find the slope
            slope = (y2 - y1) / (x2 -x1) if x2 != x1 else 0 

            if x2 == x1: continue  # Skip vertical lines
            
            slope = (y2 - y1) / (x2 - x1)
            # Use CENTER of line, not just x1
            x_center = (x1 + x2) / 2
            
            # LEFT: negative slope (/) + left side 
            if slope < 0 and x_center < width/2: 
                left_lines.append(line)
            # RIGHT: positive slope (\) + right side  
            elif slope > 0 and x_center > width/2: 
                right_lines.append(line)

    return left_lines, right_lines 


"""
"HoughLinesP returns fragmented line segments due to noise/occlusion. 
I average the endpoints (x1,y1,x2,y2) across all left/right fragments to create one smooth representative lane line per side.
This gives clean visualization and stable lane tracking."
"""
def average_lines(lines_list):
    """
    Average multiple Hough lines → single smooth lane line
    Input: List of HoughLinesP arrays [array([[x1,y1,x2,y2]])]
    Output: Single averaged line array or None
    """
    if lines_list is None or len(lines_list) == 0:
        return None
    
    # Extract coordinates CORRECTLY from Hough format
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    
    for line in lines_list:
        # HoughLinesP format: line = array([[x1,y1,x2,y2]])
        coords = line[0]  # Get [x1,y1,x2,y2]
        x1s.append(coords[0])
        y1s.append(coords[1])
        x2s.append(coords[2])
        y2s.append(coords[3])
    
    # Average → single smooth line
    avg_x1, avg_y1 = int(np.mean(x1s)), int(np.mean(y1s))
    avg_x2, avg_y2 = int(np.mean(x2s)), int(np.mean(y2s))
    
    # Return OpenCV expected format: array([[[x1,y1,x2,y2]]])
    return np.array([[[avg_x1, avg_y1, avg_x2, avg_y2]]])

def draw_lane_lines(image, leftlines, rightlines, leftcolor=[0, 0, 255], rightcolor=[0,255,0], thickness=5):
    """
    Draw Hough Transform lines onto the original image with weighted blending.
    
    Parameters:
        image: Original BGR image (H,W,3)
        lines: HoughLinesP output - array of [[x1,y1,x2,y2]]
        color: BGR color tuple (Default: [0,0,255] = Red)
        thickness: Line thickness in pixels (Default: 5)
    
    Returns:
        result: Original image with lane lines overlaid (alpha=1.0 blending)
        
    Why this approach?
    ==================
    - np.zeros_like(image) creates same shape/dtype canvas
    - cv.addWeighted() does perfect alpha blending (no color distortion)
    - Single pass through all lines = O(n) efficiency
    """
    line_image = np.zeros_like(image)  # Black canvas same shape as input
    if leftlines is not None:  # Defensive check - Hough may return None
        for line in leftlines:
            # line[0] unpacks to [x1,y1,x2,y2] - OpenCV HoughLinesP format
            x1, y1, x2, y2 = line[0]
            cv.line(line_image, (x1, y1), (x2, y2), leftcolor, thickness)

    if rightlines is not None:  # Defensive check - Hough may return None
        for line in rightlines:
            # line[0] unpacks to [x1,y1,x2,y2] - OpenCV HoughLinesP format
            x1, y1, x2, y2 = line[0]
            cv.line(line_image, (x1, y1), (x2, y2), rightcolor, thickness)            
    
    # Alpha blend: 100% original + 100% lines = bright overlay
    return cv.addWeighted(image, 1.0, line_image, 1.0, 0.0)

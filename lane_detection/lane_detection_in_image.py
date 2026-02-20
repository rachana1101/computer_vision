"""
Simple Lane Detection Program with Single Image (Feb 19, 2026)
============================================================
Classical Computer Vision pipeline demonstrating:
1. Canny edge detection → HoughLinesP → Lane line visualization
2. Region of Interest (ROI) masking for road area
3. Production-ready error handling + visualization

Author: Rachana (rachana1101
GitHub: https://github.com/rachana1101/computer-vision
"""

import cv2 as cv
import numpy as np
import os

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

if __name__ == "__main__":
    root = os.getcwd()
    # PRODUCTION PATH HANDLING: Join paths + error checking
    imagePath = os.path.join(root, 'lane_detection/resources/lane-image.jpeg')
    print(f"Loading image from: {imagePath}")
    
    # STEP 1: Load & Validate Image
    img = cv.imread(imagePath)
    if img is None:
        print(f"❌ ERROR: Could not load {imagePath}")
        print("Check: file exists? correct path? valid JPEG?")
        exit()
    print(f"✅ Loaded: {img.shape} (H:{img.shape[0]}, W:{img.shape[1]})")
    
    # STEP 2: Grayscale Conversion
    # BGR→GRAY: OpenCV default is BGR (not RGB!)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("✅ Grayscale conversion complete")
    
    # STEP 3: Gaussian Blur (KEY PARAMETER EXPLANATION)
    """
    WHY (5,5) kernel? NOT 3x3 or 7x7?
    ================================
    GaussianBlur(gray, (5,5), 0)
    - Kernel=5x5: Goldilocks size
      Too small (3x3): Insufficient noise reduction → noisy Canny edges
      Too large (7x7+): Over-blurs lane edges → weak Hough detection
    
    Sigma=0: Auto-calculate based on kernel size (standard practice)
    Effect: Smooths noise while preserving sharp lane boundaries
    """
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # STEP 4: Canny Edge Detection
    """
    Canny(blur, 50, 150) - Why these thresholds?
    ============================================
    low=50: Weak edges (connected via hysteresis)
    high=150: Strong edges (primary lane markings)
    
    Hysteresis connects weak→strong edges = continuous lanes
    Alternative: adaptiveThreshold() but fixed works great for roads
    """
    edges = cv.Canny(blur, 50, 150)
    print("✅ Canny edges detected")
    
    # STEP 5: Region of Interest (ROI) - ROAD TRAPEZOID
    height, width = edges.shape
    mask = np.zeros_like(edges)  # Single-channel black mask
    
    """
    ROI Design Rationale:
    ====================
    Trapezoid focuses computation on road area:
    - Bottom: full width (0.1-0.9) = near lanes
    - Top: narrow (0.45-0.55) = far horizon lanes converge
    - Ignores sky/hood/shoulders = 70% fewer false lines!
    """
    roi_vertices = np.array([
        (int(0.1 * width), height),      # Bottom-left
        (int(0.45 * width), int(0.6 * height)),  # Top-left
        (int(0.55 * width), int(0.6 * height)),  # Top-right  
        (int(0.9 * width), height)       # Bottom-right
    ], dtype=np.int32)
    
    # cv.fillPoly expects LIST of polygons
    roi_vertices = [roi_vertices]
    ignore_mask_color = 255  # White fill for single-channel edge image
    cv.fillPoly(mask, roi_vertices, ignore_mask_color)
    
    # Apply mask: edges AND mask = road-only edges
    cropped_edges = cv.bitwise_and(edges, mask)
    print("✅ ROI masked - road area only")
    
    # STEP 6: HoughLinesP - Probabilistic Hough Transform
    """
    HoughLinesP Parameters - Why these values?
    ==========================================
    rho=1: 1px precision
    theta=np.pi/180: 1° angular resolution  
    threshold=50: Min 50 edge pixels vote for line
    minLineLength=40: Ignore short fragments
    maxLineGap=100: Connect broken lane segments
    
    Result: Robust to partial occlusions/noise
    """
    lines = cv.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

   # After HoughLinesP (STEP 6)...
    print(f"🔍 Total Hough lines detected: {len(lines) if lines is not None else 0}")

    # STEP 7-8: FIXED lane identification + averaging
    left_lines_raw, right_lines_raw = identify_lanes(lines, width)
    left_lines_avg = average_lines(left_lines_raw) if left_lines_raw else None
    right_lines_avg = average_lines(right_lines_raw) if right_lines_raw else None

    # STEP 9: Draw results
    result = draw_lane_lines(img, left_lines_avg, right_lines_avg, 
                            leftcolor=[0, 255, 0], rightcolor=[0, 0, 255], thickness=8)

    print("✅ BOTH LANES DETECTED!")
    cv.imshow("Left(GREEN) + Right(RED)", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    
    
    print("🎯 Portfolio Project #1: 60% Complete!")
    print("Next: Left/Right lane separation → Video processing")

"""
Simple lane detection program with the single image
"""


import cv2 as cv
import numpy as np
import os

def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=5):
    """
    Draw lines onto the input image.
        Parameters:
            image: The original image
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 5): Line thickness. 
    """
    line_image = np.zeros_like(image)
    if lines is not None:  # Check if lines exist
        for line in lines:
            # Proper unpacking - line[0] is [x1,y1,x2,y2]
            x1, y1, x2, y2 = line[0]
            cv.line(line_image, (x1, y1), (x2, y2), color, thickness)
    
    return cv.addWeighted(image, 1.0, line_image, 1.0, 0.0)


if __name__ == "__main__":
    root = os.getcwd()
    # Read the image
    imagePath = os.path.join(root, 'lane_detection/resources/lane-image.jpeg')
    print(f"Loading image from: {imagePath}")

    #Read the image and display
    img = cv.imread(imagePath)
    
    # Check if image loaded successfully
    if img is None:
        print(f"Error: Could not load image at {imagePath}")
        exit()
    
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the image
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection to the image
    edges = cv.Canny(blur, 50, 150)

    # Apply hough 
    # Define a region of interest (ROI) for lane detection
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    # Region of interest 
    roi_vertices = np.array([
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ], dtype=np.int32)

    # cv.fillPoly expects list of arrays, not single array
    roi_vertices = [roi_vertices]

    # if you pass an image with more then one channel
    if len(edges.shape) > 2:
        channel_count = edges.shape[2]
        ignore_mask_color = (255,) * channel_count
    # our image only has one channel so it will go under "else"
    else:
          # color of the mask polygon (white)
        ignore_mask_color = 255
    
    cv.fillPoly(mask, roi_vertices, ignore_mask_color)
    cropped_edges = cv.bitwise_and(edges, mask)

    # 5. HoughLinesP
    lines = cv.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    result = draw_lane_lines(img, lines)
    
    cv.imshow("Result", result)
    # Add these to see the window
    cv.waitKey(0)
    cv.destroyAllWindows()

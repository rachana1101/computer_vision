"""
Document Scanner
===============================

Author: Rachana Gupta
Date: February 2026
"""

import cv2 as cv      
import os             
import matplotlib.pyplot as plt
import numpy as np    


def contours():
    """
    Main pipeline for red cross logo detection and geometric analysis.
    
    Steps:
    1. Load and preprocess image (grayscale, threshold)
    2. Find contours using morphological operations
    3. Calculate geometric properties (centroid, area, perimeter, hull, etc.)
    4. Visualize all results in 2x3 subplot grid
    """
    
    # === IMAGE LOADING ===
    root = os.getcwd()  # Get current working directory (script location)
    
    # Build cross-platform file path to test image
    imagePath = os.path.join(root, 'document_scanner/resources/book_page_1.jpeg')
    print(f"Loading image from: {imagePath}")

    # Load image as grayscale (single channel, 0-255 intensity values)
    gray = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Image not found: {imagePath}")
    

    #remove unwanted noise 
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    #For edge detection 
    edged = cv.Canny(blur, 50, 150)  # Edge detection
    #Thresholding 
    _, threshold = cv.threshold(edged, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


    # === PREPROCESSING: CREATE BINARY MASK ===
    # HIGH threshold=200 with INV creates WHITE cross on BLACK background
    # Pixels >200 (white bg) → BLACK (0)
    # Pixels <200 (dark red logo) → WHITE (255)
    _, threshold = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)
    
    # Dilation: Expand white regions, connect cross arms, fill small gaps
    kernel = np.ones((3,3), np.uint8)  # 3x3 structuring element
    threshold = cv.dilate(threshold, kernel)


    # === CONTOUR DETECTION ===
    # Find external contours only (ignores holes/nested contours)
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    # Calculate areas of all detected contours (debugging)
    areas = [cv.contourArea(c) for c in contours]
    print("Contour areas:", areas)
    
    # Use largest contour (main red cross logo)
    if contours:
        main_contour = max(contours, key=cv.contourArea)
    else:
        print("No contours found!")
        return


    # === VISUALIZATION SETUP: 2x3 GRID ===
    plt.figure(figsize=(18, 12))
    
    # Load original color image for drawing overlays
    orig_color = cv.imread(imagePath)


    # === PLOT 1: ORIGINAL GRAYSCALE IMAGE ===
    plt.subplot(231); plt.imshow(gray, cmap="gray")
    plt.title('1. Original Grayscale Image', fontsize=12, pad=10)
    plt.axis('off')


    # === PLOT 2: BINARY THRESHOLD MASK ===
    plt.subplot(232); plt.imshow(threshold, cmap="gray")
    plt.title('2. Binary Threshold (200, INV)', fontsize=12, pad=10)
    plt.axis('off')


    # === PLOT 3: ALL CONTOURS OVERLAY ===
    cv.drawContours(orig_color, contours, -1, (0, 255, 255), 2)  # Yellow contours
    plt.subplot(233); plt.imshow(orig_color)
    plt.title('3. All Contours Detected', fontsize=12, pad=10)
    plt.axis('off')


    # === PLOT 4: CONTOUR CENTROID (CENTER OF MASS) ===
    moments = cv.moments(main_contour)  # Statistical moments of contour
    if moments['m00'] != 0:  # Avoid division by zero
        Cx = int(moments['m10']/moments['m00'])  # X-coordinate of centroid
        Cy = int(moments['m01']/moments['m00'])  # Y-coordinate of centroid
        
        plt.subplot(234)
        plt.imshow(gray, cmap="gray")
        plt.plot(Cx, Cy, 'r*', markersize=20, markeredgewidth=3, markeredgecolor='white')
        plt.title(f'4. Centroid (Cx={Cx}, Cy={Cy})', fontsize=12, pad=10)
        plt.axis('off')


    # === PLOT 5: CONVEX HULL + POLYGON APPROXIMATION ===
    # Contour perimeter for approximation
    perimeter = cv.arcLength(main_contour, True)  # True = closed contour
    
    # Approximate polygon: Reduce points while preserving shape
    epsilon = 0.01 * perimeter  # Approximation accuracy (1% of perimeter)
    approx = cv.approxPolyDP(main_contour, epsilon, True)  # Closed polygon
    
    # Convex hull: Smallest convex polygon containing contour
    hull = cv.convexHull(main_contour)
    
    plt.subplot(235)
    plt.imshow(gray, cmap="gray")
    # Plot hull (red solid line) and approximation (cyan dashed line)
    hull_points = hull[:, 0, :]  # Extract (x,y) coordinates
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=3, label='Convex Hull')
    plt.plot(approx[:, 0, 0], approx[:, 0, 1], 'c--', linewidth=2, label='Polygon Approx')
    plt.title(f'5. Hull & Approx (ε={epsilon:.0f})', fontsize=12, pad=10)
    plt.legend()
    plt.axis('off')


    # === PLOT 6: BOUNDING BOX + AREA/PERIMETER STATS ===
    x, y, w, h = cv.boundingRect(main_contour)  # Axis-aligned bounding rectangle
    
    # Create copy for drawing
    image_with_box = gray.copy()
    cv.rectangle(image_with_box, (x, y), (x+w, y+h), 0, 3)  # White rectangle
    
    plt.subplot(236)
    plt.imshow(image_with_box, cmap="gray")
    plt.title(f'6. Bounding Box\nArea: {cv.contourArea(main_contour):.0f}\nPerim: {perimeter:.0f}', 
              fontsize=12, pad=10)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    print(f"\n=== CONTOUR PROPERTIES ===")
    print(f"Area: {cv.contourArea(main_contour):.1f} pixels")
    print(f"Perimeter: {perimeter:.1f} pixels")
    print(f"Bounding box: ({w} x {h}) at ({x}, {y})")
    print(f"Aspect ratio: {w/h:.2f}")


if __name__ == '__main__':
    contours()
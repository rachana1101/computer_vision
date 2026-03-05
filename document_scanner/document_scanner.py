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

def four_point_transform(image, pts):
    """Transform document to flat scan"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], 
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

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
    
    orig_color = cv.imread(imagePath)  # Load COLOR version for transform
    ratio = gray.shape[0] / 500.0    # Resize ratio for display
    gray = cv.resize(gray, (int(gray.shape[1] / ratio), 500))
    

    #remove unwanted noise 
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    #For edge detection 
    edged = cv.Canny(blur, 50, 150)  # Edge detection
    #Thresholding 
    _, threshold = cv.threshold(edged, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Dilation: Expand white regions, connect cross arms, fill small gaps
    kernel = np.ones((3,3), np.uint8)  # 3x3 structuring element
    threshold = cv.dilate(threshold, kernel)


    # === CONTOUR DETECTION ===
    # Find external contours only (ignores holes/nested contours)
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    if contours: 
        # Find largest 4-sided contour (document)
        screenCount = None 
        for c in contours: 
            perimeter = cv.arcLength(c, True) #computes the perimeter (total boundary length) of the contour, assuming it's closed.
            """
            Simplifies the contour's shape using the Douglas-Peucker algorithm
            0.02 * peri sets the approximation precision (2% of perimeter; smaller values keep more detail).

            Lower values (e.g., 0.01) retain more detail; higher (e.g., 0.05) simplify more aggressively. 
            The 0.02 value is a practical default from document scanning tutorials for balancing noise rejection and shape fidelity.

            True ensures closed polygon handling.
            Result approx is a reduced set of vertices.            
            """
            approx = cv.approxPolyDP(c, 0.02 * perimeter, True) 
        
            #if len(approx) == 4: identifies quadrilaterals (e.g., rectangles or warped documents under perspective).
            if (len(approx)) == 4: # to make sure that polygon is square or rectangle 
            
                """
                #stores the first matching contour; break stops after the first hit
                #(prioritizing largest implicitly if contours are pre-sorted by area).
                """
                screenCount = approx.reshape(4, 2) * ratio 
                break

            #main_contour = screenCount #assigns it for later use, like perspective transform to straighten the document.

    # ADD at end (Plot 7):
    if screenCount is not None:
        warped = four_point_transform(orig_color, screenCount)
        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        warped = cv.threshold(warped, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    
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

    plt.subplot(234)
    plt.imshow(warped, cmap="gray")
    plt.title('7. SCANNED DOCUMENT', fontsize=14, pad=10, fontweight='bold')
    plt.axis('off')

    plt.show()
    
if __name__ == '__main__':
    contours()
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
    # === IMAGE LOADING ===
    root = os.getcwd()
    imagePath = os.path.join(root, 'document_scanner/resources/IMG_5494.jpg')
    print(f"Loading image from: {imagePath}")

    orig_color = cv.imread(imagePath)
    gray = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Image not found: {imagePath}")
    
    ratio = gray.shape[0] / 500.0
    gray = cv.resize(gray, (int(gray.shape[1] / ratio), 500))

    # === DOCUMENT SCANNING PIPELINE ===
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    edged = cv.Canny(blur, 50, 250)
    
    # Find contours on EDGES (not threshold)
    # === CONTOUR DETECTION + FULL PAGE FALLBACK ===
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    screenCnt = None
    largest_area = 0
    page_candidates = []

    for c in cnts:
        area = cv.contourArea(c)
        if area < 5000 or area > 0.7 * gray.shape[0] * gray.shape[1]:  # 30%-70% of image
            continue
            
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.08 * peri, True)
        
        if len(approx) == 4:
            # NEW: Page shape filters
            pts = approx.reshape(4, 2)
            widths = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
            heights = [np.linalg.norm(pts[i] - pts[(i+2)%4]) for i in range(4)]
            aspect_ratio = max(widths) / max(heights)
            
            # Page = rectangular (not square logo)
            if 1.2 < aspect_ratio < 3.0:  
                page_candidates.append((area, approx))
                print(f"📄 Page candidate: {area:.0f}px² (AR: {aspect_ratio:.1f})")

    # Pick best page
    if page_candidates:
        page_candidates.sort(key=lambda x: x[0], reverse=True)
        screenCnt = page_candidates[0][1].reshape(4, 2) * ratio
        print(f"✅ SELECTED page: {page_candidates[0][0]:.0f}px²")
    else:
        # Fallback: Largest rectangular contour
        print("🔄 Using largest rectangle")
        screenCnt = max(cnts, key=cv.contourArea)
        peri = cv.arcLength(screenCnt, True)
        screenCnt = cv.approxPolyDP(screenCnt, 0.08 * peri, True).reshape(4, 2) * ratio

    if screenCnt is not None:
        print(f"📍 Document corners: {screenCnt}")
        print(f"Shape: {screenCnt.shape}")
    
    
# === STEP 2: PERSPECTIVE + THRESHOLDING FOR SCAN ===
    if screenCnt is not None:
        warped = four_point_transform(orig_color, screenCnt)
        warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        _, final_scan = cv.threshold(warped_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # === PLOTTING (MOVE ALL PLOTS INSIDE HERE) ===
        plt.figure(figsize=(20, 12))
        
        plt.subplot(231); plt.imshow(gray, cmap="gray")
        plt.title('1. Grayscale'); plt.axis('off')
        
        plt.subplot(232); plt.imshow(blur, cmap="gray")
        plt.title('2. Blur'); plt.axis('off')
        
        plt.subplot(233); plt.imshow(edged, cmap="gray")
        plt.title('3. Canny Edges'); plt.axis('off')
        
        display_img = cv.resize(orig_color, (gray.shape[1], gray.shape[0]))
        cv.drawContours(display_img, [screenCnt.astype(int)], -1, (0, 255, 0), 3)
        plt.subplot(234); plt.imshow(cv.cvtColor(display_img, cv.COLOR_BGR2RGB))
        plt.title('4. Document Detected ✅'); plt.axis('off')
        
        plt.subplot(235); plt.imshow(warped_gray, cmap="gray")
        plt.title('5. Perspective Corrected'); plt.axis('off')
        
        plt.subplot(236); plt.imshow(final_scan, cmap="gray")
        plt.title('6. FINAL SCAN ✅', fontsize=16, fontweight='bold'); plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        cv.imwrite('perfect_scan.jpg', final_scan)
        print("🎉 Day 1 COMPLETE!")
    else:
        print("❌ No document contour found")

    
if __name__ == '__main__':
    contours()
import cv2 as cv
import numpy as np



if __name__ == "__main__":
    # Read the image
    img = cv.imread('test_image.jpg')
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the image
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection to the image
    edges = cv.Canny(blur, 50, 150)
    # Define a region of interest (ROI) for lane detection
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
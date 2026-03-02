import cv2 as cv
import numpy as np
import os
import common_utility as util

def process_frame(frame):
    """Run the full lane pipeline on a single BGR frame → returns annotated frame."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    roi_vertices = np.array([
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ], dtype=np.int32)
    cv.fillPoly(mask, [roi_vertices], 255)
    cropped_edges = cv.bitwise_and(edges, mask)

    lines = cv.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    left_lines_raw, right_lines_raw = util.identify_lanes(lines, width)
    left_lines_avg = util.average_lines(left_lines_raw) if left_lines_raw else None
    right_lines_avg = util.average_lines(right_lines_raw) if right_lines_raw else None

    result = util.draw_lane_lines(
        frame,
        left_lines_avg,
        right_lines_avg,
        leftcolor=[0, 255, 0],
        rightcolor=[0, 0, 255],
        thickness=8
    )
    return result, lines, left_lines_raw, right_lines_raw

if __name__ == "__main__":
    root = os.getcwd()
    videoPath = os.path.join(root, 'lane_detection/resources/solidWhiteRight.mp4')
    print(f"Loading video from: {videoPath}")

    cap = cv.VideoCapture(videoPath)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video {videoPath}")
        exit()

    # Optional: save output video
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video stream ended")
            break

        # Initialize writer once we know frame size
        if out is None:
            h, w, _ = frame.shape
            outPath = os.path.join(root, 'lane_detection/output/lane-video-output.mp4')
            out = cv.VideoWriter(outPath, fourcc, 20.0, (w, h))
            print(f"▶️ Writing output video to: {outPath}")

        result, lines, left_lines_raw, right_lines_raw = process_frame(frame)

        # (Optional) extract ML features per frame
        # You can adapt extract_ml_features to TAKE these values instead of
        # recomputing them from a global:
        # feats = extract_ml_features(lines, left_lines_raw, right_lines_raw)
        # print(feats)

        cv.imshow("Lane Detection (Video)", result)
        out.write(result)

        # Press 'q' to quit early
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("⏹ Stopped by user")
            break

    cap.release()
    if out is not None:
        out.release()
    cv.destroyAllWindows()

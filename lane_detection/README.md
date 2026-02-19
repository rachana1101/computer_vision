# Lane Detection: Canny → HoughLinesP → Robotics

Minimal Python pipeline for real-time lane detection using OpenCV. Designed for robotics navigation - outputs **lateral error** and **heading error** control signals from camera feed.

[![Lane Detection Demo](demo.gif)](demo.gif)

## Why This Pipeline Works

Lane detection extracts straight road lanes from camera images for robotics navigation by finding edges then mathematically fitting lines. Each step filters noise and focuses computation on relevant road features.

### 1. **Grayscale + Gaussian Blur**
- **Why**: RGB has too much data. Grayscale simplifies to intensity only.
- **Blur purpose**: Smooths pixel noise so Canny doesn't detect false edges from camera sensor grain and removes unwanted smaller objects.
- **Without blur**: Edges explode into thousands of fragments.

### 2. **Canny Edge Detection**
- **Why**: Finds sharp intensity transitions. Lanes = bright white/yellow paint vs dark asphalt = strong gradients.
- **Canny advantage**: Suppresses weak edges, connects broken ones via hysteresis (`50, 150` thresholds).
- **Result**: Clean edge map vs basic Sobel.

### 3. **Region of Interest (ROI) Mask**
- **Why**: Full-frame edges = sky/trees/cars. Trapezoid focuses on road ahead only.
- **Impact**: Reduces Hough computation 70%+, eliminates non-road false lines.

### 4. **HoughLinesP (Probabilistic Hough Transform)**
**Key parameters**:
rho=1, theta=π/180 # 1px/1° resolution
threshold=50 # 50+ aligned edge points needed
minLineLength=40 # Ignore short fragments
maxLineGap=100 # Allow small breaks in lines

- **Why HoughLinesP**: Returns endpoints directly (draw-ready). Standard HoughLines needs polar→cartesian conversion.
- **Math**: Accumulator space voting - collinear edge points vote for same (ρ,θ).

### 5. **Left/Right Lane Classification + Averaging**
Left lanes: slope < 0 (diagonal "")
Right lanes: slope > 0 (diagonal "/")

```python
# polyfit(x,y,1) → y = mx + b (single smooth line per side)
m, b = np.polyfit(all_xs, all_ys, 1)

Why classify: Separate left/right for independent averaging.

Why average: Raw Hough = noisy fragments. polyfit smooths flicker across frames.

lateral_error_px = lane_center_x - image_center_x
heading_error_rad = avg_lane_angle - vertical

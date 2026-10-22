import cv2
import os
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random  # for glitch effect randomness

print("OpenCV version:", cv2.__version__)

# Determine the correct path to haarcascade files
# In OpenCV 5.0, cv2.data.haarcascades points to the data directory,
# and the actual XML files are in a subdirectory named 'haarcascades'
base_data_path = cv2.data.haarcascades
haarcascade_path = os.path.join(base_data_path, 'haarcascades')
cascade_file = os.path.join(haarcascade_path, 'haarcascade_frontalface_default.xml')

print(f"Loading cascade from: {cascade_file}")

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_file)

if face_cascade.empty():
    print("ERROR: Failed to load cascade classifier from", cascade_file)
    print("Trying alternative paths...")
    # Try without the subdirectory (for OpenCV 4.x compatibility)
    cascade_file_alt = os.path.join(base_data_path, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_file_alt)
    if face_cascade.empty():
        print("ERROR: Could not load cascade classifier from alternative path either.")
        exit(1)
    else:
        print("SUCCESS: Loaded cascade from alternative path.")
else:
    print("SUCCESS: Cascade classifier loaded.")

# Download the hand landmarker model if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"Downloading hand landmarker model to {model_path}...")
    model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        exit(1)

# Initialize MediaPipe Hands using Tasks API with VIDEO mode for better temporal consistency
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       min_hand_detection_confidence=0.7,  # Higher confidence for more reliable detection
                                       min_hand_presence_confidence=0.7,
                                       min_tracking_confidence=0.7,
                                       running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize webcam
# Use DirectShow backend on Windows for better compatibility
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Tracking toggles
face_enabled = True
hand_enabled = True
# Filter modes: 0=None, 1=Grayscale, 2=ASCII Art, 3=Glitch
filter_mode = 1  # start with grayscale as default

# Point manipulation state
points_enabled = False          # Whether to show and manipulate points
points_visible = False          # Whether points are currently visible
selected_point_index = -1       # Index of currently selected point (-1 = none)
point_positions = []            # List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
point_radius = 8                # Radius for drawing points
point_color = (0, 255, 255)     # Yellow color for points (BGR)
point_threshold = 20            # Distance threshold for point selection (pixels)

# For video timestamp
frame_count = 0
fps = 30  # Assumed FPS for timestamp calculation

print("Starting ASCII Camera. Press 'q' to quit.")
print("Press 'f' to toggle face detection (ghostly white boxes), 'h' to toggle hand landmarks.")
print("Press keys 0-3 to select filter: 0=None, 1=Grayscale, 2=ASCII Art, 3=Glitch")
print("Make 'L' shape with thumb and index finger on both hands to toggle point manipulation!")
print("Pinch thumb+index to move points, make L shape again to hide/show points.")
print("Note: Make sure you have adequate lighting and your hands are visible to the camera.")

def is_thumb_index_extended(landmarks):
    """
    Check if thumb and index finger are extended (making an L shape)
    Simple approach: Check if distance between thumb tip and index tip is large
    relative to hand size (wrist to middle finger MCP)
    """
    try:
        # Get key points
        wrist = np.array([landmarks[0].x, landmarks[0].y])      # Wrist
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])  # Thumb tip
        index_tip = np.array([landmarks[8].x, landmarks[8].y])  # Index tip
        middle_mcp = np.array([landmarks[9].x, landmarks[9].y]) # Middle finger MCP

        # Calculate hand reference size (wrist to middle MCP)
        hand_size = np.linalg.norm(wrist - middle_mcp)

        # Calculate thumb-index distance
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)

        # Thumb and index are considered extended if their distance is
        # significantly larger than typical curled position
        # When curled, thumb-index distance is roughly hand_size * 0.3
        # When extended (L shape), it's roughly hand_size * 0.8-1.2
        extended_threshold = hand_size * 0.5

        return thumb_index_dist > extended_threshold
    except:
        return False

def get_hand_landmark_points(landmarks, frame_shape):
    """Extract thumb and index tip points in pixel coordinates"""
    try:
        h, w, c = frame_shape
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

        return (thumb_x, thumb_y), (index_x, index_y)
    except:
        return None, None


def is_thumb_index_pinch(landmarks, threshold=0.05):
    """
    Check if thumb and index finger are pinched (close together)
    Returns True if distance between thumb and index tips is below threshold
    """
    try:
        # Get normalized coordinates (0-1 range)
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_tip = np.array([landmarks[8].x, landmarks[8].y])

        # Calculate distance between thumb and index tips
        distance = np.linalg.norm(thumb_tip - index_tip)

        # Return True if distance is below threshold (normalized)
        return distance < threshold
    except:
        return False


def apply_ascii_art(frame, mask):
    """
    Convert the masked region to pure ASCII art (white characters on black background)
    and overlay it onto the original frame outside the mask.
    Returns the frame with ASCII art visible inside the mask and original frame outside.
    """
    # Create a blank layer for ASCII art with black background (same size as input)
    ascii_layer = np.zeros_like(frame)

    # Get coordinates where mask is non-zero
    ys, xs = np.where(mask == 255)
    if len(ys) == 0:
        return frame  # Return original frame if no mask

    # Bounding box of the mask region
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Crop the region to the bounding box for efficiency
    region = frame[y_min:y_max+1, x_min:x_max+1]
    region_mask = mask[y_min:y_max+1, x_min:x_max+1]

    # Convert region to grayscale for brightness calculation
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Define ASCII characters from darkest to lightest
    ascii_chars = "@%#*+=-:. "
    # We'll map 0-255 to 0-(len(ascii_chars)-1)

    # Downsample region to a smaller grid for ASCII rendering
    # Choose a cell size that gives reasonably sized characters
    cell_w, cell_h = 6, 12  # approximate width/height of a character in pixels
    # Compute number of cells
    cols = max(1, region.shape[1] // cell_w)
    rows = max(1, region.shape[0] // cell_h)

    # Resize region to cell dimensions (using area interpolation for good downscaling)
    small = cv2.resize(region, (cols, rows), interpolation=cv2.INTER_AREA)
    small_gray = cv2.resize(gray_region, (cols, rows), interpolation=cv2.INTER_AREA)

    # For each cell, compute average brightness and ASCII char
    for i in range(rows):
        for j in range(cols):
            brightness = int(small_gray[i, j])
            # Map brightness to ascii index
            char_index = int((brightness / 255) * (len(ascii_chars) - 1))
            ch = ascii_chars[char_index]

            # Position in original frame
            text_x = x_min + j * cell_w
            text_y = y_min + i * cell_h + cell_h  # baseline approx

            # Draw the character on the ASCII layer using fixed white color
            # for pure ASCII art effect (like ASCII AI generator)
            cv2.putText(ascii_layer, ch, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Combine: where mask is 255 (the quadrilateral region), use the ASCII layer;
    # where mask is 0, use the original frame.
    # Expand mask to 3 channels for broadcasting
    mask_3ch = mask[:, :, None] == 255
    output_frame = np.where(mask_3ch, ascii_layer, frame)

    return output_frame

def apply_glitch(frame, mask):
    """
    Apply a simple glitch effect (RGB channel shift + noise) inside the mask.
    """
    glitch_frame = frame.copy()
    if np.count_nonzero(mask) == 0:
        return glitch_frame

    # Split channels
    b, g, r = cv2.split(glitch_frame)

    # Generate random shifts for each channel (within mask region)
    # We'll shift horizontally by a random amount for each channel
    shift_b = random.randint(-8, 8)
    shift_g = random.randint(-8, 8)
    shift_r = random.randint(-8, 8)

    # Roll each channel (shift) – note: np.roll wraps around, but we later mask
    b_shifted = np.roll(b, shift_b, axis=1)
    g_shifted = np.roll(g, shift_g, axis=1)
    r_shifted = np.roll(r, shift_r, axis=1)

    # Combine shifted channels
    shifted = cv2.merge([b_shifted, g_shifted, r_shifted])

    # Apply shifted pixels only where mask is active
    glitch_frame = np.where(mask[:,:,None] == 255, shifted, glitch_frame)

    # Add some random noise for extra glitchiness
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    glitch_frame = np.where(mask[:,:,None] == 255,
                            cv2.add(glitch_frame, noise),
                            glitch_frame)

    return glitch_frame

def apply_grayscale(frame, mask):
    """
    Convert the masked region to grayscale (existing effect).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return np.where(mask[:,:,None] == 255, gray_bgr, frame)

def apply_filter(frame, mask, mode):
    """
    Dispatch to the selected filter mode.
    mode: 0=None, 1=Grayscale, 2=ASCII Art, 3=Glitch
    """
    if mode == 0:
        return frame
    elif mode == 1:
        return apply_grayscale(frame, mask)
    elif mode == 2:
        return apply_ascii_art(frame, mask)
    elif mode == 3:
        return apply_glitch(frame, mask)
    else:
        return frame

# Main processing loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Process face detection if enabled
    if face_enabled:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Create overlay for ghostly white face boxes
        overlay = frame.copy()
        # Draw rectangle around the faces with ghostly white color
        for (x, y, w, h) in faces:
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 2)  # White rectangles
        # Apply the overlay with transparency for ghostly/smoke effect
        alpha = 0.3  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Process hand detection if enabled
    if hand_enabled:
        # Convert the frame to rgb for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Detect hand landmarks with timestamp for video mode
        timestamp_ms = int(frame_count * (1000 / fps))
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        frame_count += 1

        # Process handedness and landmarks
        left_hand_points = None
        right_hand_points = None
        left_gesture = False
        right_gesture = False

        if detection_result.hand_landmarks:
            print(f"Detected {len(detection_result.hand_landmarks)} hands")  # Debug

            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Determine handedness - use MediaPipe's if available
                handedness = "Unknown"
                if (detection_result.handedness and
                    len(detection_result.handedness) > hand_idx and
                    len(detection_result.handedness[hand_idx]) > 0):
                    handedness = detection_result.handedness[hand_idx][0].category_name
                else:
                    # Fallback: estimate based on thumb vs index x position
                    try:
                        thumb_x = hand_landmarks[4].x
                        index_x = hand_landmarks[8].x
                        handedness = "Left" if thumb_x < index_x else "Right"
                    except:
                        handedness = "Unknown"

                print(f"Hand {hand_idx}: {handedness}")  # Debug

                # Check if thumb and index are extended (making L shape)
                gesture_detected = is_thumb_index_extended(hand_landmarks)
                print(f"  Gesture (thumb-index extended): {gesture_detected}")  # Debug

                if gesture_detected:
                    # Get thumb tip and index tip points
                    thumb_point, index_point = get_hand_landmark_points(hand_landmarks, frame.shape)
                    if thumb_point and index_point:
                        if handedness == "Left":
                            left_hand_points = thumb_point
                            right_hand_points = index_point  # Store index point for left hand
                            left_gesture = True
                            print(f"  Stored left hand: thumb={thumb_point}, index={index_point}")  # Debug
                        elif handedness == "Right":
                            left_hand_points = thumb_point  # Store thumb point for right hand
                            right_hand_points = index_point
                            right_gesture = True
                            print(f"  Stored right hand: thumb={thumb_point}, index={index_point}")  # Debug

        # Debug visualization: show what we detected
        debug_text = []
        if left_hand_points and right_hand_points:
            debug_text.append("Both hands: READY")
        elif left_hand_points:
            debug_text.append("Left hand only")
        elif right_hand_points:
            debug_text.append("Right hand only")

        if debug_text:
            cv2.putText(frame, " | ".join(debug_text), (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Handle L shape gesture to toggle point manipulation on/off
        if left_hand_points and right_hand_points and left_gesture and right_gesture:
            # Toggle points_enabled state
            points_enabled = not points_enabled
            if points_enabled:
                # Enable point manipulation: initialize points
                points_visible = True
                selected_point_index = -1

                # Initialize points as a rectangle based on current hand positions
                # Re-extract thumb and index points for both hands
                left_thumb_pt = left_index_pt = right_thumb_pt = right_index_pt = None
                if detection_result.hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        handedness = "Unknown"
                        if (detection_result.handedness and
                            len(detection_result.handedness) > hand_idx and
                            len(detection_result.handedness[hand_idx]) > 0):
                            handedness = detection_result.handedness[hand_idx][0].category_name

                        thumb_point, index_point = get_hand_landmark_points(hand_landmarks, frame.shape)

                        if handedness == "Left" and thumb_point and index_point:
                            left_thumb_pt = thumb_point
                            left_index_pt = index_point
                        elif handedness == "Right" and thumb_point and index_point:
                            right_thumb_pt = thumb_point
                            right_index_pt = index_point

                # Set initial 4 points (corners of rectangle)
                if all([left_thumb_pt, left_index_pt, right_thumb_pt, right_index_pt]):
                    # Convert points to numpy arrays for vector math
                    L_T = np.array(left_thumb_pt, dtype=np.float32)
                    L_I = np.array(left_index_pt, dtype=np.float32)
                    R_T = np.array(right_thumb_pt, dtype=np.float32)
                    R_I = np.array(right_index_pt, dtype=np.float32)

                    # Base vector between thumbs (distance between hands)
                    base_vec = R_T - L_T
                    base_len = np.linalg.norm(base_vec)
                    if base_len > 0:
                        unit_base = base_vec / base_len
                        # Perpendicular pointing upward (negative y in image coords)
                        perp_up = np.array([unit_base[1], -unit_base[0]], dtype=np.float32)
                    else:
                        # fallback: use horizontal direction
                        unit_base = np.array([1.0, 0.0], dtype=np.float32)
                        perp_up = np.array([0.0, -1.0], dtype=np.float32)

                    # Finger spread distances (thumb to index) for each hand
                    left_finger_dist = np.linalg.norm(L_I - L_T)
                    right_finger_dist = np.linalg.norm(R_I - R_T)
                    # Use average as rectangle height
                    height = (left_finger_dist + right_finger_dist) * 0.5  # average

                    # Compute rectangle corners
                    # Base from left thumb to right thumb
                    p1 = L_T
                    p2 = R_T
                    # Top side shifted upward by height * perp_up
                    p3 = R_T + perp_up * height
                    p4 = L_T + perp_up * height

                    point_positions = [tuple(p1.astype(int)), tuple(p2.astype(int)),
                                     tuple(p3.astype(int)), tuple(p4.astype(int))]
                else:
                    # Fallback: create a default rectangle in center of frame
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    size = 100
                    point_positions = [
                        (center_x - size, center_y - size),  # top-left
                        (center_x + size, center_y - size),  # top-right
                        (center_x + size, center_y + size),  # bottom-right
                        (center_x - size, center_y + size)   # bottom-left
                    ]

                print("Points enabled for manipulation")  # Debug
            else:
                # Disable point manipulation
                points_visible = False
                selected_point_index = -1
                # Keep point_positions for next enable
                print("Points disabled")  # Debug

        # If point manipulation is enabled, handle point selection/drawing each frame
        if points_enabled and points_visible:
            # Check for pinch gestures to move points
            if detection_result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    handedness = "Unknown"
                    if (detection_result.handedness and
                        len(detection_result.handedness) > hand_idx and
                        len(detection_result.handedness[hand_idx]) > 0):
                        handedness = detection_result.handedness[hand_idx][0].category_name

                    # Check for pinch gesture (thumb and index close together)
                    if is_thumb_index_pinch(hand_landmarks, threshold=0.04):  # Stricter threshold for pinch
                        # Get thumb tip position (we'll use this as the "pinch point")
                        thumb_point, _ = get_hand_landmark_points(hand_landmarks, frame.shape)
                        if thumb_point:
                            # Find closest point to this thumb position
                            if point_positions:
                                thumb_array = np.array(thumb_point)
                                distances = [np.linalg.norm(thumb_array - np.array(pos)) for pos in point_positions]
                                closest_idx = np.argmin(distances)

                                # If within threshold, select this point for dragging
                                if distances[closest_idx] < point_threshold:
                                    selected_point_index = closest_idx
                                else:
                                    selected_point_index = -1
                            else:
                                selected_point_index = -1
                        else:
                            selected_point_index = -1
                    else:
                        # No pinch, release any selected point
                        selected_point_index = -1

            # Update selected point position if dragging
            if selected_point_index != -1 and selected_point_index < len(point_positions):
                # Get current position of index finger (as pointer) for the selected hand
                # We'll use the first hand that's pinching
                index_point = None
                if detection_result.hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        if is_thumb_index_pinch(hand_landmarks, threshold=0.04):
                            _, index_point = get_hand_landmark_points(hand_landmarks, frame.shape)
                            break

                if index_point:
                    point_positions[selected_point_index] = index_point

            # Draw all points
            for i, point in enumerate(point_positions):
                color = (0, 255, 255) if i != selected_point_index else (0, 0, 255)  # Yellow normally, red when selected
                cv2.circle(frame, point, point_radius, color, -1)
                cv2.circle(frame, point, point_radius + 2, (255, 255, 255), 2)  # White outline

            # Create mask from the 4 points (quadrilateral)
            if len(point_positions) == 4:
                points_array = np.array(point_positions, dtype=np.int32)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points_array], 255)

                # Apply the selected filter
                frame = apply_filter(frame, mask, filter_mode)

                # Draw the polygon outline for visualization
                cv2.polylines(frame, [points_array], True, (0, 255, 0), 2)

                # Add text indicator
                filter_names = ["None", "Grayscale", "ASCII Art", "Glitch"]
                cv2.putText(frame, f"Filter: {filter_names[filter_mode]} - Drag points to adjust", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Instructions
                cv2.putText(frame, "Pinch thumb+index to move points, make L shape to toggle on/off", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw hand landmarks (manual drawing for consistent style)
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Define connections for hand (from mediapipe.solutions.hands)
                HAND_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                                              (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
                                              (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
                                              (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
                                              (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
                                              (5, 9), (9, 13), (13, 17), (5, 17)  # palms
                                              ])
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start_point = hand_landmarks[start_idx]
                        end_point = hand_landmarks[end_idx]
                        h, w, c = frame.shape
                        start_coords = int(start_point.x * w), int(start_point.y * h)
                        end_coords = int(end_point.x * w), int(end_point.y * h)
                        cv2.line(frame, start_coords, end_coords, (255, 255, 255), 1)
                # Draw landmarks
                for idx, landmark in enumerate(hand_landmarks):
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 2, (255, 255, 255), 1)
                # Draw finger tips as slightly larger outlined circles - different colors for tip identification
                # Thumb tip (4) - Red
                thumb_tip = hand_landmarks[4]
                h, w, c = frame.shape
                cx, cy = int(thumb_tip.x * w), int(thumb_tip.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red filled
                # Index tip (8) - Green
                index_tip = hand_landmarks[8]
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Green filled

    # Display the resulting frame
    cv2.imshow('ASCII Camera', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        face_enabled = not face_enabled
        status = "ON" if face_enabled else "OFF"
        print(f"Face detection: {status}")
    elif key == ord('h'):
        hand_enabled = not hand_enabled
        status = "ON" if hand_enabled else "OFF"
        print(f"Hand detection: {status}")
    elif key == ord('0'):
        filter_mode = 0
        print("Filter: None")
    elif key == ord('1'):
        filter_mode = 1
        print("Filter: Grayscale")
    elif key == ord('2'):
        filter_mode = 2
        print("Filter: ASCII Art")
    elif key == ord('3'):
        filter_mode = 3
        print("Filter: Glitch")

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
detector.close()
print("Finished.")
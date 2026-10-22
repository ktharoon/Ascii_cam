"""
ASCII Camera - Simple Real-time Full-Frame ASCII Art Feed
Enhanced with 16:9 resolution, fullscreen, and brighter/detailed output.
"""

import cv2
import numpy as np


def apply_ascii_art(frame, mask):
    """
    Convert the masked region to color ASCII art (matching original frame colors)
    and overlay it onto the original frame outside the mask.
    Returns the frame with ASCII art visible inside the mask and original frame outside.
    This is the standalone ASCII Camera Feed - full frame color ASCII.
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
    # Increase contrast and brightness to preserve details in bright and dark regions
    gray_region = cv2.convertScaleAbs(gray_region, alpha=2.2, beta=25)

    # Define ASCII characters from darkest to lightest
    # Original set: "@%#*+=-:. " (10 characters)
    # Extended with additional characters ordered from dark to light
    ascii_chars = "@%#*+=-:.WM$&8B@QO0ZCLUXJYuunxrqphkbdpwamIzwiI!l;:;,^\"'`." + " "

    # Target cell size in pixels - original detail (6x12)
    target_cell_w, target_cell_h = 6, 12

    # Compute number of cells based on target size (ensuring at least 1)
    cols = max(1, region.shape[1] // target_cell_w)
    rows = max(1, region.shape[0] // target_cell_h)

    # For each cell, compute exact pixel boundaries to cover the entire region
    for i in range(rows):
        # Calculate y range for this row
        y_start = int(i * region.shape[0] / rows)
        y_end = int((i + 1) * region.shape[0] / rows)
        for j in range(cols):
            # Calculate x range for this column
            x_start = int(j * region.shape[1] / cols)
            x_end = int((j + 1) * region.shape[1] / cols)

            # Extract the cell region
            cell = region[y_start:y_end, x_start:x_end]
            if cell.size == 0:
                continue

            # Convert cell to grayscale for brightness calculation
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            # Compute average brightness (mean)
            brightness = cv2.mean(gray_cell)[0]
            # Increase brightness to make image brighter and more visible
            brightness = min(255, int(brightness) + 45)

            # Map brightness to ascii index (inverted so bright areas get dense chars)
            char_index = int(((255 - brightness) / 255) * (len(ascii_chars) - 1))
            ch = ascii_chars[char_index]

            # Compute average color of the cell (BGR)
            color_mean = cv2.mean(cell)[:3]  # Returns (B, G, R, maybe alpha) - we take first 3
            color = tuple(int(c) for c in color_mean)  # Ensure integers

            # Position in original frame (top-left of cell)
            text_x = x_min + x_start
            text_y = y_min + y_start + target_cell_h  # baseline approx (using target height)

            # Draw the character on the ASCII layer using the cell's average color
            # Slightly larger font scale for better visibility
            cv2.putText(ascii_layer, ch, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Combine: where mask is 255 (the quadrilateral region), use the ASCII layer;
    # where mask is 0, use the original frame.
    # Expand mask to 3 channels for broadcasting
    mask_3ch = mask[:, :, None] == 255
    output_frame = np.where(mask_3ch, ascii_layer, frame)

    return output_frame


def main():
    print("OpenCV version:", cv2.__version__)

    # Initialize webcam - Use DirectShow backend on Windows for better compatibility
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set 16:9 resolution (1920x1080 - Full HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Check if webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {actual_width}x{actual_height}")

    # Create fullscreen window
    cv2.namedWindow('ASCII Camera', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('ASCII Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Starting ASCII Camera Feed. Press 'q' to quit, 'f' to toggle fullscreen.")

    # Main processing loop for ASCII Camera Feed
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Create mask for entire frame (all white) - full frame ASCII
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # Apply ASCII art to entire frame
        frame = apply_ascii_art(frame, mask)

        # Display the frame
        cv2.imshow('ASCII Camera', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            # Toggle fullscreen
            is_fullscreen = cv2.getWindowProperty('ASCII Camera', cv2.WND_PROP_FULLSCREEN)
            if is_fullscreen == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty('ASCII Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen: OFF")
            else:
                cv2.setWindowProperty('ASCII Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen: ON")

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    print("ASCII Camera Feed stopped.")


if __name__ == "__main__":
    main()
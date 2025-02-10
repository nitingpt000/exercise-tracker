import cv2

def find_available_cameras(max_index=10):
    """
    Attempts to open camera indices from 0 to max_index-1.
    Returns a list of indices for which a camera was successfully opened.
    """
    available_cams = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read one frame to be sure the camera is functional.
            ret, frame = cap.read()
            if ret:
                print(f"[INFO] Camera found at index {i}")
                available_cams.append(i)
            else:
                print(f"[WARNING] Camera at index {i} opened but no frame was captured.")
            cap.release()  # Don't forget to release the resource!
        else:
            print(f"[INFO] No camera at index {i}")
    
    return available_cams

if __name__ == '__main__':
    # Adjust max_index if you expect more than 10 cameras.
    camera_indices = find_available_cameras(max_index=10)
    print("\nDetected camera indices:", camera_indices)

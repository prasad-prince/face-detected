import cv2

print("Testing webcam access...")

cap = None
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ Webcam found on index {i}")
        ret, frame = cap.read()
        if ret:
            print(f"✓ Successfully captured frame from camera {i}")
            print(f"Frame shape: {frame.shape}")
        else:
            print(f"✗ Could not capture frame from camera {i}")
        cap.release()
    else:
        print(f"✗ No camera on index {i}")

print("\nIf no cameras found:")
print("- Check if webcam is connected")
print("- Ensure no other apps are using the camera")
print("- Grant camera permissions in Windows Settings > Privacy & security > Camera")
print("- Try running as administrator")
print("- Update webcam drivers in Device Manager")
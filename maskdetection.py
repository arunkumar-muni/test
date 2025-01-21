import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

overlay_path = r'/home/arun/Downloads/download (1).jpeg'
overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

if overlay is None:
    print(f"Error: Unable to load the overlay image from '{overlay_path}'")
    exit()

if overlay.shape[2] != 4:
    alpha_channel = np.ones((overlay.shape[0], overlay.shape[1]), dtype=np.uint8) * 255
    overlay = np.dstack([overlay, alpha_channel])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )

    for (x, y, w, h) in faces:
        overlay_resized = cv2.resize(overlay, (w, int(w * overlay.shape[0] / overlay.shape[1])))

        overlay_h, overlay_w, _ = overlay_resized.shape
        y_offset = y
        x_offset = x

        if y_offset < 0:
            overlay_resized = overlay_resized[-y_offset:, :, :]
            y_offset = 0

        y1, y2 = y_offset, y_offset + overlay_resized.shape[0]
        x1, x2 = x_offset, x_offset + overlay_resized.shape[1]

        if x2 > frame.shape[1]:
            overlay_resized = overlay_resized[:, :frame.shape[1] - x_offset, :]
            x2 = frame.shape[1]

        if y2 > frame.shape[0]:
            overlay_resized = overlay_resized[:frame.shape[0] - y_offset, :, :]
            y2 = frame.shape[0]

        alpha = overlay_resized[:, :, 3:] / 255.0
        frame[y1:y2, x1:x2] = (alpha * overlay_resized[:, :, :3] + (1 - alpha) * frame[y1:y2, x1:x2])

    cv2.imshow('Face Mask Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





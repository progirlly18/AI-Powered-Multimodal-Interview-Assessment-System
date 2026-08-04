import cv2

from emotion_detector import detect_emotion
from eye_contact import detect_eye_contact
from head_pose import detect_head_pose

print("Starting Interview Analyzer...")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

while True:

    ret, frame = cap.read()

    print("Frame received:", ret)

    if not ret:
        print("Failed to read frame.")
        break

    emotion, confidence, bbox = detect_emotion(frame)
    eye = detect_eye_contact(frame)
    pose = detect_head_pose(frame)

    print(emotion, eye, pose)

    if bbox:
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(frame, f"Emotion: {emotion}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(frame, f"Eye: {eye}", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.putText(frame, f"Head: {pose}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)

    cv2.imshow("Interview Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
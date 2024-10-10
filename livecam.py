import cv2

# Load pre-trained face and helmet detection models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
helmet_model = cv2.dnn.readNet('helmet_detection_model.weights', 'helmet_detection_model.cfg')

# Open camera feed
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each detected face, check for helmet
    for (x, y, w, h) in faces:
        # Assuming you have a function detect_helmet() that takes a frame and returns True if helmet is detected
        if detect_helmet(frame):
            # Draw green rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Draw red rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display frame
    cv2.imshow('Helmet Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

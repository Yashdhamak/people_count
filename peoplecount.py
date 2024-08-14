import cv2
import os
import datetime
import json

# Load history from file
def load_history(file_path='people_history.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

# Save history to file
def save_history(history, file_path='people_history.json'):
    with open(file_path, 'w') as file:
        json.dump(history, file, indent=4)

# Update history with the current time for a detected person
def update_history(history, person_id):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history[person_id] = current_time

# Detect people in the frame using a simple pre-trained model (HOG + SVM)
def detect_people(frame, hog):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, _ = hog.detectMultiScale(gray, winStride=(8,8))
    people_count = len(boxes)
    return people_count

def main():
    # Initialize HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Camera source (0 for the default webcam, replace with your IP camera URL if needed)
    cap = cv2.VideoCapture(0)

    # Load existing history
    history = load_history()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people in the frame
        people_count = detect_people(frame, hog)
        print(f"Detected People: {people_count}")

        # Update history for each detected person (simplified as counting for now)
        if people_count > 0:
            update_history(history, "Person_ID")

        # Save updated history
        save_history(history)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




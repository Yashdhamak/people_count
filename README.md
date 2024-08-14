People Count With History from Camera
This project is a simple Python application that detects and counts the number of people in a video feed (from a webcam or IP camera) using the Histogram of Oriented Gradients (HOG) + Support Vector Machine (SVM) method. It also keeps a history of detected individuals, including the last time they were seen, stored in a JSON file.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Features
People Detection: Utilizes OpenCV’s pre-trained HOG + SVM model to detect people in the video feed.
History Tracking: Tracks the last detection time for each person and stores this data in a JSON file.
Single File Execution: The entire project is contained in a single Python file for ease of deployment and execution.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Requirements
Python 3.6 or above
OpenCV 4.x
JSON (built-in Python module)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

How It Works

Detection: The application captures frames from the video source (default is the webcam). It converts each frame to grayscale and uses the HOG + SVM detector to identify people in the frame.

History Tracking: Whenever a person is detected, the current time is recorded in the people_history.json file. This file maintains a log of the last seen time for each detected person.

Display: The detected people are displayed in a window. The program prints the number of detected people in the console.

#This is a simple test script to verify that the DeepFace library is working correctly. It loads an image, analyzes it for emotions, and prints the results. Make sure to have a test image named "test.jpg" in the same directory as this script for it to work.
#You can run this script to check if the DeepFace library is properly installed and functioning. If there are any issues with loading the image or analyzing it, the script will print an error message. 
#Ensure that you have the necessary dependencies installed, such as OpenCV and DeepFace, before running this script. You can install them using pip if you haven't already:
#pip install deepface opencv-python
#Run this script in your Python environment, and it should output the analysis results for the emotions detected in the image.
#Run using the command: python test_deepface.py 
from deepface import DeepFace
import cv2

img = cv2.imread("test.jpg")

if img is None:
    print("ERROR: test.jpg not found or could not be loaded")
else:
    result = DeepFace.analyze(
        img,
        actions=['emotion'],
        enforce_detection=False,
        detector_backend='opencv'
    )
    print(result)
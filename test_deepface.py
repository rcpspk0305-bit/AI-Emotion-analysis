#This is a simple test script to verify that the DeepFace library is working correctly. It loads an image, analyzes it for emotions, and prints the results. Make sure to have a test image named "test.jpg" in the same directory as this script for it to work.
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
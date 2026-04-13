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
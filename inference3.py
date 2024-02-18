import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the model
model = load_model('./model.h5')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe Hands object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Declare the labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    # Read the video capture frame 
    ret, frame = cap.read()

    #get the shape of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

    # Resize the frame to fit within a 28x28 canvas while maintaining aspect ratio
  # Resize the frame to fit within a 28x28 canvas while maintaining aspect ratio
    # Resize the frame to have the desired width and height
    frame_resized = cv2.resize(frame_gray, (28, 28))

    # Pad the resized frame to make it 28x28
    pad_x = max(0, (28 - frame_resized.shape[1]) // 2)
    pad_y = max(0, (28 - frame_resized.shape[0]) // 2)
    frame_padded = cv2.copyMakeBorder(frame_resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)

    #Print the shape of frame_padded
    print("Shape of frame_padded:", frame_padded.shape)

    # Reshape and normalize the frame to range [0, 1]
    frame_reshaped = frame_padded.reshape((1, 28, 28, 1)).astype('float32') / 255.0

    # Predict the letter for the symbol
    prediction = model.predict(frame_reshaped)

    # Get the predicted letter
    predicted_character = labels_dict[np.argmax(prediction)]

    print("Predicted Letter:", predicted_character)



    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
    #             cv2.LINE_AA)
    time.sleep(1)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()

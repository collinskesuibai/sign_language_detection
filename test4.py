import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="./converted_model_q_quantized.tflite")
interpreter.allocate_tensors()

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
    # Read the video capture frame 
    ret, frame = cap.read()

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run Mediapipe hands detection
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand keypoints
            keypoints = np.array([[lmk.x * frame.shape[1], lmk.y * frame.shape[0]] for lmk in hand_landmarks.landmark]).astype(int)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the bounding box coordinates
            x1, y1 = np.min(keypoints[:, 0]), np.min(keypoints[:, 1])
            x2, y2 = np.max(keypoints[:, 0]), np.max(keypoints[:, 1])

            # Crop and resize the hand region to fit the input size of the model
            hand_crop = frame[y1:y2, x1:x2]
            hand_resized = cv2.resize(hand_crop, (28, 28))

            # Preprocess the input
            hand_input = np.expand_dims(hand_resized, axis=0).astype('float32') / 255.0

            # Perform inference using TensorFlow Lite
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], hand_input)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            # Get the predicted letter
            predicted_character = labels_dict[np.argmax(prediction)]

            # Display the predicted letter on the frame
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2
import tensorflow as tf

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65  # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
output_file = "predictions.txt"
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3, width)
cap.set(4, height)

#### LOAD THE TRAINED MODEL
model_path = "model_trained.h5"
try:
    model = tf.keras.models.load_model(model_path)
except OSError:
    print("Error: Model file not found!")
    exit()

#### CLASS LABELS (Define your class labels here)
class_labels = ["0 : ", " 1 : ", " 2: ", "3: ", "4: ", "5: ", "6: ", "7: ", "8:", "9: "]

#### PREPROCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# Open the file in write mode
with open(output_file, "w") as file:
    while True:
        success, imgOriginal = cap.read()
        if not success:
            print("Error: Failed to capture frame!")
            break

        img = np.asarray(imgOriginal)
        img = cv2.resize(img, (32, 32))
        img = preProcessing(img)
        img = img.reshape(1, 32, 32, 1)

        # Predict
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        prob_val = np.max(predictions)

        # Display prediction if probability is above threshold
        if prob_val > threshold:
            prediction_text = f"{class_labels[class_index]} {prob_val:.2f}"
            cv2.putText(imgOriginal, prediction_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            # Write the prediction to the file
            file.write(f"{prediction_text}\n")
        else:
            cv2.putText(imgOriginal, "No Number Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow("Original Image", imgOriginal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

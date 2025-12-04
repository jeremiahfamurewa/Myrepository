from PIL import Image
import tensorflow as tf
import cv2
import os
import numpy as np

DATASET_PATH = 'Datasets/'

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except:
        print("❌ Corrupted image")
        return

    model = tf.keras.models.load_model("image_classifier.h5")

    # ✅ Auto-detect input size from model
    input_shape = model.input_shape[1:3]

    img = cv2.imread(image_path)
    if img is None:
        print("❌ OpenCV failed to read image.")
        return

    img = cv2.resize(img, input_shape)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_names = sorted(os.listdir(DATASET_PATH))
    num_classes = len(class_names)

    if num_classes == 2:
        predicted_class = class_names[int(prediction[0][0] > 0.5)]
        confidence = prediction[0][0]
    else:
        index = np.argmax(prediction)
        predicted_class = class_names[index]
        confidence = prediction[0][index]

    print(f"✅ Prediction: {predicted_class}")
    print(f"✅ Confidence: {confidence:.2f}")

predict_image("Datasets/fish/ifsh3.jpg")

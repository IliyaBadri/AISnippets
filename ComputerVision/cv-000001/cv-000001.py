import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

capture_delay = 1000

model = MobileNetV2(weights='imagenet')
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

def get_object_name(predictions):
    return predictions[0][0][1]

def get_object_score(predictions):
    return predictions[0][0][2]

def detect_object(frame):
    frame_image = cv2.resize(frame, (224, 224))
    frame_image = image.img_to_array(frame_image)
    frame_image = np.expand_dims(frame_image, axis=0)
    frame_image = preprocess_input(frame_image)
    
    predictions = model.predict(frame_image)
    decoded_predictions = decode_predictions(predictions)

    return get_object_name(decoded_predictions), get_object_score(decoded_predictions)


def main():
    camera = cv2.VideoCapture(0)
    while True:
        captured, frame = camera.read()
        if captured:
            try:
                object_name, object_score = detect_object(frame)
                text = f"OBJECT: {object_name} ( {round(object_score * 100)}% )"
            except Exception as e:
                text = "Error"
                print(str(e))
            text_position = (10, 30)    
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
            cv2.imshow('Object detector (press q to exit)', frame)
            
        if cv2.waitKey(capture_delay) & 0xFF == ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# This code has been published by the AISnippets repository at https://github.com/IliyaBadri/AISnippets.  
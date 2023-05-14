import numpy as np
from tensorflow import lite,nn
from PIL import Image
from io import BytesIO
import base64
class_names=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Load TFLite model on import and allocate tensors.
def load ():
    interpreter = lite.Interpreter(model_path="res_net_quant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter,input_details,output_details

def predict_class(my_string):
    #Load Image
    bytes_decoded=base64.b64decode(my_string)
    image = Image.open(BytesIO(bytes_decoded))
    img=image.convert('RGB')
    # print("image type",type(img))
    new_img=img.resize((256,256))
    #Load Model
    interpreter,input_details,output_details=load()
    # Set input tensor.
    input_data = np.expand_dims(new_img, axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Run inference.
    interpreter.invoke()
    # Get output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    score = nn.softmax(output_data[0])
    # Print predicted class label.
    return class_names[np.argmax(score)],float(max(list(score)))*100,score.numpy().tolist()
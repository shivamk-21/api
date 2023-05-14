import numpy as np
from tensorflow import lite,nn
from PIL import Image
from io import BytesIO
import base64
#TODO:Update Class Names List
class_names=['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'healthy', 'leaf_spot', 'Common_rust_', 
             'Northern_Leaf_Blight', 'healthy', 'Black_rot', 'Esca_(Black_Measles)', 'healthy', 'Early_blight', 'Late_blight', 'healthy',
             'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mite', 
             'Target_Spot', 'Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'healthy']
classes={"apple":0,"corn":4,"grape":8,"potato":11,"tomato":14,"Select a crop":0}
# Load TFLite model on import and allocate tensors.
def load (plant):
    interpreter = lite.Interpreter(model_path=plant+".tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter,input_details,output_details

def predict_class(my_string,plant):
    #Load Image
    bytes_decoded=base64.b64decode(my_string)
    image = Image.open(BytesIO(bytes_decoded))
    img=image.convert('RGB')
    # print("image type",type(img))
    new_img=img.resize((256,256))
    #Load Model
    interpreter,input_details,output_details=load(plant)
    # Set input tensor.
    input_data = np.expand_dims(new_img, axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Run inference.
    interpreter.invoke()
    # Get output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    score = nn.softmax(output_data[0])
    # Print predicted class label.
    return class_names[classes[plant]:][np.argmax(score)],float(max(list(score)))*100,score.numpy().tolist()
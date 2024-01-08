import cv2
import onnxruntime as ort
import numpy as np
from datetime import datetime
from PIL import Image
import os 

os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(ort.get_device())

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def calculate_padding(length, multiple=16):
    padding = length % multiple
    if padding % multiple != 0:
        padding = multiple - length % multiple
    return padding    

def run_onnx_session(onnx_file_path):
    return ort.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def image_preprocessing(input_data_path):
    input_data = Image.open(input_data_path)
    width, height=  input_data.width, input_data.height
    
    input_data = np.asarray(input_data).astype(np.float32)
    cv2.normalize(input_data, input_data, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    pad_w, pad_h = calculate_padding(width, 16), calculate_padding(height, 16)
    resized_input_data = np.pad(input_data, pad_width=((0, 0+pad_h), (0, 0+pad_w)), mode='edge')
    resized_input_data = resized_input_data.reshape([1, 1, height+pad_h, width+pad_w])
    
    return resized_input_data, width, height

def image_postprocessing(ort_outs, width, height):
    ort_outs = np.array(ort_outs).squeeze()
    ort_outs = ort_outs[:height, :width]
    output_data = ort_outs * 255

    alpha = -0.4 # 기울기
    output_data = np.clip((1+alpha)*output_data - 128*alpha, 0, 255).astype(np.uint8)
    output_data = np.where(output_data < 0, 0, output_data)
    output_data = np.where(output_data > 255, 255, output_data)
    output_data = output_data.reshape([height, width]).astype('uint8')
    
    return output_data
    
if __name__ == "__main__":
    onnx_file_path = r''.replace('\\', '/') # directory of .onnx file
    input_path = r''.replace('\\', '/')     # directory of input filename or folder
    output_path = r''.replace('\\', '/')    # directory of output filename or folder
    
    ort_session = run_onnx_session(onnx_file_path)
    
    is_file = os.path.isfile(input_path)
    
    if is_file:
        input_data_path = input_path
        input_data, width, height = image_preprocessing(input_data_path)
        
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        
        output_data = image_postprocessing(ort_outs, width, height)
        imwrite(output_path, output_data)
    else:
        input_data_lst = os.listdir(input_path)
        print(len(input_data_lst))
        for step in range(len(input_data_lst)):
            input_data_path = os.path.join(input_path, input_data_lst[step])
            input_data, width, height = image_preprocessing(input_data_path)
        
            ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            ort_outs = ort_session.run(None, ort_inputs)
            
            output_data = image_postprocessing(ort_outs, width, height)
            imwrite(os.path.join(output_path, input_data_lst[step]), output_data)
    
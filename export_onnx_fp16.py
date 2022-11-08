import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16
import time  

def convert():
    input_onnx_model = '480x640.onnx'
    output_onnx_model = '480x640_fp16.onnx'
    
    onnx_model = onnxmltools.utils.load_model(input_onnx_model)
    onnx_model = convert_float_to_float16(onnx_model)
    onnxmltools.utils.save_model(onnx_model, output_onnx_model)

def test():
    import onnxruntime as ort
    t1 = time.time()
    model_path = '480x640_fp16.onnx'
    # model_path = '480x640.onnx'

    providers = [
        ('TensorrtExecutionProvider',
        'CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 6 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]

    session = ort.InferenceSession(model_path, providers=providers)
    print('load model done %.4f'%(time.time() - t1))
    del session 
    print("delete session")
    time.sleep(5)

if __name__=="__main__":
    test()


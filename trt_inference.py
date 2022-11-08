import tensorrt as trt
import pycuda.driver as cuda 
import pycuda.autoinit
import cv2
import numpy as np 
import time


class TRT_Model():
    def __init__(self, model_path: str):
        # load model 
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        f = open(model_path, "rb")
        engine = runtime.deserialize_cuda_engine(f.read())
        f.close()
        self.context = engine.create_execution_context()
        print("load model done")
        output = self.warmup(np.ones((480,640,3), dtype=np.int8))
        print(output.shape)

    def warmup(self, img: np.ndarray):
        dtype = type(img)
        output = np.empty((1,6300,6), dtype=dtype)

        d_input = cuda.mem_alloc(1*img.nbytes)
        d_output = cuda.mem_alloc(1*output.nbytes)

        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()
        
        cuda.memcpy_htod_async(d_input, img, stream)
        self.context.execute_async_v2(bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
        return output 

def preproc(image, input_size, dtype=np.float32):
    t1 = time.time()
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])

    # img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    # size = (int(img.shape[0] * r), int(img.shape[1] * r), 3)
    # img_tensor = F.interpolate(img_tensor, size=size, mode="nearest")
    # resized_img = img_tensor.squeeze().cpu().numpy()

    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_NEAREST,
    )
    t2 = time.time()

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    print("resize %.5f pad %.5f"%(t2-t1, time.time()-t2))
 
    return padded_img, r

if __name__=="__main__":
    model_path = "480x640_int8.engine"
    model = TRT_Model(model_path)
    
    ## test image
    img = cv2.imread("1.png").astype(np.uint8)
    img, ratio = preproc(img, (480,640), img.dtype)
    print(ratio)
    time.sleep(5)



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

model_path = "./yolov7.onnx"
precision = "fp16"

success = parser.parse_from_file(model_path)
config = builder.create_builder_config()
if precision == "fp16":
   config.set_flag(trt.BuilderFlag.FP16)
elif precision == "bf16":
    config.set_flag(trt.BuilderFlag.BF16)

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 35)
profile = builder.create_optimization_profile()


input_shape = [1, 3, 640, 640]
# input1_shape=[2]
# input_value=[512,512]

profile.set_shape("images", input_shape, input_shape, input_shape)


config.add_optimization_profile(profile)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

engine_file_path = "engine_file_path_" + precision
if os.path.exists(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
else:
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("save engine for later use.")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

# logging.debug("create execution context")
context = engine.create_execution_context()
# tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
# for tensor in tensor_names:
#     print("tensor的名字",tensor)
#     dtype=trt.nptype(engine.get_tensor_dtype(tensor))
#     print("tensor的dtype",dtype)
#     if engine.get_tensor_mode(tensor)==trt.TensorIOMode.INPUT:
#         if engine.is_shape_inference_io(tensor):
#             context.set_input_shape(tensor,)
context.set_input_shape("images",input_shape)


# context.set_binding_shape(0, input_shape)
# context.set_binding_shape(1,input1_shape)
# context.set_binding_shape(2,input1_shape)


h_input0 = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape("images")),dtype=np.float32)
h_input0 = np.zeros(h_input0.shape).astype(np.float32)
h_output =cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape("output")),dtype=np.float32)


d_input0 = cuda.mem_alloc(h_input0.nbytes)

d_output = cuda.mem_alloc(h_output.nbytes)
context.set_tensor_address("images",int(d_input0))
context.set_tensor_address("output",int(d_output))

stream = cuda.Stream()
for i in range(10):
    cuda.memcpy_htod_async(d_input0, h_input0, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
stream.synchronize()


import datetime
import time

stream.synchronize()
starttime = datetime.datetime.now()

for i in range(10):
    cuda.memcpy_htod_async(d_input0, h_input0, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

stream.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫

print(np.std(h_output), np.mean(h_output))

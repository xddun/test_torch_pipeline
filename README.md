----

step 1. install all requirements pakages

run `pip install -r requirements.txt`

----

step 2. train one simple model about MNIST

run `python train_and_test.py`

----

step 3. to run test() on single way

run `python load_model_test.py`

----

step 4. to export onnx and tensorrt

run `python export.py`

----

step 5. to run onnx inference 

run `python inference_onnx.py`

we will get this output of onnx inference:
```
预测的类别是否正确： True
模型的输出数值是： [[-2.2913797e+01 -1.7695108e+01 -1.3406548e+01 -1.5655020e+01
  -2.2644754e+01 -2.3998449e+01 -3.7063354e+01 -1.9073486e-06
  -2.1006844e+01 -1.4032949e+01]]
```

----

step 6. to run torch inference

run `python inference_torch.py`

```
预测的类别是否正确： True
模型的输出数值是： [-2.29137955e+01 -1.76951065e+01 -1.34065456e+01 -1.56550188e+01
 -2.26447525e+01 -2.39984474e+01 -3.70633507e+01 -2.50339190e-06
 -2.10068398e+01 -1.40329485e+01]
```
----

step 7. to run TensorRT inference

run `python inference_tensorrt.py`

oh, encounter a BUG !!

```angular2html
True
[06/29/2022-20:09:18] [TRT] [I] [MemUsageChange] Init CUDA: CPU +275, GPU +0, now: CPU 351, GPU 501 (MiB)
[06/29/2022-20:09:18] [TRT] [I] Loaded engine size: 0 MiB
[06/29/2022-20:09:18] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
[06/29/2022-20:09:21] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
[06/29/2022-20:09:21] [TRT] [E] 1: [executionContext.cpp::executeInternal::667] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
Traceback (most recent call last):
  File "/data/dong_xie/test_torch_pipeline/inference_tensorrt.py", line 45, in <module>
    print(y)
  File "/data/dong_xie/miniconda3/envs/py38/lib/python3.8/site-packages/torch/tensor.py", line 193, in __repr__
    return torch._tensor_str._str(self)
  File "/data/dong_xie/miniconda3/envs/py38/lib/python3.8/site-packages/torch/_tensor_str.py", line 383, in _str
    return _str_intern(self)
  File "/data/dong_xie/miniconda3/envs/py38/lib/python3.8/site-packages/torch/_tensor_str.py", line 358, in _str_intern
    tensor_str = _tensor_str(self, indent)
  File "/data/dong_xie/miniconda3/envs/py38/lib/python3.8/site-packages/torch/_tensor_str.py", line 242, in _tensor_str
    formatter = _Formatter(get_summarized_data(self) if summarize else self)
  File "/data/dong_xie/miniconda3/envs/py38/lib/python3.8/site-packages/torch/_tensor_str.py", line 90, in __init__
    nonzero_finite_vals = torch.masked_select(tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0))
RuntimeError: CUDA error: an illegal memory access was encountered
[06/29/2022-20:09:21] [TRT] [E] 1: [fusedConvActRunner.cpp::destroyFilterTexture::293] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [defaultAllocator.cpp::deallocate::42] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaStream::47] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [defaultAllocator.cpp::deallocate::42] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaDriverHelpers.cpp::operator()::29] Error Code 1: Cuda Driver (an illegal memory access was encountered)
[06/29/2022-20:09:21] [TRT] [E] 1: [cudaDriverHelpers.cpp::operator()::29] Error Code 1: Cuda Driver (an illegal memory access was encountered)

```







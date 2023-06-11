import torch
from torch.utils.cpp_extension import CUDA_HOME

print(torch.__version__) # pytorch version
print(torch.version.cuda) # cuda version
print(torch.cuda.is_available()) # cuda present or not
print(CUDA_HOME)




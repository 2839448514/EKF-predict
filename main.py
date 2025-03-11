import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
    print("Number of CUDA devices:", torch.cuda.device_count())
    current_device = torch.cuda.current_device()
    print("Current CUDA device:", torch.cuda.get_device_name(current_device))
else:
    print("CUDA is not available.")
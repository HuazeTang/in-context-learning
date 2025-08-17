#import torch
#print("Torch version:", torch.version)
#print("CUDA available:", torch.cuda.is_available())
#print("CUDA version:", torch.version.cuda)
#print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


#import jax
#import jax.numpy as jnp
#print("JAX version:", jax.__version__)
#print("JAX backend platform:", jax.lib.xla_bridge.get_backend().platform)


import torch
x = torch.rand(3, 3).cuda()  # 创建 GPU 张量
print(x)
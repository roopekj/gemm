from time import time

import torch

size = 16384

start = time()
for i in range(350):
    A, B = [
        torch.randn([size, size], dtype=torch.float16, device="cuda") for _ in range(2)
    ]
    C = A @ B
end = time()
print(end - start)

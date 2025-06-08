from time import sleep, time

import cutlass
import numpy as np

size = 16384
plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
A, B = [np.ones((size, size), dtype=np.float16) for _ in range(2)]
C, D = [np.zeros((size, size), dtype=np.float32) for _ in range(2)]

for _ in range(10):
    plan.run(A, B, C, D)

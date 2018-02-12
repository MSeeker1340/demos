"""
Module: matmul

Contains the class MatMulKernel which constructs and compiles the parametrized 
matrix multiplication kernel. Also contains the test function test_kernel.
"""

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

class MatMulKernel:
    """
    The parametrized matrix multiplication kernel
    
    Fields:
    dtype -- floating point data type, can be numpy.float32 or numpy.float64.
    tile_width -- the width of tiles.
    loop_unroll -- boolean. Indicates whether or not to unroll the inner loops.
    src -- the generated source code for the kernel.
    kernelfunc -- the function handle for the compiled kernel.

    Methods:
    gen_source -- generate src using template.cu. Called by the constructor.
    compile -- compile and load kernel.
    matmul -- main interface to compute M*N using the compiled kernel.
    """
    def __init__(self, dtype, tile_width, loop_unroll, compile=True):
        """
        The constructor calls gen_source to set up src. If compile is set to 
        True (by default), also compiles the generated source code and load 
        binary onto the GPU.
        """
        self.dtype = dtype
        self.tile_width = tile_width
        self.loop_unroll = loop_unroll
        self.gen_source()
        if compile:
            self.compile() # This step requires an initialized CUDA context
    
    def gen_source(self):
        """
        Generate the source code using `str.format` as template engine.
        The template kernel is read from template.cu.
        """
        if self.dtype == np.float32:
            real_t = 'float'
            fzero = '0.0f'
        elif self.dtype == np.float64:
            real_t = 'double'
            fzero = '0.0'
        else:
            raise ValueError('Datatype of {0} not supported.'.format(dtype))

        # Read template
        with open('template.cu', 'r') as f:
            template = f.read()

        # Generate the inner loop (or unrolled loop)
        if self.loop_unroll:
            loop = ''.join(["""Pvalue += Ms[ty][{k}] * Ns[{k}][tx];
        """.format(k=k) for k in range(self.tile_width)])
        else:
            loop = """for (int k = 0; k < {TW}; ++k)
        {{
            Pvalue += Ms[ty][k] * Ns[k][tx];
        }}""".format(TW = self.tile_width)

        self.src = template.format(real=real_t, fzero=fzero, TW=self.tile_width, loop=loop)

    def compile(self, **kwargs):
        """
        Compile src and load onto GPU. Function handle for the compiled module 
        is stored in kernelfunc. Optional kerword arguments are passed to the 
        SourceModule interface as specified by PyCUDA.

        Requires an initialized CUDA context to run.
        """
        mod = SourceModule(self.src, **kwargs)
        self.kernelfunc = mod.get_function("matmul")

    def matmul(self, M, N, timed=False):
        """
        Apply the compiled kernel to compute M*N and return the result. 
        Optionally, report GPU execution time (in miliseconds) if timed is set 
        to True.

        M, N and P are all assumed to be square matrices of the same dimension 
        and whose dtype is compatible with that of the kernel.
        """
        n = np.shape(M)[0]
        P = np.empty((n, n), dtype=self.dtype) # initialize empty P
        block_dim = (self.tile_width, self.tile_width, 1)
        grid_width = int(np.ceil(n / float(self.tile_width)))
        grid_dim = (grid_width, grid_width, 1)
        if timed:
            start = cuda.Event()
            end = cuda.Event()
            start.record()
            self.kernelfunc(
                cuda.In(M), cuda.In(N), cuda.Out(P), np.int32(n),
                block=block_dim, grid=grid_dim)
            end.record()
            end.synchronize()
            milisecs = start.time_till(end)
            return P, milisecs
        else:
            self.kernelfunc(
                cuda.In(M), cuda.In(N), cuda.Out(P), np.int32(n),
                block=block_dim, grid=grid_dim)
            return P

def test_kernel(n, dtype=np.float64, tile_width=8, loop_unroll=False):
    """
    Test numerical accuracy of the matrix multiplication kernel
    
    Note that because of the way CPU and GPU handles floating point arithmetic 
    (rounding modes, FFMA, etc) there might be significant differences, 
    especially in single precision mode.
    """
    M = np.random.randn(n, n).astype(dtype)
    N = np.random.randn(n, n).astype(dtype)
    MN = M.dot(N)
    kernel = MatMulKernel(dtype=dtype, tile_width=tile_width, loop_unroll=loop_unroll)
    P = kernel.matmul(M,N)
    err = np.max(np.abs((P - MN) / MN))
    print("Error between CPU and GPU result: {:e}".format(err))

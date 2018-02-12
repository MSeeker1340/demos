__global__ void matmul({real} *M, {real} *N, {real} *P, int Width)
{{
    // Compute M * N and store result in P
    // M and N are Width * Width matrices
    __shared__ {real} Ms[{TW}][{TW}];
    __shared__ {real} Ns[{TW}][{TW}];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = blockIdx.y * {TW} + ty;
    int Col = blockIdx.x * {TW} + tx;

    {real} Pvalue = {fzero};
    for (int ph = 0; ph < ceil(Width / ({real}){TW}); ++ph)
    {{
        // Cooperatively load tile into shared memory
        if (Row < Width && ph*{TW} + tx < Width)
        {{
            Ms[ty][tx] = M[Row*Width + ph*{TW} + tx];
        }}
        else
        {{
            Ms[ty][tx] = {fzero};
        }}
        if (Col < Width && ph*{TW} + ty < Width)
        {{
            Ns[ty][tx] = N[(ph*{TW} + ty)*Width + Col];
        }}
        else
        {{
            Ns[ty][tx] = {fzero};
        }}
        __syncthreads();

        {loop}
        __syncthreads();
    }}

    if (Row < Width && Col < Width)
    {{
        P[Row*Width + Col] = Pvalue;
    }}
}}

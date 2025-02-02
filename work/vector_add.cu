#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(int*A, int*B,int *C, int N) {
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

int main(){
    int N = 1 << 20;
    size_t sizeN = sizeof(int)*N;
    int* A = (int*)malloc(sizeN);
    int* B = (int*)malloc(sizeN);
    int* C = (int*)malloc(sizeN);

    for (int i=0;i<N;i++){
        A[i] = i;
        B[i] = i+2;
    }
    int* d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A,sizeN);
    cudaMalloc((void**)&d_B,sizeN);
    cudaMalloc((void**)&d_C,sizeN);

    cudaMemcpy(d_A,A,sizeN,cudaMemcpyHostToDevice);

    cudaMemcpy(d_B,B,sizeN,cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N+threadsPerBlock -1 )/threadsPerBlock;

    vectorAdd<<<threadsPerBlock,blocksPerGrid>>>(d_A,d_B,d_C,N);

    cudaMemcpy(C,d_C,sizeN,cudaMemcpyDeviceToHost);


    for (int i=0;i<10;i++){
        printf("Key: %d, Value: %d",i,C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;


}
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define GRID_WIDTH 32
#define GRID_HEIGHT 32

__global__ void conway_game_simple(char* current, char* next){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x<GRID_WIDTH && y<GRID_HEIGHT){
        int liveNeighbors = 0;
        for(int i=-1;i<=1;i+=1){
            for(int j=-1; j<=1;j+=1){
                int nx = x+i;
                int ny = y+j;
                if (nx<GRID_WIDTH && ny<GRID_HEIGHT && nx>=0 && ny>=0){
                    liveNeighbors+=current[nx*GRID_WIDTH+ny];
                }
            }
        }
        char cell = current[x*GRID_WIDTH+y];
        liveNeighbors-=cell;

        if(cell==1 && (liveNeighbors==2 || liveNeighbors==3)){
            next[x*GRID_WIDTH+y]= 1;
        }
        else{
            next[x*GRID_WIDTH+y]=0;
        }
        if(cell==0 && liveNeighbors==3){
            next[x*GRID_WIDTH+y]=1;
        }
    }
}


int main(){
    int sizeN =sizeof(char)*GRID_HEIGHT*GRID_WIDTH;
    char* grid = malloc(sizeN);
    char* current = malloc(sizeN);
    char* grid_C;
    char* current_C;

    cudaMalloc((void**)&grid_C,sizeN);
    cudaMalloc((void**)&current_C,sizeN);

    for(int i=0;i<GRID_HEIGHT;i++){
        for (int j=0; j<GRID_WIDTH;j++){
            grid[i*GRID_WIDTH+j] = 0;
            current[i*GRID_WIDTH+j] = 0;
        }
    }
    cudaMemcpy(grid_C,grid,sizeN,cudaMemcpyHostToDevice);
    cudaMemcpy(current_C,current,sizeN,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((GRID_WIDTH+threadsPerBlock.x-1)/threadsPerBlock.x,(GRID_WIDTH+threadsPerBlock.y-1)/threadsPerBlock.y);
    conway_game_simple<<<numBlocks,threadsPerBlock>>>(current_C,grid_C);

    cudaDeviceSynchronize();

    cudaMemcpy(grid,grid_C,sizeN,cudaMemcpyDeviceToHost);
    for(int i=0;i <10; i++){
        for(int j=0;j<10;j++){
            printf("%d ",grid[i*GRID_WIDTH+j]);
        }
    }
    printf("\n");
    cudaFree(current_C);
    cudaFree(grid_C);
    free(grid);
    free(current);
    return 0;
}
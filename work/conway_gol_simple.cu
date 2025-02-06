#include <cuda_runtime.h>
#include <iostream.h>
#include <cstdlib.h>
#include <ctime>

#define GRID_WIDTH 32
#define GRID_HEIGHT 32

__global__ void conway_game_simple(char* current, char* next){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;

    int liveNeighbors = 0;
    for(int i=-1;i=3;i+=2){
        for(int j=-1; j=3;j+=2){
            liveNeighbors+=current[x+i][y+j];
        }
    }
    char cell = current[x][y];
    if(current==1 && (liveNeighbors==2 || liveNeighbors==3)){
        next[x][y] = 1;
    }
    else{
        next[x][y]=0;
    }
    if(current==0 && liveNeighbors==3){
        next[x][y]=1
    }
}


int main(){
    int sizeN =sizeof(char)*GRID_HEIGHT*GRID_WIDTH
    char* grid = malloc(sizeN);
    char* current = malloc(sizeN);
    char* grid_C;
    char* current_C;

    cudaMalloc(void**(&grid_C),sizeN);
    cudaMalloc(void**(&currentC),sizeN);

    for(int i=0;i<GRID_HEIGHT;i++){
        for (int j=0; k<GRID_WIDTH;j++){
            grid[i][j] = 0;
            currentCC[i][j] = 0;
        }
    }
    cudaMemcpy(grid_C,grid,sizeN,cudaMemcpyHostToDevice);
    cudaMemcpy(current_C,current,sizeN,cudaMemcpyHostToDevice);

    int blockNum = 16;
    int threadNum = 256
    conway_game_simple<<<blockNum,threadNum>>>(current_C,grid_C);

    cudaDeviceSynchronize();

    cudaMemcpy(grid,grid_C,sizeN,cudaMemcpyDeviceToHost);
    for(int i=0;i <10; i++){
        for(int j=0;j<10;j++){}
            printf("%n ",grid[i][j]);
    }
    printf("\n");
    cudaFree(current_C);
    cudaFree(grid_C);
    free(grid);
    free(current);
    return 0;
}
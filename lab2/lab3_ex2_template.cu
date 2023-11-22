#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <string>
#include <math.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows, int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x < numARows) && (y < numBColumns)){
    DataType sum = (DataType) 0;
    for (int k = 0; k < numAColumns; k++) {
        sum += A[x * numAColumns + k] * B[k * numBColumns + y];
    }
    C[x * numAColumns + y] = sum;
  }
}

void gemmCPU(DataType *A, DataType *B, DataType *C, int numARows,int numAColumns, int numBRows, int numBColumns) {
  // naive matrix multiplication
  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numBColumns; j++) {
      DataType sum = (DataType) 0;
      for (int k = 0; k < numAColumns; k++) {
        sum += A[i * numAColumns + k] * B[k * numBColumns + j];
      }
      C[i * numAColumns + j] = sum;
    }
  }
}

int calculateDiff(DataType *host, DataType *device, int rows, int cols) {
  int flag = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int id = i * cols + j;
      if (fabs(host[id] - device[id]) > 1e-6) {
        printf("Results differ at (%d,%d): Host: %f; Device: %f", i, j, host[id], device[id]);
        flag = 1;
      }
    }
  }

  return flag;
}


struct timeval* startTime = (timeval *) malloc(sizeof(timeval));

//@@ Insert code to implement timer start

void starTimer() {
  gettimeofday(startTime, NULL);
}

//@@ Insert code to implement timer stop

void stopTimer() {
  struct timeval* currentTime = (timeval *) malloc(sizeof(timeval));
  struct timeval* diff = (timeval *) malloc(sizeof(timeval));

  gettimeofday(currentTime, NULL);
  timersub(currentTime, startTime, diff);

  printf("Time in seconds: %ld\n", diff->tv_sec);
  printf("Time in microseconds: %ld\n", diff->tv_usec);

  free(currentTime);
  free(diff);
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns,
  numARows = std::stoi(argv[1], nullptr);   
  numAColumns = std::stoi(argv[2], nullptr);
  numBRows = std::stoi(argv[3], nullptr);   
  numBColumns = std::stoi(argv[4], nullptr);
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  // check dims match
  if (numAColumns != numBRows) {
    fprintf(stderr, "Input dim mismatch.\n");
    return -1;
  }

  //@@ Insert code below to allocate Host memory for input and output
  // index by offset = i + numAColumns * j
  hostA = (DataType *) malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *) malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *) malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType *) malloc(numCRows * numCColumns * sizeof(DataType));


  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numAColumns; j++) {
      hostA[i * numAColumns + j] = ((DataType)rand() / RAND_MAX) * (DataType) 10;
    }
  }
  for (int i = 0; i < numBRows; i++) {
    for (int j = 0; j < numBColumns; j++) {
      hostB[i * numBColumns + j] = ((DataType)rand() / RAND_MAX) * (DataType) 10;
    }
  }

  printf("\nRunning on CPU...\n");
  starTimer();
  gemmCPU(hostA, hostB, resultRef, numARows, numAColumns, numBRows, numBColumns);
  stopTimer();

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  printf("Copying memory to device...\n");

  //@@ Insert code to below to Copy memory to the GPU here
  starTimer();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  stopTimer();

  //@@ Initialize the grid and block dimensions here
  // cudaDeviceProp *prop = (cudaDeviceProp *) malloc (sizeof(cudaDeviceProp));
  // cudaGetDeviceProperties_v2(prop, 0);
  // int threads_per_block = prop->maxThreadsPerBlock;
  // int no_of_blocks = (int) ceil(double(inputLength) / double(threads_per_block));

  dim3 block(32, 32);
  dim3 grid(ceil((double) numARows / (double) 32), ceil((double) numBColumns / (double) 32));

  printf("\nRunning kernel...\n");

  //@@ Launch the GPU Kernel here
  starTimer();
  gemm<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  stopTimer();

  printf("\nCopying results to host...\n");

  //@@ Copy the GPU memory back to the CPU here
  starTimer();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  stopTimer();

  //@@ Insert code below to compare the output with the reference

  printf("\nResults match: %s\n", calculateDiff(resultRef, hostC, numCRows, numCColumns) ? "false" : "true");

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);
  free(startTime);

  return 0;
}

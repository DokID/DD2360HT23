#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <random>
#include <math.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  // for (int iter = 0; iter < len; iter++) {
    
  // }
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < len) {
    out[i] = in1[i] + in2[i];
  }
}

void vecAddCPU(DataType *in1, DataType *in2, DataType *out, int len) {
 //@@ Insert code to implement vector addition here
 for (int iter = 0; iter < len; iter++) {
   out[iter] = in1[iter] + in2[iter];
 }
}

int calculateDiff(DataType *host, DataType *device, int len) {
  int flag = 0;
  for (int i = 0; i < len; i++) {
    if (fabs(host[i] - device[i]) > 1e-6) {
      printf("Results differ at %d: Host: %f; Device: %f", i, host[i], device[i]);
      flag = 1;
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
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = std::stoi(argv[1], nullptr);

  //printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*) malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType*) malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType*) malloc(inputLength * sizeof(DataType));
  resultRef = (DataType*) malloc(inputLength * sizeof(DataType));

  // printf("Initializing vectors...\n");
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(0));
  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = ((double)rand() / RAND_MAX) * (double) 10;
    hostInput2[i] = ((double)rand() / RAND_MAX) * (double) 10;
  }

  printf("\nRunning on CPU...\n");
  starTimer();
  vecAddCPU(hostInput1, hostInput2, resultRef, inputLength);
  stopTimer();

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(double));
  cudaMalloc(&deviceInput2, inputLength * sizeof(double));
  cudaMalloc(&deviceOutput, inputLength * sizeof(double));

  printf("Copying memory to device...\n");

  //@@ Insert code to below to Copy memory to the GPU here <-- TIME THIS
  starTimer();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(double), cudaMemcpyHostToDevice);
  stopTimer();

  //@@ Initialize the 1D grid and block dimensions here
  cudaDeviceProp *prop = (cudaDeviceProp *) malloc (sizeof(cudaDeviceProp));
  cudaGetDeviceProperties_v2(prop, 0);
  int threads_per_block = prop->maxThreadsPerBlock;
  int no_of_blocks = (int) ceil(double(inputLength) / double(threads_per_block));


  printf("\nRunning kernel...\n");

  //@@ Launch the GPU Kernel here <-- TIME THIS
  starTimer();
  vecAdd<<<no_of_blocks, threads_per_block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  stopTimer();

  printf("\nCopying results to host...\n");

  //@@ Copy the GPU memory back to the CPU here <-- TIME THIS
  starTimer();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(double), cudaMemcpyDeviceToHost);
  stopTimer();

  //@@ Insert code below to compare the output with the reference

  printf("\nResults match: %s\n", calculateDiff(resultRef, hostOutput, inputLength) ? "false" : "true");

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  free(startTime);

  return 0;
}

#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <random>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  for (int iter = 0; iter < len; iter++) {
    out[iter] = in1[iter] + in2[iter];
  }
}

void vecAddCPU(DataType *in1, DataType *in2, DataType *out, int len) {
 //@@ Insert code to implement vector addition here
 for (int iter = 0; iter < len; iter++) {
   out[iter] = in1[iter] + in2[iter];
 }
}



struct timeval* startTime = (timeval *) malloc(sizeof(timeval));

//@@ Insert code to implement timer start

void starTimer() {
  gettimeofday(startTime, NULL);
}

//@@ Insert code to implement timer stop

struct timeval* stopTimer() {
  struct timeval* currentTime;
  struct timeval* returnTime;

  gettimeofday(currentTime, NULL);
  timersub(startTime, currentTime, returnTime);

  return returnTime;
}

void printTime(timeval * time) {
  printf("Time in seconds: %ld", time->tv_sec);
  printf("Time in microseconds: %ld", time->tv_usec);
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

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  if((hostInput1 = (DataType*) malloc(inputLength * sizeof(DataType))) == NULL) {
    printf("Could not malloc hostInput1");
  };
  if((hostInput2 = (DataType*) malloc(inputLength * sizeof(DataType))) == NULL) {
    printf("Could not malloc hostInput2");
  };
  if((hostOutput = (DataType*) malloc(inputLength * sizeof(DataType))) == NULL) {
    printf("Could not malloc hostOutput");
  };
  if((resultRef = (DataType*) malloc(inputLength * sizeof(DataType))) == NULL) {
    printf("Could not malloc resultRef");
  };

  printf("Initializing vectors...");
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(0));
  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = ((double)rand() / RAND_MAX) * (double) 10;
    hostInput2[i] = ((double)rand() / RAND_MAX) * (double) 10;
  }

  printf("Running on CPU...");

  vecAddCPU(hostInput1, hostInput2, resultRef, inputLength);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(double));
  cudaMalloc(&deviceInput2, inputLength * sizeof(double));
  cudaMalloc(&deviceOutput, inputLength * sizeof(double));

  printf("Copying memory to device...");

  //@@ Insert code to below to Copy memory to the GPU here <-- TIME THIS
  starTimer();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(double), cudaMemcpyHostToDevice);
  printTime(stopTimer());

  //@@ Initialize the 1D grid and block dimensions here
  dim3 grid(1);
  dim3 block(16);

  printf("Running kernel...");

  //@@ Launch the GPU Kernel here <-- TIME THIS
  starTimer();
  vecAdd<<<grid, block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  printTime(stopTimer());

  printf("Copying results to host...");

  //@@ Copy the GPU memory back to the CPU here <-- TIME THIS
  starTimer();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(double), cudaMemcpyDeviceToHost);
  printTime(stopTimer());

  //@@ Insert code below to compare the output with the reference

  printf("Cleaning up...");

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

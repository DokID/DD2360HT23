#include <stdio.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop


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
  inputLength = argv[1];

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = malloc(inputLength * sizeof(double));
  hostInput2 = malloc(inputLength * sizeof(double));
  hostOutput = malloc(inputLength * sizeof(double));
  resultRef = malloc(inputLength * sizeof(double));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  // hostInput1 
  // hostInput2
  vecAdd(hostInput1, hostInput2, resultRef, inputLength);


  //@@ Insert code below to allocate GPU memory here
  double *deviceInput1 = NULL;
  double *deviceInput2 = NULL;
  double *deviceOutput = NULL;
  cudaMalloc(&deviceInput1, inputLength * sizeof(double));
  cudaMalloc(&deviceInput2, inputLength * sizeof(double));
  cudaMalloc(&deviceOutput, inputLength * sizeof(double));


  //@@ Insert code to below to Copy memory to the GPU here <-- TIME THIS
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(double), cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here
  dim3 grid(1);
  dim3 block(16);

  //@@ Launch the GPU Kernel here <-- TIME THIS
  vecAdd<<<grid, block>>>(inputs);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here <-- TIME THIS
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(double), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  

  //@@ Free the GPU memory here

  //@@ Free the CPU memory here

  return 0;
}

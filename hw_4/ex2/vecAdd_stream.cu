#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <random>
#include <math.h>

#define DataType float
#define N_STREAMS 4

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
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
      printf("Results differ at %d: Host: %f; Device: %f\n", i, host[i], device[i]);
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
  
// 1. Divide an input vector into multiple segments of a given size (S_seg)

// 2. Create 4 CUDA streams to copy asynchronously from host to GPU memory, perform vector addition on GPU, and copy back the results from GPU memory to host memory

// 3. Add timers to compare the performance using different segment size by varying the value of S_seg. 


  int inputLength;
  int segmentLength;
  int n_segments;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  cudaStream_t streams[N_STREAMS];

  //@@ Insert code below to read in inputLength from args
  inputLength = std::stoi(argv[1], nullptr);
  segmentLength = std::stoi(argv[2], nullptr);

  n_segments = (int)ceil(inputLength/segmentLength); // should probably be a whole number

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
    hostInput1[i] = ((DataType)rand() / RAND_MAX) * (DataType) 10;
    hostInput2[i] = ((DataType)rand() / RAND_MAX) * (DataType) 10;
  }

  printf("\nRunning on CPU...\n");
  starTimer();
  vecAddCPU(hostInput1, hostInput2, resultRef, inputLength);
  stopTimer();

  for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

  // printf("Copying memory to device...\n");

  //@@ Insert code to below to Copy memory to the GPU here <-- TIME THIS
  //starTimer();
  //cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  //cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  //stopTimer();

  //@@ Initialize the 1D grid and block dimensions here
  // cudaDeviceProp *prop = (cudaDeviceProp *) malloc (sizeof(cudaDeviceProp));
  // cudaGetDeviceProperties_v2(prop, 0);
  int threads_per_block = 32; // Set to warp size, just to keep it simple.
  int no_of_blocks = (int) ceil(double(segmentLength) / double(threads_per_block));


  printf("Streaming data...\n");

  //@@ Launch the GPU Kernel here <-- TIME THIS
  //starTimer();
  //vecAdd<<<no_of_blocks, threads_per_block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  //cudaDeviceSynchronize();
  //stopTimer();

  int offset, s_id, last_seg_len;
  for (int i = 0; i < n_segments; i++) {
    offset = i * segmentLength;
    s_id = i % N_STREAMS;
    // printf("Offset: %d\n", offset);
    // printf("Stream id: %d\n", s_id);
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], segmentLength * sizeof(DataType), cudaMemcpyHostToDevice, streams[s_id]);
    // printf("Copy 1 done.\n");
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], segmentLength * sizeof(DataType), cudaMemcpyHostToDevice, streams[s_id]);
    // printf("Copy 2 done.\n");
    vecAdd<<<no_of_blocks, threads_per_block, 0, streams[s_id]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], segmentLength);
    // printf("Kernel invoked.\n");
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], segmentLength * sizeof(DataType), cudaMemcpyDeviceToHost, streams[s_id]);
    // printf("Copy 3 done.\n");
  }

  // Last segment can be shorter than segmentLength, so we run a special sequence to avoid going out of bounds in GPU memory.
  
  offset = n_segments * segmentLength;
  s_id = n_segments % N_STREAMS;
  last_seg_len = inputLength - offset;

  // printf("Offset: %d\n", offset);
  // printf("Stream id: %d\n", s_id);
  // printf("Final segment length: %d\n", last_seg_len);
  
  cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], last_seg_len * sizeof(DataType), cudaMemcpyHostToDevice, streams[s_id]);
  cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], last_seg_len * sizeof(DataType), cudaMemcpyHostToDevice, streams[s_id]);
  vecAdd<<<no_of_blocks, threads_per_block, 0, streams[s_id]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], last_seg_len);
  cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], last_seg_len * sizeof(DataType), cudaMemcpyDeviceToHost, streams[s_id]);
    
  printf("Streaming finished!\n");

  // printf("\nCopying results to host...\n");

  //@@ Copy the GPU memory back to the CPU here <-- TIME THIS
  //starTimer();
  //cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  //stopTimer();

  //@@ Insert code below to compare the output with the reference
  cudaDeviceSynchronize();  //Wait for all streams to finish before verifying result.
  printf("All streams done!\n");
  printf("\nResults match: %s\n", calculateDiff(resultRef, hostOutput, inputLength) ? "false" : "true");
  
  //@@ Kill the Streams
  for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }

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
  //free(prop);

  return 0;
}

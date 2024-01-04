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

  n_segments = (int)ceil(inputLength/segmentLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*) malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType*) malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType*) malloc(inputLength * sizeof(DataType));
  resultRef = (DataType*) malloc(inputLength * sizeof(DataType));
  
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

  //@@ Initialize the 1D grid and block dimensions here
  int threads_per_block = 32; // Set to warp size, just to keep it simple.
  int no_of_blocks = (int) ceil(double(segmentLength) / double(threads_per_block));

  printf("Streaming data...\n");

  int offset, s_id, last_seg_len;
  for (int i = 0; i < n_segments; i++) {
    offset = i * segmentLength;
    s_id = i % N_STREAMS;

    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], segmentLength * sizeof(DataType), cudaMemcpyHostToDevice, streams[s_id]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], segmentLength * sizeof(DataType), cudaMemcpyHostToDevice, streams[s_id]);
    vecAdd<<<no_of_blocks, threads_per_block, 0, streams[s_id]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], segmentLength);
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], segmentLength * sizeof(DataType), cudaMemcpyDeviceToHost, streams[s_id]);
  }

  // Last segment can be shorter than segmentLength, so we run a special sequence to avoid going out of bounds in GPU memory.
  
  offset = n_segments * segmentLength;
  s_id = n_segments % N_STREAMS;
  last_seg_len = inputLength - offset;
  
  cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], last_seg_len * sizeof(DataType), cudaMemcpyHostToDevice, streams[s_id]);
  cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], last_seg_len * sizeof(DataType), cudaMemcpyHostToDevice, streams[s_id]);
  vecAdd<<<no_of_blocks, threads_per_block, 0, streams[s_id]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], last_seg_len);
  cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], last_seg_len * sizeof(DataType), cudaMemcpyDeviceToHost, streams[s_id]);
    
  printf("Streaming finished!\n");

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

  return 0;
}

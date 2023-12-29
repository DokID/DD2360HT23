
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <random>
#include <math.h>

#define NUM_BINS 4096
#define MAX_INPUT_LENGTH 1 << 32;
#define MAX_SATURATION 127
#define DataType unsigned int

__global__ void histogram_kernel(DataType *input, DataType *bins, DataType num_elements, DataType num_bins) {

  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

//@@ Insert code below to compute histogram of input using shared memory and atomics
  __shared__ DataType s_bins[NUM_BINS];
  if (idx < num_elements) {
    atomicAdd(&s_bins[input[idx]], 1);
  }
  __syncthreads();

  int share = ((int)num_bins/blockDim.x); // if blockDim.x == 1024
  for (int i = share*threadIdx.x ; i < share*threadIdx.x + share ; i++) {
    atomicAdd(&bins[i], atomicExch(&s_bins[i], 0));
  }
}


__global__ void convert_kernel(DataType *bins, DataType num_bins) {

  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_bins) return;

//@@ Insert code below to clean up bins that saturate at 127
  if(bins[idx] > MAX_SATURATION) bins[idx] = MAX_SATURATION; // set value of saturated bin to MAX_SATURATION
}


void histogram_CPU(DataType *input, DataType *bins, DataType num_elements, DataType num_bins) {
  for (int i = 0; i < num_elements; i++) {
    bins[input[i]] += 1;
  }

  for (int i = 0; i < num_bins; i++) {
    if (bins[i] > MAX_SATURATION) bins[i] = MAX_SATURATION;
  }
}


int calculateDiff(DataType *host_bins, DataType *device_bins, DataType num_bins) {
  int flag = 0;
  int misses = 0;
  for (int i = 0; i < num_bins; i++) {
    if (host_bins[i] != device_bins[i]) {
      misses += 1;
      printf("Results differ at %d: Host: %u; Device: %u\n", i, host_bins[i], device_bins[i]);
      flag = 1;
    }
  }
  printf("\nMiss rate: %d / %d\n", misses, num_bins);
  return flag;
}


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput;
  DataType *hostBins;
  DataType *resultRef;
  DataType *deviceInput;
  DataType *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = std::stoi(argv[1], nullptr);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (DataType*) malloc(inputLength * sizeof(DataType));
  hostBins = (DataType*) malloc(NUM_BINS * sizeof(DataType));
  resultRef = (DataType*) malloc(NUM_BINS * sizeof(DataType));
  memset(resultRef, 0, NUM_BINS);
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = (DataType)floor(((double)rand() / (double)RAND_MAX) * (double)(NUM_BINS));
  }

  //@@ Insert code below to create reference result in CPU
  histogram_CPU(hostInput, resultRef, inputLength, NUM_BINS);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(DataType));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(DataType));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS);

  //@@ Initialize the grid and block dimensions here
  cudaDeviceProp *prop = (cudaDeviceProp *) malloc (sizeof(cudaDeviceProp));
  cudaGetDeviceProperties_v2(prop, 0);
  int threads_per_block = prop->maxThreadsPerBlock;
  
  dim3 block1(threads_per_block);
  dim3 grid1((int)ceil((double)inputLength/(double)block1.x));
  printf("max threads / block: %d\nblocks / grid: %d\n", block1.x, grid1.x);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<grid1, block1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Initialize the second grid and block dimensions here
  dim3 block2(threads_per_block);
  dim3 grid2((int)ceil((double)NUM_BINS/(double)block2.x));

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<grid2, block2>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(DataType), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  printf("Results match: %s\n", calculateDiff(resultRef, hostBins, NUM_BINS) ? "false" : "true");

  FILE *f;
  f = fopen("vecadd_result.txt", "w");
  for (int i = 0; i < NUM_BINS; i++) {
    fprintf(f, "%d, %u\n", i, hostBins[i]);
  }
  fclose(f);

  printf("Wrote bins to file \'vecadd_result.txt\'\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);
  
  return 0;
}


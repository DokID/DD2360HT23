//
// Created by gaborn on 2024-01-07.
//
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kmeans_cuda_kernel.cuh"
#include "alloc.h"

extern "C"
void allocateFeatures(float **features, int npoints, int nfeatures) {
    gpuCheck(cudaMallocManaged(features, npoints * nfeatures * sizeof(float)));
    gpuCheck(cudaMemPrefetchAsync(features[0], npoints * nfeatures * sizeof(float), cudaCpuDeviceId));
    gpuCheck(cudaDeviceSynchronize());
}

extern "C"
void prefetchFeaturesToHost(float *features, int npoints, int nfeatures) {
    gpuCheck(cudaMemPrefetchAsync(features, npoints * nfeatures * sizeof(float), cudaCpuDeviceId));
    gpuCheck(cudaDeviceSynchronize());
}

extern "C"
void prefetchFeaturesToDevice(float *features, int npoints, int nfeatures) {
    gpuCheck(cudaMemPrefetchAsync(features, npoints * nfeatures * sizeof(float), 0));
}

extern "C"
void deallocateFeatures(float *features) {
    gpuCheck(cudaFree(features));
}

extern "C"
void allocateClusters(float **clusters, int nclusters, int nfeatures) {
    gpuCheck(cudaMallocManaged(clusters, nclusters * nfeatures * sizeof(float)));
    gpuCheck(cudaMemPrefetchAsync(clusters[0], nclusters * nfeatures * sizeof(float), cudaCpuDeviceId));
    gpuCheck(cudaDeviceSynchronize());
}

extern "C"
void prefetchClustersToHost(float *clusters, int nclusters, int nfeatures) {
    gpuCheck(cudaMemPrefetchAsync(clusters, nclusters * nfeatures * sizeof(float), cudaCpuDeviceId));
    gpuCheck(cudaDeviceSynchronize());
}

extern "C"
void prefetchClustersToDevice(float *clusters, int nclusters, int nfeatures) {
    gpuCheck(cudaMemPrefetchAsync(clusters, nclusters * nfeatures * sizeof(float), 0));
}

extern "C"
void deallocateClusters(float *clusters) {
    gpuCheck(cudaFree(clusters));
}

extern "C"
void allocateMembership(int **membership, int npoints) {
    gpuCheck(cudaMallocManaged(membership, npoints * sizeof(int)));
    gpuCheck(cudaMemPrefetchAsync(membership[0], npoints * sizeof(int), cudaCpuDeviceId));
    gpuCheck(cudaDeviceSynchronize());
}

extern "C"
void prefetchMembershipToHost(int *memberhsip, int npoints) {
    gpuCheck(cudaMemPrefetchAsync(memberhsip, npoints * sizeof(int), cudaCpuDeviceId));
    gpuCheck(cudaDeviceSynchronize());
}

extern "C"
void prefetchMembershipToDevice(int *memberhsip, int npoints) {
    gpuCheck(cudaMemPrefetchAsync(memberhsip, npoints * sizeof(int), 0));
}

extern "C"
void deallocateMembership(int *membership) {
    gpuCheck(cudaFree(membership));
}

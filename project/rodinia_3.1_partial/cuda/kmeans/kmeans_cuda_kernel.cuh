//
// Created by gaborn on 2023-12-27.
//

#ifndef GPUPROG_HT23_KMEANS_CUDA_KERNEL_CUH
#define GPUPROG_HT23_KMEANS_CUDA_KERNEL_CUH

__global__ void invert_mapping(float*, float*, int, int);
__global__ void kmeansPoint(float*, int, int, int, int*, float*, float*, int*, cudaTextureObject_t, cudaTextureObject_t);

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

__constant__ float c_clusters[ASSUMED_NR_CLUSTERS*34];		/* constant memory for cluster centers */

#endif //GPUPROG_HT23_KMEANS_CUDA_KERNEL_CUH

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <omp.h>

#include <cuda.h>

#include "kmeans.h"
#include "kmeans_cuda_kernel.cuh"
#include "alloc.h"

//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */
         int concurrentAccessQ;                                     // Check if concurrent access flag is set
         int device = 0;                                            // Device to be used

/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_inverted;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */

extern "C"
void initializeKernelLayout(int npoints) {
    num_blocks = npoints / num_threads;
    if (npoints % num_threads > 0)		/* defeat truncation */
        num_blocks++;

    num_blocks_perdim = sqrt((double) num_blocks);
    while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
        num_blocks_perdim++;

    num_blocks = num_blocks_perdim*num_blocks_perdim;
}

/* -------------- deallocateMemory() end ------------------- */



////////////////////////////////////////////////////////////////////////////////
// Program main																  //

int
main( int argc, char** argv) 
{
	// make sure we're running on the big card
    cudaSetDevice(0);
	// as done in the CUDA start/help document provided
	setup(argc, argv);    
}

//																			  //
////////////////////////////////////////////////////////////////////////////////


/* ------------------- kmeansCuda() ------------------------ */    
extern "C"
int	// delta -- had problems when return value was of float type
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
           int      nfeatures,				/* number of attributes for each point */
           int      npoints,				/* number of data points */
           int      nclusters,				/* number of clusters */
           int     *membership,				/* which cluster the point belongs to */
		   float  **clusters,				/* coordinates of cluster centers */
		   int     *new_centers_len,		/* number of elements in each cluster */
           float  **new_centers				/* sum of elements in each cluster */
		   )
{
	int delta = 0;			/* if point has moved */
	int i,j;				/* counters */

    cudaResourceDesc resDescFeatures{};
    cudaResourceDesc resDescFeaturesFlipped{};
    cudaResourceDesc resDescClusters{};

    allocateFeatures(&feature_inverted, npoints, nfeatures);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat );

    memset(&resDescFeatures, 0, sizeof(resDescFeatures));
    resDescFeatures.resType = cudaResourceTypeLinear;
    resDescFeatures.res.linear.devPtr = feature_inverted;
    resDescFeatures.res.linear.desc = channelDesc;
    //resDescFeatures.res.linear.desc.f = cudaChannelFormatKindFloat;
    //resDescFeatures.res.linear.desc.x = 8;  // Default value for cudaChannelFormatDescription
    resDescFeatures.res.linear.sizeInBytes = npoints * nfeatures * sizeof(float);

    memset(&resDescFeaturesFlipped, 0, sizeof(resDescFeaturesFlipped));
    resDescFeaturesFlipped.resType = cudaResourceTypeLinear;
    resDescFeaturesFlipped.res.linear.devPtr = feature[0];
    resDescFeaturesFlipped.res.linear.desc = channelDesc;
    //resDescFeaturesFlipped.res.linear.desc.f = cudaChannelFormatKindFloat;
    //resDescFeaturesFlipped.res.linear.desc.x = 8;  // Default value for cudaChannelFormatDescription
    resDescFeaturesFlipped.res.linear.sizeInBytes = npoints * nfeatures * sizeof(float);

    memset(&resDescClusters, 0, sizeof(resDescClusters));
    resDescClusters.resType = cudaResourceTypeLinear;
    resDescClusters.res.linear.devPtr = clusters[0];
    resDescClusters.res.linear.desc = channelDesc;
    //resDescClusters.res.linear.desc.f = cudaChannelFormatKindFloat;
    //resDescClusters.res.linear.desc.x = 8;  // Default value for cudaChannelFormatDescription
    resDescClusters.res.linear.sizeInBytes = nclusters * nfeatures * sizeof(float);

    cudaTextureDesc texDesc{};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t t_features;
    cudaTextureObject_t t_features_flipped;
    cudaTextureObject_t t_clusters;
    gpuCheck(cudaCreateTextureObject(&t_features, &resDescFeatures, &texDesc, nullptr));
    gpuCheck(cudaCreateTextureObject(&t_features_flipped, &resDescFeatures, &texDesc, nullptr));
    gpuCheck(cudaCreateTextureObject(&t_clusters, &resDescFeatures, &texDesc, nullptr));


	/* copy membership (host to device) */
	//gpuCheck(cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice));

	/* copy clusters (host to device) */
	//gpuCheck(cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice));

	/* set up texture */
    // FIXME: Convert to new Texture Object API
//    cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
//    t_features.filterMode = cudaFilterModePoint;
//    t_features.normalized = false;
//    t_features.channelDesc = chDesc0;
//
//	if(cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
//        printf("Couldn't bind features array to texture!\n");

    //// FIXME: Convert to new Texture Object API
	//cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
    //t_features_flipped.filterMode = cudaFilterModePoint;
    //t_features_flipped.normalized = false;
    //t_features_flipped.channelDesc = chDesc1;

	//if(cudaBindTexture(NULL, &t_features_flipped, feature_inverted, &chDesc1, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
    //    printf("Couldn't bind features_flipped array to texture!\n");

    //// FIXME: Convert to new Texture Object API
    //cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
    //t_clusters.filterMode = cudaFilterModePoint;
    //t_clusters.normalized = false;
    //t_clusters.channelDesc = chDesc2;

	//if(cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2, nclusters*nfeatures*sizeof(float)) != CUDA_SUCCESS)
    //    printf("Couldn't bind clusters array to texture!\n");

	/* copy clusters to constant memory */
	//gpuCheck(cudaMemcpyToSymbol(c_clusters,clusters[0],nclusters*nfeatures*sizeof(float),0,cudaMemcpyHostToDevice));

    /* setup execution parameters.
	   changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    dim3  grid( num_blocks_perdim, num_blocks_perdim );
    dim3  threads( num_threads_perdim*num_threads_perdim );

    prefetchFeaturesToDevice(feature[0], npoints, npoints);
    prefetchFeaturesToDevice(feature_inverted, npoints, nfeatures);

    invert_mapping<<<num_blocks, num_threads>>>(feature[0],
                                                feature_inverted,
                                                npoints,
                                                nfeatures);

    prefetchClustersToDevice(clusters[0], nclusters, nfeatures);
    prefetchMembershipToDevice(membership, npoints);

	/* execute the kernel */
    kmeansPoint<<< grid, threads >>>( feature_inverted,
                                      nfeatures,
                                      npoints,
                                      nclusters,
                                      membership,
                                      clusters[0],
									  block_clusters_d,
									  block_deltas_d,
                                      t_features,
                                      t_features_flipped);

	gpuCheck(cudaDeviceSynchronize());


    prefetchClustersToHost(clusters[0], nclusters, nfeatures);
    prefetchFeaturesToHost(feature[0], npoints, npoints);
    prefetchMembershipToHost(membership, npoints);

	/* copy back membership (device to host) */
	//gpuCheck(cudaMemcpy(membership_new, membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost));

#ifdef BLOCK_CENTER_REDUCE
    /*** Copy back arrays of per block sums ***/
    float * block_clusters_h = (float *) malloc(
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
        
	cudaMemcpy(block_clusters_h, block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float), 
        cudaMemcpyDeviceToHost);
#endif
#ifdef BLOCK_DELTA_REDUCE
    int * block_deltas_h = (int *) malloc(
        num_blocks_perdim * num_blocks_perdim * sizeof(int));
        
	cudaMemcpy(block_deltas_h, block_deltas_d, 
        num_blocks_perdim * num_blocks_perdim * sizeof(int), 
        cudaMemcpyDeviceToHost);
#endif
    
	/* for each point, sum data points in each cluster
	   and see if membership has changed:
	     if so, increase delta and change old membership, and update new_centers;
	     otherwise, update new_centers */
	delta = 0;
	for (i = 0; i < npoints; i++)
	{		
		int cluster_id = membership[i];
		new_centers_len[cluster_id]++;
//		if (membership_new[i] != membership[i])
//		{
//#ifdef CPU_DELTA_REDUCE
//			delta++;
//#endif
//			membership[i] = membership_new[i];
//		}
#ifdef CPU_CENTER_REDUCE
		for (j = 0; j < nfeatures; j++)
		{			
			new_centers[cluster_id][j] += feature[i][j];
		}
#endif
	}
	

#ifdef BLOCK_DELTA_REDUCE	
    /*** calculate global sums from per block sums for delta and the new centers ***/    
	
	//debug
	//printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		//printf("block %d delta is %d \n",i,block_deltas_h[i]);
        delta += block_deltas_h[i];
    }
        
#endif
#ifdef BLOCK_CENTER_REDUCE	
	
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			block_new_centers[j*nfeatures + k] = 0.f;
		}
	}

    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		for(int j = 0; j < nclusters;j++) {
			for(int k = 0; k < nfeatures;k++) {
				block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
			}
		}
    }
	

#ifdef CPU_CENTER_REDUCE
	//debug
	/*for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			if(new_centers[j][k] >	1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] <	0.999 * block_new_centers[j*nfeatures + k]) {
				printf("\t \t for %d:%d, normal value is %e and gpu reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures + k]);
			}
		}
	}*/
#endif

#ifdef BLOCK_CENTER_REDUCE
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++)
			new_centers[j][k]= block_new_centers[j*nfeatures + k];		
	}
#endif

#endif

	return delta;
	
}
/* ------------------- kmeansCuda() end ------------------------ */    


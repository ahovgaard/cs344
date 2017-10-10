/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>
#include <float.h>
#include "utils.h"

// Min or max reduction. Produces a reduced value per block.
__global__
void reduce_minmax(float *d_out, const float *d_in, size_t n, bool maxMode)
{
  // allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float cache[];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  float tmp = FLT_MAX;
  if (maxMode)
    tmp = FLT_MIN;

  // reduction of values outside the span of the grid
  while (tid < n) {
    if (maxMode)
      tmp = max(tmp, d_in[tid]);
    else
      tmp = min(tmp, d_in[tid]);

    tid += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[cacheIndex] = tmp;

  // synchronize threads in this block
  __syncthreads();

  // tree reduction of values in cache
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (cacheIndex < i) {
      if (maxMode)
        cache[cacheIndex] = max(cache[cacheIndex], cache[cacheIndex + i]);
      else
        cache[cacheIndex] = min(cache[cacheIndex], cache[cacheIndex + i]);
    }
    __syncthreads();
  }

  if (cacheIndex == 0)
    d_out[blockIdx.x] = cache[0];
}

// Computes a histogram of the logLuminance channel.
__global__
void histogram(unsigned int *d_bins, size_t numBins,
               const float *d_in, size_t n, float lumMin, float lumRange)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // increment global thread index by the total number of threads for
  // each iteration, to handle the case where there are more input
  // values than threads
  while (tid < n) {
    int bin = ((d_in[tid] - lumMin) / lumRange) * numBins;
    atomicAdd(&d_bins[bin], 1);
    tid += blockDim.x * gridDim.x;
  }
}

// Hillis & Steele exclusive sum scan.
__global__
void hillis_steele_excl_scan(unsigned int *d_out, const unsigned int *d_in,
                             size_t n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
    d_out[tid] = d_in[tid];

  for (int i = 1; i < n; i *= 2) {
    if (tid < n && tid - i >= 0)
      d_out[tid] += d_out[tid - i];
    __syncthreads();
  }

  // convert to exclusive scan
  if (tid < n)
    d_out[tid] -= d_in[tid];
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /* Here are the steps you need to implement
     1) find the minimum and maximum value in the input logLuminance channel
        store in min_logLum and max_logLum
     2) subtract them to find the range
     3) generate a histogram of all the values in the logLuminance channel using
        the formula: bin = (lum[i] - lumMin) / lumRange * numBins
     4) Perform an exclusive scan (prefix sum) on the histogram to get
        the cumulative distribution of luminance values (this should go in the
        incoming d_cdf pointer which already has been allocated for you)
  */

  // Expected:
  //   min_logLum = -3.109206
  //   max_logLum =  2.265088
  //   num_bins   =  1024
  //   bins       =  [1, 0, 0, ..., 9, 10, 9, 7, 11, 9, 15, 13, 12, 18, 16, 32, 29, ..]
  //   cdf        =  [1, 1, 1, 1, ..., 6, 8, 9, 10, 11, 12, 13, 16, 20 ,21 ,22, 31, ..]

  const int threads = 128;
  const int blocks = min(64, (int) ((numRows * numCols) + threads - 1) / threads);

  float res, lumRange;
  float *d_intermediate;
  float *intermediate;


  // 1) find the minimum and maximum value in the input logLuminance
  //    channel store in min_logLum and max_logLum

  // allocate memory for intermediate values on the CPU and GPU
  intermediate = (float *) malloc(sizeof(float) * blocks);
  checkCudaErrors(cudaMalloc((void **) &d_intermediate, sizeof(float) * blocks));

  // launch min reduction kernel
  reduce_minmax<<<blocks, threads, sizeof(float) * threads>>>
    (d_intermediate, d_logLuminance, numRows * numCols, false);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // copy the intermediate values back from the GPU to the CPU
  checkCudaErrors(cudaMemcpy(intermediate, d_intermediate,
                             sizeof(float) * blocks, cudaMemcpyDeviceToHost));

  // finish up on the CPU side
  res = FLT_MAX;
  for (int i = 0; i < blocks; i++)
    res = min(res, intermediate[i]);
  min_logLum = res;

  // launch max kernel
  reduce_minmax<<<blocks, threads, sizeof(float) * threads>>>
    (d_intermediate, d_logLuminance, numRows * numCols, true);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // copy the intermediate values back from the GPU to the CPU
  checkCudaErrors(cudaMemcpy(intermediate, d_intermediate,
                             sizeof(float) * blocks, cudaMemcpyDeviceToHost));

  // finish up on the CPU side
  res = FLT_MIN;
  for (int i = 0; i < blocks; i++)
    res = max(res, intermediate[i]);
  max_logLum = res;

  printf("min_logLum = %f\n", min_logLum);
  printf("max_logLum = %f\n", max_logLum);


  // 2) subtract them to find the range
  lumRange = max_logLum - min_logLum;
  printf("lumRange = %f\n", lumRange);


  // 3) generate a histogram of all the values in the logLuminance channel
  //    using the formula: bin = (lum[i] - lumMin) / lumRange * numBins

  unsigned int *d_bins;
  size_t histoSize = sizeof(unsigned int) * numBins;

  // allocate memory for the bins on the device and initialize to zero
  checkCudaErrors(cudaMalloc((void **) &d_bins, histoSize));
  checkCudaErrors(cudaMemset(d_bins, 0, histoSize));

  // launch histogram kernel
  histogram<<<blocks, threads>>>(d_bins, numBins, d_logLuminance,
                                 numRows * numCols, min_logLum, lumRange);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  // 4) Perform an exclusive scan (prefix sum) on the histogram to get
  //    the cumulative distribution of luminance values (this should go in the
  //    incoming d_cdf pointer which already has been allocated for you)

  hillis_steele_excl_scan<<<1, numBins>>>(d_cdf, d_bins, numBins);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // free memeory
  free(intermediate);
  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_bins));
}

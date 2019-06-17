#include <stdio.h>
#include <stdlib.h>

/**
* Computes the log of reaction rate.
* @param a: Pointer to coefficient matrix.
* @param temp: Pointer to temperature array.
* @param lam: Matrix to write the results to.
* @param nsets: Number of sets / number of rows in coefficient matrix.
* @param ncells: Number of cells / length of temperature array.
* @param ncoeff: Number of coefficients / number of columns in coefficient matrix.
*/
template <class dtype>
__device__ void rates(dtype *a, dtype *temp, dtype *lam, int nsets, int ncells, int ncoeff)
{
    int istart = blockIdx.x * blockDim.x + threadIdx.x;
    int istep = blockDim.x * gridDim.x;
    
    int jstart = blockIdx.y * blockDim.y + threadIdx.y;
    int jstep = blockDim.y * gridDim.y;
    
    int kstart = blockIdx.z * blockDim.z + threadIdx.z;
    int kstep = blockDim.z * gridDim.z;
    
    for(int i = istart; i < nsets; i += istep)
    {
        for(int j = jstart; j < ncells; j += jstep)
        {
            dtype temp9 = temp[j] * 1.0e-9;
            
            for(int k = kstart; k < ncoeff; k += kstep)
            {
                switch(k)
                {
                    case 0:
                        atomicAdd(&lam[i * ncells + j], a[i * ncoeff + k]);
                        break;
                    case 6: 
                        atomicAdd(&lam[i * ncells + j], a[i * ncoeff + k] * log(temp9)); 
                        break;
                    default: 
                        atomicAdd(&lam[i * ncells + j], a[i * ncoeff + k] * pow(temp9, (2 * k - 5) / 3.0)); 
                        break;
                }
            }
        }
    }
}

template <>
__device__ void rates<float>(float *a, float *temp, float *lam, int nsets, int ncells, int ncoeff)
{
    int istart = blockIdx.x * blockDim.x + threadIdx.x;
    int istep = blockDim.x * gridDim.x;
    
    int jstart = blockIdx.y * blockDim.y + threadIdx.y;
    int jstep = blockDim.y * gridDim.y;
    
    int kstart = blockIdx.z * blockDim.z + threadIdx.z;
    int kstep = blockDim.z * gridDim.z;
    
    for(int i = istart; i < nsets; i += istep)
    {
        for(int j = jstart; j < ncells; j += jstep)
        {
            float temp9 = temp[j] * 1.0e-9;
            
            for(int k = kstart; k < ncoeff; k += kstep)
            {
                switch(k)
                {
                    case 0:
                        atomicAdd(&lam[i * ncells + j], a[i * ncoeff + k]);
                        break;
                    case 6: 
                        atomicAdd(&lam[i * ncells + j], a[i * ncoeff + k] * logf(temp9)); 
                        break;
                    default: 
                        atomicAdd(&lam[i * ncells + j], a[i * ncoeff + k] * powf(temp9, (2 * k - 5) / 3.0f)); 
                        break;
                }
            }
        }
    }
}

template <class dtype, int nsets, int ncells, int ncoeff>
__global__ void exec(dtype* lam)
{
    // Tensors
    __shared__ dtype a[nsets * ncoeff];
    __shared__ dtype temp[ncells];
    
    int xInd = blockIdx.x * blockDim.x + threadIdx.x;
    int yInd = blockIdx.y * blockDim.y + threadIdx.y;
    int ySize = blockDim.y * gridDim.y;
    int zInd = blockIdx.z * blockDim.z + threadIdx.z;
    int zSize = blockDim.z * gridDim.z;
    
    int ind = xInd * ySize * zSize + yInd * zSize + zInd;
    
    /********************************
    * Initialize coefficient matrix *
    ********************************/
    if(ind < nsets * ncoeff)
    {
        if(ind % ncoeff != 7)
        {
            a[ind] = ind - (ind / ncoeff - 1);
        }
        else
        {
            a[ind] = 0.0;
        }
    }
    
    /******************************************
    * Initialize the temperature in each cell *
    ******************************************/
    if(ind < ncells)
    {
        temp[ind] = (ind + 1) * 1e9;
    }
    
    /****************************
    * Zero the array of results *
    ****************************/
    if(ind < nsets * ncells)
    {
        lam[ind] = 0.0;
    }
    
    /*******************************************
    * Compute ln(lambda) for each set and cell *
    *******************************************/
    rates<dtype>(a, temp, lam, nsets, ncells, ncoeff);
}

int main()
{
    // Tensor dimensions
    const int nsets = 4, ncells = 4, ncoeff = 8;
    
    // Results and elapsed time
    float *lam;
    cudaMallocManaged(&lam, nsets * ncells * sizeof(float));
    
    // Compute the rates
    dim3 threadsPerBlock(nsets, ncells, ncoeff);
    dim3 numBlocks(1, 1, 1);
    exec<float, nsets, ncells, ncoeff><<<numBlocks, threadsPerBlock>>>(lam);
    
    // Print ln(lambda)
    cudaDeviceSynchronize();
    printf("lambda:\n");
    
    for(int i = 0; i < nsets; i++)
    {
        for(int j = 0; j < ncells; j++)
        {
            printf("%.3f\t", lam[i * ncells + j]);
        }
        
        printf("\n");
    }
    
    return 0;
}

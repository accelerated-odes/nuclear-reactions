#include <stdio.h>
#include <stdlib.h>
#include <fstream>

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
            dtype temp9 = temp[j] * 1e-9;
            
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
            float temp9 = temp[j] * 1e-9;
            
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
__global__ void exec(dtype *lam)
{
    int xInd = blockIdx.x * blockDim.x + threadIdx.x;
    int yInd = blockIdx.y * blockDim.y + threadIdx.y;
    int ySize = blockDim.y * gridDim.y;
    int zInd = blockIdx.z * blockDim.z + threadIdx.z;
    int zSize = blockDim.z * gridDim.z;
    
    int ind = xInd * ySize * zSize + yInd * zSize + zInd;
    
    // Tensors
    __shared__ dtype a[nsets * ncoeff];
    
    // These are all of the sets in reaclib with two nuclei as reactants
    // where one of them is carbon-12.
    if(ind == 0)
    {
        // c12 + c12 -> n + mg23
        a[0] = -12.8056;
        a[1] = -30.1498;
        a[2] = 0.0;
        a[3] = 11.4826;
        a[4] = 1.82849;
        a[5] = -0.34844;
        a[6] = 0.0;
        
        // c12 + c12 -> p + na23
        a[7] = 60.9649;
        a[8] = 0.0;
        a[9] = -84.165;
        a[10] = -1.4191;
        a[11] = -0.114619;
        a[12] = -0.070307;
        a[13] = -0.666667;
        
        // c12 + c12 -> he4 + ne20
        a[14] = 61.2863;
        a[15] = 0.0;
        a[16] = -84.165;
        a[17] = -1.56627;
        a[18] = -0.0736084;
        a[19] = -0.072797;
        a[20] = -0.666667;
        
        // he4 + he4 + he4 -> c12 (1)
        a[21] = -0.971052;
        a[22] = 0.0;
        a[23] = -37.06;
        a[24] = 29.3493;
        a[25] = -115.507;
        a[26] = -10.0;
        a[27] = -1.33333;
        
        // he4 + he4 + he4 -> c12 (2)
        a[28] = -11.7884;
        a[29] = -1.02446;
        a[30] = -23.57;
        a[31] = 20.4886;
        a[32] = -12.9882;
        a[33] = -20.0;
        a[34] = -2.16667;
        
        // he4 + he4 + he4 -> c12 (3)
        a[35] = -24.3505;
        a[36] = -4.12656;
        a[37] = -13.49;
        a[38] = 21.4259;
        a[39] = -1.34769;
        a[40] = 0.0879816;
        a[41] = -13.1653;
    }
    
    __shared__ dtype temp[ncells];
    
    if(ind == 0)
    {
        #pragma unroll
        for(int i = 0; i < ncells; i++)
        {
            temp[i] = pow(10.0, 7 + i * 3.0 / ncells);
        }
    }
    
    __syncthreads();
    
    /*******************************************
    * Compute ln(lambda) for each set and cell *
    *******************************************/
    rates<dtype>(a, temp, lam, nsets, ncells, ncoeff);
}

int main()
{
    // Tensor dimensions
    const int nsets = 6, ncells = 10, ncoeff = 7;
    
    // Results matrix
    double *lam;
    cudaError_t code = cudaMallocManaged(&lam, nsets * ncells * sizeof(double));
    if(code != cudaSuccess) return -1;
    
    for(int i = 0; i < nsets; i++)
    {
        for(int j = 0; j < ncells; j++)
        {
            lam[i * ncells + j] = 0.0f;
        }
        
        printf("\n");
    }
    
    // Compute the rates
    dim3 threadsPerBlock(nsets, ncells, ncoeff);
    dim3 numBlocks(1, 1, 1);
    exec<double, nsets, ncells, ncoeff><<<numBlocks, threadsPerBlock>>>(lam);
    
    // Write lambda to file
    cudaDeviceSynchronize();
    
    std::ofstream file;
    file.open("double.dat");
    
    for(int i = 0; i < nsets; i++)
    {
        for(int j = 0; j < ncells; j++)
        {
            printf("%8.3f ", lam[i * ncells + j]);
            file << exp(lam[i * ncells + j]);
            if(j != ncells - 1) file << " ";
        }
        
        file << "\n";
        printf("\n");
    }
    
    file.close();
    
    return 0;
}

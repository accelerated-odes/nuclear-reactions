#include <stdio.h>
#include <iostream>
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
            for(int k = kstart; k < ncoeff; k += kstep)
            {
                atomicAdd(&lam[i * ncells + j], a[i * ncoeff + k] * temp[k * ncells + j]);
            }
        }
    }
}

template <class dtype, int nsets, int ncells, int ncoeff>
__global__ void exec(dtype *lam)
{
    int xInd = blockIdx.x * blockDim.x + threadIdx.x;
    int xSize = blockDim.x * gridDim.x;
    int yInd = blockIdx.y * blockDim.y + threadIdx.y;
    int ySize = blockDim.y * gridDim.y;
    int zInd = blockIdx.z * blockDim.z + threadIdx.z;
    int zSize = blockDim.z * gridDim.z;
    
    int ind = xInd * ySize * zSize + yInd * zSize + zInd;
    int numThreads = xSize * ySize * zSize;
    
    // Tensors
    __shared__ dtype a[nsets * ncoeff];
    
    // These are all of the sets in reaclib with two nuclei as reactants
    // where one of them is carbon-12.
    if(ind == 0)
    {
        // he4 + c12 -> o16 (1)
        a[0] = 69.6526;
        a[1] = -1.39254;
        a[2] = 58.9128;
        a[3] = -148.273;
        a[4] = 9.08324;
        a[5] = -0.541041;
        a[6] = 70.3554;
        
        // he4 + c12 -> o16 (2)
        a[7] = 254.634;
        a[8] = -1.84097;
        a[9] = 103.411;
        a[10] = -420.567;
        a[11] = 64.0874;
        a[12] = -12.4624;
        a[13] = 137.303;
        
        // he4 + c12 -> n + o15
        a[14] = 17.0115;
        a[15] = -98.6615;
        a[16] = 0.0;
        a[17] = 0.124787;
        a[18] = 0.0588937;
        a[19] = -0.00679206;
        a[20] = 0.0;
        
        // he4 + c12 -> p + n15 (1)
        a[21] = 27.118;
        a[22] = -57.6279;
        a[23] = -15.253;
        a[24] = 1.59318;
        a[25] = 2.4479;
        a[26] = -2.19708;
        a[27] = -0.666667;
        
        // he4 + c12 -> p + n15 (2)
        a[28] = -5.2319;
        a[29] = -59.6491;
        a[30] = 0.0;
        a[31] = 30.8497;
        a[32] = -8.50433;
        a[33] = -1.54426;
        a[34] = -1.5;
        
        // he4 + c12 -> p + n15 (3)
        a[35] = 20.5388;
        a[36] = -65.034;
        a[37] = 0.0;
        a[38] = 0.0;
        a[39] = 0.0;
        a[40] = 0.0;
        a[41] = -1.5;
        
        // he4 + c12 -> p + n15 (4)
        a[42] = -6.93365;
        a[43] = -58.7917;
        a[44] = 0.0;
        a[45] = 22.7105;
        a[46] = -2.90707;
        a[47] = 0.205754;
        a[48] = -1.5;
        
        // c12 + c12 -> n + mg23
        a[49] = -12.8056;
        a[50] = -30.1498;
        a[51] = 0.0;
        a[52] = 11.4826;
        a[53] = 1.82849;
        a[54] = -0.34844;
        a[55] = 0.0;
        
        // c12 + c12 -> p + na23
        a[56] = 60.9649;
        a[57] = 0.0;
        a[58] = -84.165;
        a[59] = -1.4191;
        a[60] = -0.114619;
        a[61] = -0.070307;
        a[62] = -0.666667;
        
        // c12 + c12 -> he4 + ne20
        a[63] = 61.2863;
        a[64] = 0.0;
        a[65] = -84.165;
        a[66] = -1.56627;
        a[67] = -0.0736084;
        a[68] = -0.072797;
        a[69] = -0.666667;
        
        // he4 + he4 + he4 -> c12 (1)
        a[70] = -0.971052;
        a[71] = 0.0;
        a[72] = -37.06;
        a[73] = 29.3493;
        a[74] = -115.507;
        a[75] = -10.0;
        a[76] = -1.33333;
        
        // he4 + he4 + he4 -> c12 (2)
        a[77] = -11.7884;
        a[78] = -1.02446;
        a[79] = -23.57;
        a[80] = 20.4886;
        a[81] = -12.9882;
        a[82] = -20.0;
        a[83] = -2.16667;
        
        // he4 + he4 + he4 -> c12 (3)
        a[84] = -24.3505;
        a[85] = -4.12656;
        a[86] = -13.49;
        a[87] = 21.4259;
        a[88] = -1.34769;
        a[89] = 0.0879816;
        a[90] = -13.1653;
    }
    
    __shared__ dtype temp[ncoeff * ncells];
    
    for(int i = ind; i < ncells; i += numThreads)
    {
        dtype val = pow(10.0, 7 + 3.0 / ncells * i) * 1.0e-9;
        
        temp[i] = 1.0;
        temp[ncells + i] = 1 / val;
        temp[2 * ncells + i] = pow(val, (dtype)(-1.0 / 3.0));
        temp[3 * ncells + i] = pow(val, (dtype)(1.0 / 3.0));
        temp[4 * ncells + i] = val;
        temp[5 * ncells + i] = pow(val, (dtype)(5.0 / 3.0));
        temp[6 * ncells + i] = log(val);
        temp[7 * ncells + i] = 0.0;
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
    const int nsets = 13, ncells = 10, ncoeff = 7;
    
    // Results matrix
    float *lam;
    cudaMallocManaged(&lam, nsets * ncells * sizeof(float));
    
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
    exec<float, nsets, ncells, ncoeff><<<numBlocks, threadsPerBlock>>>(lam);
    
    // Print ln(lambda)
    cudaDeviceSynchronize();
    printf("lambda:\n");
    
    for(int i = 0; i < nsets; i++)
    {
        for(int j = 0; j < ncells; j++)
        {
            printf("%8.3f   ", lam[i * ncells + j]);
        }
        
        printf("\n");
    }
    
    return 0;
}

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
__global__ void rates(float *a, float *temp, float *lam, int nsets, int ncells, int ncoeff)
{
    int istart = blockIdx.x * blockDim.x + threadIdx.x;
    int istep = blockDim.x * gridDim.x;
    
    int jstart = blockIdx.y * blockDim.y + threadIdx.y;
    int jstep = blockDim.y * gridDim.y;
    
    for(int i = istart; i < nsets; i += istep)
    {
        for(int j = jstart; j < ncells; j += jstep)
        {
            float temp9 = temp[j] * 1.0e-9;
            
            for(int k = 0; k < ncoeff; k++)
            {
                switch(k)
                {
                    case 0:
                        lam[i * ncells + j] += a[i * ncoeff + k]; 
                        break;
                    case 6: 
                        lam[i * ncells + j] += a[i * ncoeff + k] * logf(temp9); 
                        break;
                    default: 
                        lam[i * ncells + j] += a[i * ncoeff + k] * powf(temp9, (2 * k - 5) / 3.0f); 
                        break;
                }
            }
        }
    }
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
        
    // Tensor dimensions
    int nsets = 4, ncells = 4, ncoeff = 8;
    // Loop variables
    int i, j;
    // Tensors
    float *a, *temp, *lam;
    
    /*********************************************************
    * Allocate memory for coefficients and initialize matrix *
    *********************************************************/
    cudaMallocManaged(&a, nsets * ncoeff * sizeof(float));
    
    printf("a:\n");
    
    for(i = 0; i < nsets; i++)
    {
        for(j = 0; j < ncoeff; j++)
        {
            if(j != 7)
            {
                a[i * ncoeff + j] = i * (ncoeff - 1) + j + 1;
            }
            else
            {
                a[i * ncoeff + j] = 0.0;
            }
            printf("%.3f\t", a[i * ncoeff + j]);
        }
        
        printf("\n");
    }
    
    printf("\n");
    
    /***********************************************
    * Do the same for the temperature of each cell *
    ***********************************************/
    cudaMallocManaged(&temp, ncells * sizeof(float));
    printf("temp:\n");
    
    for(i = 0; i < ncells; i++)
    {
        temp[i] = (i + 1) * 1e9;
        printf("%.3f\t", temp[i]);
    }
    
    printf("\n\n");
    
    /*******************************************
    * Allocate space for the summation results *
    *******************************************/
    cudaMallocManaged(&lam, nsets * ncells * sizeof(float));
    
    for(i = 0; i < nsets; i++)
    {
        for(j = 0; j < ncells; j++)
        {
            lam[i * ncells + j] = 0.0f;
        }
    }
    
    /****************************************************************
    * Compute ln(lambda) for each set and cell and print the result *
    *****************************************************************/
    dim3 threadsPerBlock(nsets, ncells);
    dim3 numBlocks(1, 1);
    
    cudaEventRecord(start);
    rates<<<numBlocks, threadsPerBlock>>>(a, temp, lam, nsets, ncells, ncoeff);
    cudaEventRecord(stop);
    
    cudaDeviceSynchronize();
    printf("lambda:\n");
    
    for(i = 0; i < nsets; i++)
    {
        for(j = 0; j < ncells; j++)
        {
            printf("%.3f\t", lam[i * ncells + j]);
        }
        
        printf("\n");
    }
    
    /*********************
    * Print elapsed time *
    **********************/
    cudaEventSynchronize(stop);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("\nTime elapsed: %.1f us\n", 1000 * elapsed);
    
    /**************
    * Free memory *
    **************/
    cudaFree(a);
    cudaFree(temp);
    cudaFree(lam);
    
    return 0;
}

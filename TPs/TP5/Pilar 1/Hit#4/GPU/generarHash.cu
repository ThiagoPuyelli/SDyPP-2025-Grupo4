#include "md5.cu"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <arpa/inet.h>

// Kernel de CUDA
__global__ void aplicarHashKernel(unsigned char* data, uint length, uint *a1, uint *b1, uint *c1, uint *d1) {
    md5_vfy(data, length, a1, b1, c1, d1);
}

uint32_t swapEndian(uint32_t value) {
    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}

void aplicarHash(unsigned char* data, uint length, uint *a1, uint *b1, uint *c1, uint *d1) {
    // Reservar memoria en el dispositivo
    unsigned char *d_data;
    uint *d_a1, *d_b1, *d_c1, *d_d1;
    
    cudaMalloc((void**)&d_data, length);
    cudaMalloc((void**)&d_a1, sizeof(uint));
    cudaMalloc((void**)&d_b1, sizeof(uint));
    cudaMalloc((void**)&d_c1, sizeof(uint));
    cudaMalloc((void**)&d_d1, sizeof(uint));
    
    // Copiar datos de host a device
    cudaMemcpy(d_data, data, length, cudaMemcpyHostToDevice);
    
    // Lanzar kernel - 1 bloque con 1 hilo (ajusta según necesites)
    aplicarHashKernel<<<1, 1>>>(d_data, length, d_a1, d_b1, d_c1, d_d1);
    
    // Copiar resultados de device a host
    cudaMemcpy(a1, d_a1, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1, d_b1, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(c1, d_c1, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(d1, d_d1, sizeof(uint), cudaMemcpyDeviceToHost);

    *a1 = swapEndian(*a1);
    *b1 = swapEndian(*b1);
    *c1 = swapEndian(*c1);
    *d1 = swapEndian(*d1);
    
    // Liberar memoria del dispositivo
    cudaFree(d_data);
    cudaFree(d_a1);
    cudaFree(d_b1);
    cudaFree(d_c1);
    cudaFree(d_d1);
}



int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s <cadena>\n", argv[0]);
        return 1;
    }
    unsigned char * texto = (unsigned char*) argv[1];

    uint v1, v2, v3, v4;
    aplicarHash(texto, strlen(argv[1]), &v1, &v2, &v3, &v4);
    printf("Hash: %08x %08x %08x %08x\n", v1, v2, v3, v4);
    
    return 0;
}
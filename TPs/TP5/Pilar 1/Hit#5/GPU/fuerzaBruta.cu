#include "md5.cu"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <arpa/inet.h>

// Kernel de CUDA para probar múltiples números en paralelo
__global__ void encontrarNumeroMagico(
    const unsigned char* cadenaOriginal,
    uint longitudCadena,
    const char* prefijoHash,
    uint longitudPrefijo,
    uint* numeroEncontrado,
    uint* hashResultadoA,
    uint* hashResultadoB,
    uint* hashResultadoC,
    uint* hashResultadoD,
    uint inicioRango,
    uint finRango
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint numeroPrueba = inicioRango + idx;

    if (numeroPrueba >= finRango) return;

    // Preparar buffer para la cadena + número (como string)
    unsigned char buffer[256];
    memcpy(buffer, cadenaOriginal, longitudCadena);

    // Convertir el número a string y concatenarlo
    char numStr[20];
    sprintf((char*)buffer + longitudCadena, "%u", numeroPrueba);
    uint nuevaLongitud = longitudCadena + strlen((char*)buffer + longitudCadena);

    // Calcular MD5
    uint a, b, c, d;
    md5_vfy(buffer, nuevaLongitud, &a, &b, &c, &d);

    // Convertir a big-endian para comparación
    a = ((a & 0x000000FF) << 24) | ((a & 0x0000FF00) << 8) | ((a & 0x00FF0000) >> 8) | ((a & 0xFF000000) >> 24);
    b = ((b & 0x000000FF) << 24) | ((b & 0x0000FF00) << 8) | ((b & 0x00FF0000) >> 8) | ((b & 0xFF000000) >> 24);
    c = ((c & 0x000000FF) << 24) | ((c & 0x0000FF00) << 8) | ((c & 0x00FF0000) >> 8) | ((c & 0xFF000000) >> 24);
    d = ((d & 0x000000FF) << 24) | ((d & 0x0000FF00) << 8) | ((d & 0x00FF0000) >> 8) | ((d & 0xFF000000) >> 24);

    // Verificar si el hash comienza con el prefijo deseado
    char hashStr[33];
    sprintf(hashStr, "%08x%08x%08x%08x", a, b, c, d);

    if (strncmp(hashStr, prefijoHash, longitudPrefijo) == 0) {
        *numeroEncontrado = numeroPrueba;
        *hashResultadoA = a;
        *hashResultadoB = b;
        *hashResultadoC = c;
        *hashResultadoD = d;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Uso: %s <hash_prefijo> <cadena> <rango_max>\n", argv[0]);
        return 1;
    }

    const char* prefijoHash = argv[1];
    const char* cadena = argv[2];
    uint rangoMax = atoi(argv[3]);

    // Configuración de CUDA
    uint threadsPorBloque = 256;
    uint bloques = (rangoMax + threadsPorBloque - 1) / threadsPorBloque;

    // Reservar memoria en GPU
    unsigned char* d_cadena;
    char* d_prefijoHash;
    uint* d_numeroEncontrado;
    uint* d_hashA, *d_hashB, *d_hashC, *d_hashD;

    cudaMalloc((void**)&d_cadena, strlen(cadena) + 1);
    cudaMalloc((void**)&d_prefijoHash, strlen(prefijoHash) + 1);
    cudaMalloc((void**)&d_numeroEncontrado, sizeof(uint));
    cudaMalloc((void**)&d_hashA, sizeof(uint));
    cudaMalloc((void**)&d_hashB, sizeof(uint));
    cudaMalloc((void**)&d_hashC, sizeof(uint));
    cudaMalloc((void**)&d_hashD, sizeof(uint));

    // Inicializar valores en GPU
    uint numeroInicial = 0;
    cudaMemcpy(d_cadena, cadena, strlen(cadena) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefijoHash, prefijoHash, strlen(prefijoHash) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_numeroEncontrado, &numeroInicial, sizeof(uint), cudaMemcpyHostToDevice);

    // Lanzar kernel
    encontrarNumeroMagico<<<bloques, threadsPorBloque>>>(
        d_cadena,
        strlen(cadena),
        d_prefijoHash,
        strlen(prefijoHash),
        d_numeroEncontrado,
        d_hashA,
        d_hashB,
        d_hashC,
        d_hashD,
        0,
        rangoMax
    );

    // Copiar resultados
    uint numeroEncontrado;
    uint hashA, hashB, hashC, hashD;
    cudaMemcpy(&numeroEncontrado, d_numeroEncontrado, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hashA, d_hashA, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hashB, d_hashB, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hashC, d_hashC, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hashD, d_hashD, sizeof(uint), cudaMemcpyDeviceToHost);

    // Mostrar resultados
    if (numeroEncontrado != 0) {
        printf("Número encontrado: %u\n", numeroEncontrado);
        printf("Hash MD5: %08x%08x%08x%08x\n", hashA, hashB, hashC, hashD);
    } else {
        printf("No se encontró un número en el rango 0-%u que cumpla la condición.\n", rangoMax);
    }

    // Liberar memoria
    cudaFree(d_cadena);
    cudaFree(d_prefijoHash);
    cudaFree(d_numeroEncontrado);
    cudaFree(d_hashA);
    cudaFree(d_hashB);
    cudaFree(d_hashC);
    cudaFree(d_hashD);

    return 0;
}
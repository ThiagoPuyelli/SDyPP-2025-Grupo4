#include "md5.cu"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>

// Convierte un entero a su representación hexadecimal (para GPU)
__device__ void uintToHex(uint val, char* hexStr) {
    const char* hexChars = "0123456789abcdef";
    for (int i = 7; i >= 0; --i) {
        hexStr[i] = hexChars[val & 0xF];
        val >>= 4;
    }
    hexStr[8] = '\0';
}

// Kernel de CUDA para buscar en un rango específico
__global__ void buscarEnRango(
    const char* cadena,
    uint lenCadena,
    const char* prefijo,
    uint lenPrefijo,
    unsigned long long inicioRango,
    unsigned long long finRango,
    unsigned long long* numeroEncontrado,
    uint* hashEncontrado
) {
    unsigned long long numero = inicioRango + blockIdx.x * blockDim.x + threadIdx.x;
    if (numero >= finRango) return;

    // Preparamos la cadena: "cadena" + "numero"
    char buffer[256];
    memcpy(buffer, cadena, lenCadena);

    // Convertimos el número a string (optimizado para GPU)
    unsigned long long temp = numero;
    uint lenNum = 0;
    do {
        buffer[lenCadena + lenNum++] = '0' + (temp % 10);
        temp /= 10;
    } while (temp > 0);

    // Invertimos el número (porque se generó al revés)
    for (uint i = 0; i < lenNum / 2; ++i) {
        char tmp = buffer[lenCadena + i];
        buffer[lenCadena + i] = buffer[lenCadena + lenNum - 1 - i];
        buffer[lenCadena + lenNum - 1 - i] = tmp;
    }

    uint lenTotal = lenCadena + lenNum;

    // Calculamos MD5
    uint a, b, c, d;
    md5_vfy((unsigned char*)buffer, lenTotal, &a, &b, &c, &d);

    // Convertimos el hash a hexadecimal
    char hashStr[33];
    uintToHex(a, hashStr);
    uintToHex(b, hashStr + 8);
    uintToHex(c, hashStr + 16);
    uintToHex(d, hashStr + 24);
    hashStr[32] = '\0';

    // Comparamos con el prefijo
    bool match = true;
    for (uint i = 0; i < lenPrefijo; ++i) {
        if (hashStr[i] != prefijo[i]) {
            match = false;
            break;
        }
    }

    // Si coincide, guardamos el resultado
    if (match) {
        *numeroEncontrado = numero;
        hashEncontrado[0] = a;
        hashEncontrado[1] = b;
        hashEncontrado[2] = c;
        hashEncontrado[3] = d;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Uso: %s <prefijo> <cadena> <inicio_rango> <fin_rango>\n", argv[0]);
        return 1;
    }

    const char* prefijo = argv[1];
    const char* cadena = argv[2];
    unsigned long long inicioRango = strtoull(argv[3], NULL, 10);
    unsigned long long finRango = strtoull(argv[4], NULL, 10);

    // Configuración de CUDA
    uint threadsPorBloque = 256;
    unsigned long long totalNumeros = finRango - inicioRango;
    uint bloques = (totalNumeros + threadsPorBloque - 1) / threadsPorBloque;

    // Reservamos memoria en la GPU
    char *d_cadena, *d_prefijo;
    unsigned long long *d_numeroEncontrado;
    uint *d_hashEncontrado;

    cudaMalloc(&d_cadena, strlen(cadena) + 1);
    cudaMalloc(&d_prefijo, strlen(prefijo) + 1);
    cudaMalloc(&d_numeroEncontrado, sizeof(unsigned long long));
    cudaMalloc(&d_hashEncontrado, 4 * sizeof(uint));

    // Inicializamos el resultado
    unsigned long long numeroInicial = 0;
    uint hashInicial[4] = {0};
    cudaMemcpy(d_cadena, cadena, strlen(cadena) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefijo, prefijo, strlen(prefijo) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_numeroEncontrado, &numeroInicial, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashEncontrado, hashInicial, 4 * sizeof(uint), cudaMemcpyHostToDevice);

    // Lanzamos el kernel
    buscarEnRango<<<bloques, threadsPorBloque>>>(
        d_cadena,
        strlen(cadena),
        d_prefijo,
        strlen(prefijo),
        inicioRango,
        finRango,
        d_numeroEncontrado,
        d_hashEncontrado
    );

    // Copiamos el resultado de vuelta al host
    unsigned long long numeroEncontrado;
    uint hashEncontrado[4];
    cudaMemcpy(&numeroEncontrado, d_numeroEncontrado, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hashEncontrado, d_hashEncontrado, 4 * sizeof(uint), cudaMemcpyDeviceToHost);

    // Mostramos el resultado
    if (numeroEncontrado != 0) {
        printf("Número encontrado: %llu\n", numeroEncontrado);
        printf("Hash MD5: ");
        for (int i = 0; i < 4; ++i) {
            printf("%08x", hashEncontrado[i]);
        }
        printf("\n");
    } else {
        printf("No se encontró un número en el rango %llu-%llu.\n", inicioRango, finRango);
    }

    // Liberamos memoria
    cudaFree(d_cadena);
    cudaFree(d_prefijo);
    cudaFree(d_numeroEncontrado);
    cudaFree(d_hashEncontrado);

    return 0;
}
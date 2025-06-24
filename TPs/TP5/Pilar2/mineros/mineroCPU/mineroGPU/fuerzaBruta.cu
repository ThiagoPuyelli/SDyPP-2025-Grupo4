#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include "md5.cu"

// Convierte un entero a su representación hexadecimal (para GPU)
__device__ void uintToHex(uint val, char* hexStr) {
    const char* hexChars = "0123456789abcdef";
    for (int i = 7; i >= 0; --i) {
        hexStr[i] = hexChars[val & 0xF];
        val >>= 4;
    }
    hexStr[8] = '\0';
}

__device__ bool compareHexPrefix(const uint8_t digest[16], const char* prefix, int prefixLength) {
    // Tabla de caracteres hexadecimales (minúsculas)
    const char hexChars[] = "0123456789abcdef";
    
    // Convertir cada byte del digest a 2 caracteres hex
    for (int i = 0; i < prefixLength; i++) {
        // Obtener el byte correspondiente (cada byte son 2 caracteres hex)
        int byteIndex = i / 2;
        if (byteIndex >= 16) return false; // Evitar overflow
        
        // Extraer el nibble (4 bits) correcto:
        // - Si i es par: primer nibble (>> 4)
        // - Si i es impar: segundo nibble (& 0x0F)
        uint8_t nibble = (i % 2 == 0) ? (digest[byteIndex] >> 4) : (digest[byteIndex] & 0x0F);
        char currentHexChar = hexChars[nibble];
        
        // Comparar con el carácter esperado en el prefijo
        if (currentHexChar != prefix[i]) {
            return false;
        }
    }
    return true;
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
    if (*numeroEncontrado != 0) return;
    unsigned long long numero = inicioRango + blockIdx.x * blockDim.x + threadIdx.x;
    if (numero >= finRango) return;

    // Buffer para el mensaje (cadena + número como string)
    const uint MAX_BUFFER = 512;
    char buffer[MAX_BUFFER];
    
    // Verificar que la cadena + número no exceda el buffer
    uint maxNumDigits = 20; // Máximo para unsigned long long
    if (lenCadena + maxNumDigits >= MAX_BUFFER) {
        return; // No hay espacio suficiente
    }

    // Copiar cadena
    memcpy(buffer, cadena, lenCadena);

    // Convertir número a string
    unsigned long long temp = numero;
    uint lenNum = 0;
    uint pos = lenCadena;
    
    // Manejar caso 0
    if (temp == 0) {
        buffer[pos++] = '0';
        lenNum = 1;
    } else {
        // Convertir dígitos (en orden inverso primero)
        while (temp > 0 && pos < MAX_BUFFER - 1) {
            buffer[pos++] = '0' + (temp % 10);
            temp /= 10;
            lenNum++;
        }
    }

    // Invertir los dígitos para obtener el orden correcto
    uint start = lenCadena;
    uint end = pos - 1;
    while (start < end) {
        char tmp = buffer[start];
        buffer[start] = buffer[end];
        buffer[end] = tmp;
        start++;
        end--;
    }

    uint lenTotal = lenCadena + lenNum;

    // 1. Calcular MD5 (esto está bien)
    uint8_t digest[16];
    cuda_md5((const uint8_t*)buffer, lenTotal, digest);
    
    // Convertir bytes a uint (manteniendo el orden del digest)
    uint a = (digest[0] << 24) | (digest[1] << 16) | (digest[2] << 8) | digest[3];
    uint b = (digest[4] << 24) | (digest[5] << 16) | (digest[6] << 8) | digest[7];
    uint c = (digest[8] << 24) | (digest[9] << 16) | (digest[10] << 8) | digest[11];
    uint d = (digest[12] << 24) | (digest[13] << 16) | (digest[14] << 8) | digest[15];
    
    // 3. Convertir a hexadecimal (versión segura)
    char hashStr[33];
    const char hexChars[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        hashStr[i*2]   = hexChars[(digest[i] >> 4) & 0x0F]; // Primer nibble
        hashStr[i*2+1] = hexChars[digest[i] & 0x0F];        // Segundo nibble
    }
    hashStr[32] = '\0';
    // Debug: imprimir entrada y hash
    char debugBuffer[MAX_BUFFER];
    memcpy(debugBuffer, buffer, lenTotal);
    debugBuffer[lenTotal] = '\0';

    // Comparar con prefijo
    bool match = false;
    if (compareHexPrefix(digest, prefijo, lenPrefijo)) {
        match = true;
    }
    //bool match = true;
    //for (uint i = 0; i < lenPrefijo; ++i) {
    //    char    
    //    if (digest[i] != prefijo[i]) {
    //        match = false;
    //        break;
    //    }
    //}

    // Escribir resultado de forma atómica
    if (match) {
        atomicExch(numeroEncontrado, numero);
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

    FILE *file = fopen("file.txt", "w");
    fprintf(file, "cadena: %s", cadena);

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

    // Verificamos errores del kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error después de lanzar el kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Esperamos a que termine el kernel antes de continuar
    cudaDeviceSynchronize();

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

    if (numeroEncontrado == 0) {
        return 1;
    }

    char hashHex[33];
    for (int i = 0; i < 4; ++i) {
        sprintf(&hashHex[i * 8], "%08x", hashEncontrado[i]);
    }
    hashHex[32] = '\0';

    FILE *json_file = fopen("json_output.txt", "w");
    fprintf(json_file, "{\"numero\": %llu, \"hash_md5_result\": \"%s\"}", numeroEncontrado, hashHex);

    // Liberamos memoria
    cudaFree(d_cadena);
    cudaFree(d_prefijo);
    cudaFree(d_numeroEncontrado);
    cudaFree(d_hashEncontrado);

    return 0;
}
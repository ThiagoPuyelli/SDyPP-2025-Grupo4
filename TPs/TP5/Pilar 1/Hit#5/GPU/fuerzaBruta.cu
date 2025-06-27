#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <string>
#include <iostream>

// Definimos nuestros tipos de manera segura
#ifndef UINT_DEFINED
#define UINT_DEFINED
typedef uint32_t my_uint;
#endif

#ifndef UCHAR_DEFINED
#define UCHAR_DEFINED
typedef uint8_t my_uchar;
#endif

// Actualizamos todas las referencias en el código:
#define uint my_uint
#define uchar my_uchar

#define block_size 64

#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define FF(a, b, c, d, x, s, ac)                    \
    {                                               \
        (a) += F((b), (c), (d)) + (x) + (uint)(ac); \
        (a) = ROTATE_LEFT((a), (s));                \
        (a) += (b);                                 \
    }

#define GG(a, b, c, d, x, s, ac)                    \
    {                                               \
        (a) += G((b), (c), (d)) + (x) + (uint)(ac); \
        (a) = ROTATE_LEFT((a), (s));                \
        (a) += (b);                                 \
    }

#define HH(a, b, c, d, x, s, ac)                    \
    {                                               \
        (a) += H((b), (c), (d)) + (x) + (uint)(ac); \
        (a) = ROTATE_LEFT((a), (s));                \
        (a) += (b);                                 \
    }

#define II(a, b, c, d, x, s, ac)                    \
    {                                               \
        (a) += I((b), (c), (d)) + (x) + (uint)(ac); \
        (a) = ROTATE_LEFT((a), (s));                \
        (a) += (b);                                 \
    }

__device__ constexpr uchar padding[block_size] = { 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

__device__ uint byteswap(uint word)
{
    return ((word >> 24) & 0x000000FF) | ((word >> 8) & 0x0000FF00) | ((word << 8) & 0x00FF0000) | ((word << 24) & 0xFF000000);
}

// Función para convertir un valor a hexadecimal (versión device)
__device__ void to_hex(char* output, uint value) {
    const char* hex_chars = "0123456789abcdef";
    for (int i = 7; i >= 0; --i) {
        output[i] = hex_chars[value & 0xf];
        value >>= 4;
    }
}

uint host_byteswap(uint word)
{
    return ((word >> 24) & 0x000000FF) | ((word >> 8) & 0x0000FF00) | ((word << 8) & 0x00FF0000) | ((word << 24) & 0xFF000000);
}

// Variables globales en device memory
__device__ int found = 0;  // Cambiado de bool a int para atomicCAS
__device__ uint found_nonce = 0;
__device__ uint found_hash[4];
__device__ char found_input[256];
__device__ unsigned long long total_hashes = 0;

__device__ void transform(uint state[4], const uchar block[block_size])
{
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint x[16];

    for (uint i = 0, j = 0; j < block_size && i < 16; i++, j += 4)
    {
        x[i] = (uint)block[j] | ((uint)block[j + 1] << 8) | ((uint)block[j + 2] << 16) | ((uint)block[j + 3] << 24);
    }

    FF(a, b, c, d, x[0], S11, 0xd76aa478);
    FF(d, a, b, c, x[1], S12, 0xe8c7b756);
    FF(c, d, a, b, x[2], S13, 0x242070db);
    FF(b, c, d, a, x[3], S14, 0xc1bdceee);
    FF(a, b, c, d, x[4], S11, 0xf57c0faf);
    FF(d, a, b, c, x[5], S12, 0x4787c62a);
    FF(c, d, a, b, x[6], S13, 0xa8304613);
    FF(b, c, d, a, x[7], S14, 0xfd469501);
    FF(a, b, c, d, x[8], S11, 0x698098d8);
    FF(d, a, b, c, x[9], S12, 0x8b44f7af);
    FF(c, d, a, b, x[10], S13, 0xffff5bb1);
    FF(b, c, d, a, x[11], S14, 0x895cd7be);
    FF(a, b, c, d, x[12], S11, 0x6b901122);
    FF(d, a, b, c, x[13], S12, 0xfd987193);
    FF(c, d, a, b, x[14], S13, 0xa679438e);
    FF(b, c, d, a, x[15], S14, 0x49b40821);

    GG(a, b, c, d, x[1], S21, 0xf61e2562);
    GG(d, a, b, c, x[6], S22, 0xc040b340);
    GG(c, d, a, b, x[11], S23, 0x265e5a51);
    GG(b, c, d, a, x[0], S24, 0xe9b6c7aa);
    GG(a, b, c, d, x[5], S21, 0xd62f105d);
    GG(d, a, b, c, x[10], S22, 0x2441453);
    GG(c, d, a, b, x[15], S23, 0xd8a1e681);
    GG(b, c, d, a, x[4], S24, 0xe7d3fbc8);
    GG(a, b, c, d, x[9], S21, 0x21e1cde6);
    GG(d, a, b, c, x[14], S22, 0xc33707d6);
    GG(c, d, a, b, x[3], S23, 0xf4d50d87);
    GG(b, c, d, a, x[8], S24, 0x455a14ed);
    GG(a, b, c, d, x[13], S21, 0xa9e3e905);
    GG(d, a, b, c, x[2], S22, 0xfcefa3f8);
    GG(c, d, a, b, x[7], S23, 0x676f02d9);
    GG(b, c, d, a, x[12], S24, 0x8d2a4c8a);

    HH(a, b, c, d, x[5], S31, 0xfffa3942);
    HH(d, a, b, c, x[8], S32, 0x8771f681);
    HH(c, d, a, b, x[11], S33, 0x6d9d6122);
    HH(b, c, d, a, x[14], S34, 0xfde5380c);
    HH(a, b, c, d, x[1], S31, 0xa4beea44);
    HH(d, a, b, c, x[4], S32, 0x4bdecfa9);
    HH(c, d, a, b, x[7], S33, 0xf6bb4b60);
    HH(b, c, d, a, x[10], S34, 0xbebfbc70);
    HH(a, b, c, d, x[13], S31, 0x289b7ec6);
    HH(d, a, b, c, x[0], S32, 0xeaa127fa);
    HH(c, d, a, b, x[3], S33, 0xd4ef3085);
    HH(b, c, d, a, x[6], S34, 0x4881d05);
    HH(a, b, c, d, x[9], S31, 0xd9d4d039);
    HH(d, a, b, c, x[12], S32, 0xe6db99e5);
    HH(c, d, a, b, x[15], S33, 0x1fa27cf8);
    HH(b, c, d, a, x[2], S34, 0xc4ac5665);

    II(a, b, c, d, x[0], S41, 0xf4292244);
    II(d, a, b, c, x[7], S42, 0x432aff97);
    II(c, d, a, b, x[14], S43, 0xab9423a7);
    II(b, c, d, a, x[5], S44, 0xfc93a039);
    II(a, b, c, d, x[12], S41, 0x655b59c3);
    II(d, a, b, c, x[3], S42, 0x8f0ccc92);
    II(c, d, a, b, x[10], S43, 0xffeff47d);
    II(b, c, d, a, x[1], S44, 0x85845dd1);
    II(a, b, c, d, x[8], S41, 0x6fa87e4f);
    II(d, a, b, c, x[15], S42, 0xfe2ce6e0);
    II(c, d, a, b, x[6], S43, 0xa3014314);
    II(b, c, d, a, x[13], S44, 0x4e0811a1);
    II(a, b, c, d, x[4], S41, 0xf7537e82);
    II(d, a, b, c, x[11], S42, 0xbd3af235);
    II(c, d, a, b, x[2], S43, 0x2ad7d2bb);
    II(b, c, d, a, x[9], S44, 0xeb86d391);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

__device__ void md5(const uchar* data, const uint size, uint result[4])
{
    uint state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 }, i;

    for (i = 0; i + block_size <= size; i += block_size)
    {
        transform(state, data + i);
    }

    uint size_in_bits = size << 3;
    uchar buffer[block_size];

    memcpy(buffer, data + i, size - i);
    memcpy(buffer + size - i, padding, block_size - (size - i));
    memcpy(buffer + block_size - (2 * sizeof(uint)), &size_in_bits, sizeof(uint));

    transform(state, buffer);

    memcpy(result, state, 4 * sizeof(uint));
}

// Variables globales en device memory
//__device__ bool found = false;
//__device__ uint found_nonce = 0;
//__device__ uint found_hash[4];
//__device__ char found_input[256];
//__device__ unsigned long long total_hashes = 0;

// Kernel para buscar el nonce
__global__ void find_nonce_kernel(const char* input_str, uint input_len, const char* prefix, uint prefix_len, uint max_nonce) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    
    // Cada hilo prueba nonces desde idx hasta max_nonce, incrementando por stride
    for (uint nonce = idx; nonce < max_nonce && atomicAdd(&found, 0) == 0; nonce += stride) {
        // Construir el mensaje: input_str + nonce (como string)
        char message[256];
        uint message_len = 0;
        
        // Copiar el string de entrada
        for (uint i = 0; i < input_len; i++) {
            message[message_len++] = input_str[i];
        }
        
        // Convertir el nonce a string y añadirlo
        uint temp = nonce;
        char nonce_str[20];
        uint nonce_str_len = 0;
        
        if (temp == 0) {
            nonce_str[nonce_str_len++] = '0';
        } else {
            while (temp > 0) {
                nonce_str[nonce_str_len++] = '0' + (temp % 10);
                temp /= 10;
            }
            // Invertir el string
            for (uint i = 0; i < nonce_str_len / 2; i++) {
                char tmp = nonce_str[i];
                nonce_str[i] = nonce_str[nonce_str_len - 1 - i];
                nonce_str[nonce_str_len - 1 - i] = tmp;
            }
        }
        
        for (uint i = 0; i < nonce_str_len; i++) {
            message[message_len++] = nonce_str[i];
        }
        
        // Calcular MD5
        uint hash[4];
        md5((const uchar*)message, message_len, hash);
        
        // Contador de hashes (solo un hilo actualiza para evitar conflictos)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            atomicAdd(&total_hashes, 1);
        }
        
        // Convertir hash a string hexadecimal (sin usar sprintf)
        char hash_str[33];
        for (int i = 0; i < 4; i++) {
            uint swapped = byteswap(hash[i]);
            to_hex(hash_str + i * 8, swapped);
        }
        hash_str[32] = '\0';
        
        // Verificar si el hash comienza con el prefijo
        bool match = true;
        for (uint i = 0; i < prefix_len; i++) {
            if (hash_str[i] != prefix[i]) {
                match = false;
                break;
            }
        }
        
        if (match) {
            // Usamos atomicCAS para asegurar que solo un hilo escriba los resultados
            int expected = 0;
            if (atomicCAS(&found, expected, 1) == 0) {
                found_nonce = nonce;
                for (int i = 0; i < 4; i++) {
                    found_hash[i] = hash[i];
                }
                for (uint i = 0; i < message_len; i++) {
                    found_input[i] = message[i];
                }
                found_input[message_len] = '\0';
            }
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: %s <prefijo> <cadena> <maximo>\n", argv[0]);
        printf("Ejemplo: %s 000000 hola 1000000\n", argv[0]);
        return 1;
    }
    
    const char* prefix = argv[1];
    const char* input_str = argv[2];
    uint max_nonce = atoi(argv[3]);
    uint prefix_len = strlen(prefix);
    uint input_len = strlen(input_str);
    
    // Validar el prefijo (debe ser hexadecimal)
    for (uint i = 0; i < prefix_len; i++) {
        if (!isxdigit(prefix[i])) {
            printf("Error: El prefijo debe contener solo caracteres hexadecimales (0-9, a-f)\n");
            return 1;
        }
    }
    
    // Copiar datos al dispositivo
    char* d_input_str;
    char* d_prefix;
    
    cudaMalloc((void**)&d_input_str, input_len + 1);
    cudaMalloc((void**)&d_prefix, prefix_len + 1);
    
    cudaMemcpy(d_input_str, input_str, input_len + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix, prefix_len + 1, cudaMemcpyHostToDevice);
    
    // Configurar kernel
    int threads_per_block = 256;
    int blocks_per_grid = (max_nonce + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 65535) blocks_per_grid = 65535;
    
    printf("Buscando nonce para '%s' con prefijo '%s' (max nonce: %u)\n", input_str, prefix, max_nonce);
    printf("Configuración del kernel: %d bloques x %d hilos\n", blocks_per_grid, threads_per_block);
    
    // Lanzar kernel
    find_nonce_kernel<<<blocks_per_grid, threads_per_block>>>(d_input_str, input_len, d_prefix, prefix_len, max_nonce);
    
    // Esperar a que termine el kernel
    cudaDeviceSynchronize();
    
    // Copiar resultados de vuelta
    int h_found;
    uint h_nonce;
    uint h_hash[4];
    char h_input[256];
    unsigned long long h_total_hashes;
    
    cudaMemcpyFromSymbol(&h_found, found, sizeof(int));
    cudaMemcpyFromSymbol(&h_nonce, found_nonce, sizeof(uint));
    cudaMemcpyFromSymbol(&h_hash, found_hash, 4 * sizeof(uint));
    cudaMemcpyFromSymbol(&h_input, found_input, 256);
    cudaMemcpyFromSymbol(&h_total_hashes, total_hashes, sizeof(unsigned long long));
    
    // Mostrar resultados
    if (h_found) {
        // Convertir hash a string hexadecimal
        char hash_str[33];
        for (int i = 0; i < 4; i++) {
            uint swapped = host_byteswap(h_hash[i]);
            sprintf(hash_str + i * 8, "%08x", swapped);
        }
        
        printf("\nNonce encontrado!\n");
        printf("Input: %s\n", h_input);
        printf("Nonce: %u\n", h_nonce);
        printf("Hash MD5: %s\n", hash_str);
    } else {
        printf("\nNo se encontro un nonce valido en el rango especificado (0-%u)\n", max_nonce);
    }
    
    // Liberar memoria
    cudaFree(d_input_str);
    cudaFree(d_prefix);
    
    return 0;
}
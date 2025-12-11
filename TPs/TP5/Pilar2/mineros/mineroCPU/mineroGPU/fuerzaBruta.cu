#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <string>
#include <iostream>
#include <chrono>
#include <fstream> // Necesario para escribir el output JSON en el host

// Definimos nuestros tipos de manera segura
typedef uint32_t my_uint;
typedef uint8_t my_uchar;
typedef uint64_t my_uint64; // Tipo para el nonce de 64 bits

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

// Rellenado MD5
__device__ constexpr uchar padding[block_size] = { 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

// Byte swap para little-endian
__device__ uint byteswap(uint word)
{
    return ((word >> 24) & 0x000000FF) | ((word >> 8) & 0x0000FF00) | ((word << 8) & 0x00FF0000) | ((word << 24) & 0xFF000000);
}

// MD5 Transformación de un bloque (función completa)
__device__ void transform(uint state[4], const uchar block[block_size])
{
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint x[16];

    // Cargar bloque a little-endian
    for (uint i = 0, j = 0; j < block_size && i < 16; i++, j += 4)
    {
        x[i] = (uint)block[j] | ((uint)block[j + 1] << 8) | ((uint)block[j + 2] << 16) | ((uint)block[j + 3] << 24);
    }

    // Rondas MD5... (código abreviado por espacio, es el mismo que proporcionaste)
    FF(a, b, c, d, x[0], S11, 0xd76aa478); FF(d, a, b, c, x[1], S12, 0xe8c7b756); FF(c, d, a, b, x[2], S13, 0x242070db); FF(b, c, d, a, x[3], S14, 0xc1bdceee); FF(a, b, c, d, x[4], S11, 0xf57c0faf); FF(d, a, b, c, x[5], S12, 0x4787c62a); FF(c, d, a, b, x[6], S13, 0xa8304613); FF(b, c, d, a, x[7], S14, 0xfd469501); FF(a, b, c, d, x[8], S11, 0x698098d8); FF(d, a, b, c, x[9], S12, 0x8b44f7af); FF(c, d, a, b, x[10], S13, 0xffff5bb1); FF(b, c, d, a, x[11], S14, 0x895cd7be); FF(a, b, c, d, x[12], S11, 0x6b901122); FF(d, a, b, c, x[13], S12, 0xfd987193); FF(c, d, a, b, x[14], S13, 0xa679438e); FF(b, c, d, a, x[15], S14, 0x49b40821);
    GG(a, b, c, d, x[1], S21, 0xf61e2562); GG(d, a, b, c, x[6], S22, 0xc040b340); GG(c, d, a, b, x[11], S23, 0x265e5a51); GG(b, c, d, a, x[0], S24, 0xe9b6c7aa); GG(a, b, c, d, x[5], S21, 0xd62f105d); GG(d, a, b, c, x[10], S22, 0x2441453); GG(c, d, a, b, x[15], S23, 0xd8a1e681); GG(b, c, d, a, x[4], S24, 0xe7d3fbc8); GG(a, b, c, d, x[9], S21, 0x21e1cde6); GG(d, a, b, c, x[14], S22, 0xc33707d6); GG(c, d, a, b, x[3], S23, 0xf4d50d87); GG(b, c, d, a, x[8], S24, 0x455a14ed); GG(a, b, c, d, x[13], S21, 0xa9e3e905); GG(d, a, b, c, x[2], S22, 0xfcefa3f8); GG(c, d, a, b, x[7], S23, 0x676f02d9); GG(b, c, d, a, x[12], S24, 0x8d2a4c8a);
    HH(a, b, c, d, x[5], S31, 0xfffa3942); HH(d, a, b, c, x[8], S32, 0x8771f681); HH(c, d, a, b, x[11], S33, 0x6d9d6122); HH(b, c, d, a, x[14], S34, 0xfde5380c); HH(a, b, c, d, x[1], S31, 0xa4beea44); HH(d, a, b, c, x[4], S32, 0x4bdecfa9); HH(c, d, a, b, x[7], S33, 0xf6bb4b60); HH(b, c, d, a, x[10], S34, 0xbebfbc70); HH(a, b, c, d, x[13], S31, 0x289b7ec6); HH(d, a, b, c, x[0], S32, 0xeaa127fa); HH(c, d, a, b, x[3], S33, 0xd4ef3085); HH(b, c, d, a, x[6], S34, 0x4881d05); HH(a, b, c, d, x[9], S31, 0xd9d4d039); HH(d, a, b, c, x[12], S32, 0xe6db99e5); HH(c, d, a, b, x[15], S33, 0x1fa27cf8); HH(b, c, d, a, x[2], S34, 0xc4ac5665);
    II(a, b, c, d, x[0], S41, 0xf4292244); II(d, a, b, c, x[7], S42, 0x432aff97); II(c, d, a, b, x[14], S43, 0xab9423a7); II(b, c, d, a, x[5], S44, 0xfc93a039); II(a, b, c, d, x[12], S41, 0x655b59c3); II(d, a, b, c, x[3], S42, 0x8f0ccc92); II(c, d, a, b, x[10], S43, 0xffeff47d); II(b, c, d, a, x[1], S44, 0x85845dd1); II(a, b, c, d, x[8], S41, 0x6fa87e4f); II(d, a, b, c, x[15], S42, 0xfe2ce6e0); II(c, d, a, b, x[6], S43, 0xa3014314); II(b, c, d, a, x[13], S44, 0x4e0811a1); II(a, b, c, d, x[4], S41, 0xf7537e82); II(d, a, b, c, x[11], S42, 0xbd3af235); II(c, d, a, b, x[2], S43, 0x2ad7d2bb); II(b, c, d, a, x[9], S44, 0xeb86d391);

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

// Función MD5 completa (device function)
__device__ void md5(const uchar* data, const uint size, uint result[4])
{
    uint state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 }, i;

    // Procesar bloques completos
    for (i = 0; i + block_size <= size; i += block_size)
    {
        transform(state, data + i);
    }

    // Manejar el último bloque con padding
    uint size_in_bits = size << 3;
    uchar buffer[block_size];

    // Copiar el resto de los datos
    memcpy(buffer, data + i, size - i);
    // Añadir el bit 1 (0x80) y ceros
    memcpy(buffer + size - i, padding, block_size - (size - i));
    // Añadir la longitud en bits (little-endian)
    memcpy(buffer + block_size - (2 * sizeof(uint)), &size_in_bits, sizeof(uint));

    transform(state, buffer);

    memcpy(result, state, 4 * sizeof(uint));
}

// Función para convertir un valor hash MD5 de 4*uint a string hexadecimal (device)
__device__ void to_hex_string(char* output, const uint hash[4]) {
    const char* hex_chars = "0123456789abcdef";
    for(int k=0; k<4; ++k) {
        uint value = hash[k];
        for (int i = 7; i >= 0; --i) {
            output[k * 8 + i] = hex_chars[value & 0xf];
            value >>= 4;
        }
    }
    output[32] = '\0'; // Null terminator
}


// Variables globales en device memory para almacenar el resultado encontrado
__device__ int found = 0;
__device__ my_uint64 found_nonce = 0;
__device__ char found_hash_hex[33]; // 32 chars + null terminator


// Kernel para buscar el nonce
__global__ void find_nonce_kernel(const char* input_str, uint input_len, const char* prefix, uint prefix_len, my_uint64 min_nonce, my_uint64 max_nonce) {
    my_uint64 idx = min_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    my_uint64 stride = (my_uint64)gridDim.x * blockDim.x;
    
    // Asumimos un tamaño máximo razonable para el mensaje
    uchar message[256];
    
    // Copiar el string de entrada una vez (es constante para todos los hilos)
    for (uint i = 0; i < input_len; i++) {
        message[i] = input_str[i];
    }
    
    for (my_uint64 nonce = idx; nonce < max_nonce; nonce += stride) {
        if (atomicAdd(&found, 0) != 0) break;

        uint current_message_len = input_len;
        
        // Convertir el nonce (64 bits) a string decimal y añadirlo
        char nonce_str[22]; // Suficiente para 2^64 en decimal
        int n_len = 0;
        my_uint64 temp = nonce;
        if (temp == 0) {
            nonce_str[n_len++] = '0';
        } else {
            while (temp > 0) {
                nonce_str[n_len++] = (temp % 10) + '0';
                temp /= 10;
            }
            // Invertir el string
            for(int i = 0; i < n_len / 2; ++i) {
                char t = nonce_str[i];
                nonce_str[i] = nonce_str[n_len - i - 1];
                nonce_str[n_len - i - 1] = t;
            }
        }

        for (int i = 0; i < n_len; i++) {
            message[current_message_len++] = (uchar)nonce_str[i];
        }
        
        // Calcular el hash MD5
        uint hash_result[4];
        md5(message, current_message_len, hash_result);

        // Convertir el hash a formato hexadecimal para la comparación
        char hex_hash[33];
        to_hex_string(hex_hash, hash_result);

        // Comprobar el prefijo
        bool prefix_match = true;
        for (uint i = 0; i < prefix_len; i++) {
            if (hex_hash[i] != prefix[i]) {
                prefix_match = false;
                break;
            }
        }

        // Si hay coincidencia, almacenar el resultado de manera segura (atomic)
        if (prefix_match) {
            if (atomicCAS(&found, 0, 1) == 0) {
                found_nonce = nonce;
                memcpy(found_hash_hex, hex_hash, 33 * sizeof(char));
            }
        }
    }
}

// Función principal Host para manejar la E/S y lanzar el kernel
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Uso: " << argv[0] << " <prefix_target> <input_string> <min_nonce> <max_nonce>" << std::endl;
        return 1;
    }

    // 1. Parsear argumentos de la línea de comandos
    const std::string prefix_target_str(argv[1]);
    const std::string input_string_str(argv[2]);
    // Usamos stoull (string to unsigned long long) para 64 bits
    my_uint64 min_nonce_host = std::stoull(argv[3]);
    my_uint64 max_nonce_host = std::stoull(argv[4]);

    uint input_len_host = input_string_str.length();
    uint prefix_len_host = prefix_target_str.length();

    // 2. Asignación de memoria en Device (GPU)
    char *d_input_str, *d_prefix;
    size_t input_bytes = input_len_host + 1;
    size_t prefix_bytes = prefix_len_host + 1;

    cudaMalloc(&d_input_str, input_bytes);
    cudaMalloc(&d_prefix, prefix_bytes);

    // 3. Copiar datos del Host al Device
    cudaMemcpy(d_input_str, input_string_str.c_str(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix_target_str.c_str(), prefix_bytes, cudaMemcpyHostToDevice);

    // 4. Configurar y lanzar el Kernel
    dim3 blocks(128); // Puedes ajustar estos valores para optimizar
    dim3 threads(256);
    
    find_nonce_kernel<<<blocks, threads>>>(d_input_str, input_len_host, d_prefix, prefix_len_host, min_nonce_host, max_nonce_host);

    cudaDeviceSynchronize(); // Esperar a que todos los hilos terminen

    // 5. Recuperar resultados del Device al Host
    int h_found = 0;
    my_uint64 h_found_nonce = 0;
    char h_found_hash_hex[33];

    cudaMemcpy(&h_found, &found, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_found) {
        cudaMemcpy(&h_found_nonce, &found_nonce, sizeof(my_uint64), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_found_hash_hex, &found_hash_hex, sizeof(h_found_hash_hex), cudaMemcpyDeviceToHost);
    }

    // 6. Limpieza de memoria Device
    cudaFree(d_input_str);
    cudaFree(d_prefix);

    // 7. Reportar el resultado al script de Python mediante el archivo JSON
    std::ofstream output_file("json_output.txt");
    if (output_file.is_open()) {
        if (h_found) {
            // Escribir el JSON exactamente como Python espera
            output_file << "{\"numero\": " << h_found_nonce << ", \"hash_md5_result\": \"" << h_found_hash_hex << "\"}" << std::endl;
        } else {
            // Escribir JSON vacío si no se encuentra
             output_file << "{\"numero\": 0, \"hash_md5_result\": \"\"}" << std::endl;
        }
        output_file.close();
    } else {
        std::cerr << "Error opening json_output.txt for writing." << std::endl;
        return 1;
    }

    return 0;
}

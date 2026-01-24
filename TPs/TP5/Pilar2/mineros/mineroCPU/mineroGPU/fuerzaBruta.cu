#include <stdlib.h>
#include <iostream>
#include <string>

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ================= CONFIG ================= */

// Reducido para no reventar el stack del kernel
#define MAX_MESSAGE_LEN 256
#define HASH_SIZE 16

/* ================= DEVICE CONSTANTS ================= */

__device__ __constant__ char d_prefix[64];
__device__ __constant__ char d_base[256];
__device__ __constant__ int d_prefix_len;
__device__ __constant__ int d_base_len;

__device__ uint64_t d_found_nonce;
__device__ char d_found_hash[33];
__device__ int d_found;

/* ================= HELPER (CPU) ================= */

std::string base64_decode(const std::string &input) {
    BIO *bio, *b64;
    char *buffer = (char *)malloc(input.size());
    memset(buffer, 0, input.size());

    b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_new_mem_buf(input.data(), input.size());
    bio = BIO_push(b64, bio);

    int decoded_size = BIO_read(bio, buffer, input.size());
    BIO_free_all(bio);

    std::string result(buffer, decoded_size);
    free(buffer);
    return result;
}

/* ================= MD5 CONSTANTS ================= */

__device__ __constant__ uint32_t k[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,
    0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,
    0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,
    0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,
    0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,
    0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,
    0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,
    0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,
    0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

__device__ __constant__ uint32_t r[64] = {
    7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
    5,9,14,20, 5,9,14,20, 5,9,14,20, 5,9,14,20,
    4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
    6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
};

__device__ inline uint32_t ROT(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

/* ================= MD5 MULTIBLOCK ================= */

__device__ void md5_multiblock(const uint8_t *msg, int len, uint8_t *out) {
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xefcdab89;
    uint32_t h2 = 0x98badcfe;
    uint32_t h3 = 0x10325476;

    uint8_t buffer[MAX_MESSAGE_LEN + 64];
    memcpy(buffer, msg, len);

    buffer[len] = 0x80;
    int padded_len = len + 1;

    while ((padded_len % 64) != 56) {
        buffer[padded_len++] = 0;
    }

    uint64_t bits = (uint64_t)len * 8;
    memcpy(buffer + padded_len, &bits, 8);
    padded_len += 8;

    for (int offset = 0; offset < padded_len; offset += 64) {
        uint32_t w[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            w[i] =
                (uint32_t)buffer[offset + i*4 + 0] |
                ((uint32_t)buffer[offset + i*4 + 1] << 8) |
                ((uint32_t)buffer[offset + i*4 + 2] << 16) |
                ((uint32_t)buffer[offset + i*4 + 3] << 24);
        }
        uint32_t a = h0, b = h1, c = h2, d = h3;

        for (int i = 0; i < 64; i++) {
            uint32_t f, g;

            if (i < 16)      { f = (b & c) | (~b & d); g = i; }
            else if (i < 32) { f = (d & b) | (~d & c); g = (5*i + 1) % 16; }
            else if (i < 48) { f = b ^ c ^ d;          g = (3*i + 5) % 16; }
            else             { f = c ^ (b | ~d);       g = (7*i) % 16; }

            uint32_t temp = d;
            d = c;
            c = b;
            b = b + ROT(a + f + k[i] + w[g], r[i]);
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
    }

    memcpy(out,      &h0, 4);
    memcpy(out + 4,  &h1, 4);
    memcpy(out + 8,  &h2, 4);
    memcpy(out + 12, &h3, 4);
}

/* ================= KERNEL ================= */

__global__ void kernel(uint64_t start, uint64_t end) {

    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    for (
        uint64_t idx = start + blockIdx.x * blockDim.x + threadIdx.x;
        idx <= end;
        idx += stride
    ) {

        if (atomicAdd(&d_found, 0)) return;

        char buffer[MAX_MESSAGE_LEN];

        int len = d_base_len;
        if (len + 20 >= MAX_MESSAGE_LEN) return;

        memcpy(buffer, d_base, len);

        uint64_t n = idx;
        char tmp[32];
        int t = 0;
        do {
            tmp[t++] = '0' + (n % 10);
            n /= 10;
        } while (n);

        for (int i = 0; i < t; i++) {
            buffer[len + i] = tmp[t - 1 - i];
        }

        int total_len = len + t;

        uint8_t digest[16];
        md5_multiblock((uint8_t *)buffer, total_len, digest);

        const char *hex = "0123456789abcdef";
        char hexhash[33];
        for (int i = 0; i < 16; i++) {
            hexhash[i * 2]     = hex[digest[i] >> 4];
            hexhash[i * 2 + 1] = hex[digest[i] & 0xF];
        }
        hexhash[32] = 0;

        for (int i = 0; i < d_prefix_len; i++) {
            if (hexhash[i] != d_prefix[i]) goto next;
        }

        if (atomicCAS(&d_found, 0, 1) == 0) {
            d_found_nonce = idx;
            for (int i = 0; i < 33; i++) {
                d_found_hash[i] = hexhash[i];
            }
            __threadfence();
        }
        return;

    next:
        ;
    }
}

/* ================= MAIN ================= */

int main(int argc, char **argv) {
    if (argc != 5) return 1;

    std::string decoded = base64_decode(argv[2]);
    int base_len = decoded.size();

    if (base_len >= MAX_MESSAGE_LEN) {
        fprintf(stderr, "Base message too long\n");
        return 1;
    }

    cudaMemcpyToSymbol(d_base, decoded.c_str(), base_len);
    cudaMemcpyToSymbol(d_base_len, &base_len, sizeof(int));

    int prefix_len = strlen(argv[1]);
    cudaMemcpyToSymbol(d_prefix, argv[1], prefix_len + 1);
    cudaMemcpyToSymbol(d_prefix_len, &prefix_len, sizeof(int));

    uint64_t min = strtoull(argv[3], NULL, 10);
    uint64_t max = strtoull(argv[4], NULL, 10);

    int zero = 0;
    uint64_t zero64 = 0;
    char empty_hash[33] = {0};

    cudaMemcpyToSymbol(d_found, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_found_nonce, &zero64, sizeof(uint64_t));
    cudaMemcpyToSymbol(d_found_hash, empty_hash, 33);

    int threads = 256;
    uint64_t total = max - min + 1;
    int blocks = (int)((total + threads - 1) / threads);

    kernel<<<blocks, threads>>>(min, max);
    cudaDeviceSynchronize();

    int found;
    cudaMemcpyFromSymbol(&found, d_found, sizeof(int));

    if (found) {
        uint64_t nonce;
        char hash[33];
        cudaMemcpyFromSymbol(&nonce, d_found_nonce, sizeof(uint64_t));
        cudaMemcpyFromSymbol(hash, d_found_hash, 33);
        printf("{\"found\": true, \"nonce\": %llu, \"hash\": \"%s\"}\n",
               (unsigned long long)nonce, hash);
    } else {
        printf("{\"found\": false}\n");
    }

    return 0;
}

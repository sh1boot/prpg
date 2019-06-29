#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  uint64_t mat[64];
  uint64_t mult;
} stage_t;

#define NSTAGES 4
typedef struct {
  stage_t stage[NSTAGES];
  stage_t istage[NSTAGES];
  uint64_t z;
  uint64_t max;
  uint64_t m0, m1;
  uint64_t im0, im1;
  uint8_t s0, s1, s2;
  uint8_t bits;
} state_t;


// Just a source of some random bits.  This implementation makes no guarantee
// about the size of its state and shouldn't generally be used.
uint64_t rand64(void) {
  return (uint64_t)rand() << 45 ^ (uint64_t)rand() << 20 ^ rand() >> 5;
}


// Calculate the multiplicative inverse, mod-2**64, of x such that:
//     (x * multinv(x) & 0xffffffffffffffff) == 1
// This result is also useful for smaller power-of-two ranges.
static inline uint64_t multinv(uint64_t x) {
  uint64_t ix = x;
  while ((ix * x) != 1) {
    ix *= 2 - ix * x;
  }
  return ix;
}


// Multiply square matrices of bits (up to 64x64 bits) together.
static inline void matmul(uint64_t out[64], uint64_t a[64], uint64_t b[64], int bits) {
  for (int i = 0; i < bits; i++) {
    uint64_t m = a[i];
    uint64_t r = 0;
    for (int j = 0; j < bits; j++) {
      if (m & 1) r ^= b[j];
      m >>= 1;
    }
    out[i] = r;
  }
}


// Generate a random matrix and its inverse.  There are invertible matrices
// which this will not generate, but the subset that it does generate is easy
// to think about.  It's the product of upper and lower unitriangular matrices.
static inline void invertiblematrix(uint64_t* out, uint64_t* inv, int bits) {
  uint64_t a[64], b[64];
  uint64_t m = ((uint64_t)2 << (bits - 1)) - 1;

  // Generate upper and lower unitriangular matrices.
  for (int i = 0; i < bits; i++) {
    uint64_t r = rand64();
    a[i] = ((r + r + 1) << i) & m;
    b[i] = (r | ((uint64_t)1 << 63)) >> (63 - i);
  }
  // And multiply them together
  matmul(out, a, b, bits);

  if (inv != NULL) {
    // Invert upper and lower matrices, because that's really simple when
    // they're triangular.
    uint64_t ia[64], ib[64];
    for (int i = 0; i < bits; i++) {
      ia[i] = ib[i] = (uint64_t)1 << i;
    }
    for (int i = bits - 2; i >= 0; i--) {
      uint64_t m = a[i];
      uint64_t r = ia[i];
      for (int j = bits - 1; j > i; j--) {
        if ((m >> j) & 1) r ^= ia[j];
      }
      ia[i] = r;
    }
    for (int i = 1; i < bits; i++) {
      uint64_t m = b[i];
      uint64_t r = ib[i];
      for (int j = 0; j < i; j++) {
        if ((m >> j) & 1) r ^= ib[j];
      }
      ib[i] = r;
    }

    // And multiply them together for the inverse result.
    matmul(inv, ib, ia, bits);
  }
}


void init(state_t* state, uint64_t max) {
  // A table of murmur3-style mix function parameters.  These were discovered
  // by searching for values with good hashing properties:
  // https://sh1blog.blogspot.com/2016/08/n-bit-mixer-functions-eight-to-64-bits.html
  static const struct {
    uint8_t s[3];
    uint64_t m[2];
  } config[] = { /*{{{*/
    [  0] = { {  1,  1,  1 }, { 1, 1 } },
    [  1] = { {  1,  1,  1 }, { 1, 1 } },
    [  2] = { {  1,  1,  1 }, { 1, 1 } },
    [  3] = { {  1,  1,  1 }, { 1, 1 } },
    [  4] = { {  1,  1,  1 }, { 1, 1 } },
    [  5] = { {  1,  1,  1 }, { 1, 1 } },
    [  6] = { {  1,  1,  1 }, { 1, 1 } },
    [  7] = { {  1,  1,  1 }, { 1, 1 } },
    [  8] = { {  4,  3,  4 }, { 0x0000000000000000000000000000000b,0x00000000000000000000000000000013 } },
    [  9] = { {  7,  5,  5 }, { 0x0000000000000000000000000000002b,0x00000000000000000000000000000093 } },
    [ 10] = { {  4,  4,  5 }, { 0x00000000000000000000000000000007,0x000000000000000000000000000002b5 } },
    [ 11] = { {  6,  5,  5 }, { 0x000000000000000000000000000005ab,0x00000000000000000000000000000d35 } },
    [ 12] = { {  7,  5,  7 }, { 0x00000000000000000000000000000347,0x0000000000000000000000000000052d } },
    [ 13] = { {  6,  7,  7 }, { 0x000000000000000000000000000018ab,0x00000000000000000000000000000a53 } },
    [ 14] = { {  8,  8,  8 }, { 0x000000000000000000000000000068ab,0x0000000000000000000000000000594b } },
    [ 15] = { {  7,  7,  8 }, { 0x00000000000000000000000000001bab,0x00000000000000000000000000004b53 } },
    [ 16] = { {  8,  7,  9 }, { 0x0000000000000000000000000000994b,0x000000000000000000000000000044d3 } },
    [ 17] = { {  8,  8, 10 }, { 0x0000000000000000000000000000a15b,0x0000000000000000000000000001a653 } },
    [ 18] = { {  9,  8, 10 }, { 0x0000000000000000000000000002b755,0x00000000000000000000000000012653 } },
    [ 19] = { {  9,  9, 11 }, { 0x00000000000000000000000000048933,0x0000000000000000000000000005b2d3 } },
    [ 20] = { { 10, 10, 10 }, { 0x0000000000000000000000000007c14b,0x000000000000000000000000000ba653 } },
    [ 21] = { { 11, 10, 10 }, { 0x0000000000000000000000000007814b,0x000000000000000000000000000ba653 } },
    [ 22] = { { 12, 10, 12 }, { 0x0000000000000000000000000003814b,0x000000000000000000000000002ba653 } },
    [ 23] = { { 12, 10, 12 }, { 0x0000000000000000000000000063ed4b,0x000000000000000000000000000ba653 } },
    [ 24] = { { 12, 10, 12 }, { 0x0000000000000000000000000046e963,0x000000000000000000000000006da653 } },
    [ 25] = { { 12, 12, 14 }, { 0x0000000000000000000000000140e96b,0x000000000000000000000000010da6d3 } },
    [ 26] = { { 13, 12, 14 }, { 0x0000000000000000000000000340e96b,0x000000000000000000000000010da4d3 } },
    [ 27] = { { 14, 12, 14 }, { 0x0000000000000000000000000840e96b,0x0000000000000000000000000149a653 } },
    [ 28] = { { 14, 12, 14 }, { 0x0000000000000000000000000b7829a9,0x00000000000000000000000008cad969 } },
    [ 29] = { { 14, 14, 14 }, { 0x000000000000000000000000016069ab,0x00000000000000000000000018cad969 } },
    [ 30] = { { 15, 14, 17 }, { 0x000000000000000000000000035069ab,0x00000000000000000000000018cad969 } },
    [ 31] = { { 18, 15, 16 }, { 0x000000000000000000000000204c2ca5,0x00000000000000000000000055c8ad1d } },
    [ 32] = { { 17, 13, 16 }, { 0x0000000000000000000000002256a58d,0x000000000000000000000000f3ea6b47 } },
    [ 33] = { { 17, 15, 17 }, { 0x0000000000000000000000003e10a9ad,0x0000000000000000000000019b3cb5b3 } },
    [ 34] = { { 18, 16, 19 }, { 0x0000000000000000000000023c65b4cd,0x000000000000000000000001a1ecb5a7 } },
    [ 35] = { { 19, 16, 18 }, { 0x000000000000000000000005e056b58d,0x00000000000000000000000091ae4b47 } },
    [ 36] = { { 19, 16, 20 }, { 0x00000000000000000000000be6d6a5a7,0x00000000000000000000000083ad6c67 } },
    [ 37] = { { 20, 18, 20 }, { 0x00000000000000000000001a7fd6a5a5,0x00000000000000000000000193b54e67 } },
    [ 38] = { { 19, 17, 20 }, { 0x000000000000000000000001777ea5a7,0x00000000000000000000003393b96c67 } },
    [ 39] = { { 19, 16, 19 }, { 0x00000000000000000000005a1fb6b1a7,0x0000000000000000000000109b354c6b } },
    [ 40] = { { 20, 17, 21 }, { 0x0000000000000000000000a90158b6a5,0x0000000000000000000000372fadb365 } },
    [ 41] = { { 21, 17, 21 }, { 0x00000000000000000000006e2f72aced,0x0000000000000000000001b48e6b8a2d } },
    [ 42] = { { 22, 18, 21 }, { 0x0000000000000000000003cd00c2a6b5,0x0000000000000000000000ab2094b165 } },
    [ 43] = { { 23, 21, 25 }, { 0x000000000000000000000395426aa6ad,0x0000000000000000000003e65cb946c5 } },
    [ 44] = { { 23, 19, 21 }, { 0x00000000000000000000052ec35ea48d,0x000000000000000000000089a3894c67 } },
    [ 45] = { { 22, 18, 23 }, { 0x0000000000000000000003a9426e35ad,0x000000000000000000000521d0e9ab71 } },
    [ 46] = { { 21, 20, 23 }, { 0x0000000000000000000025798b76e55b,0x0000000000000000000018dcae1b1a91 } },
    [ 47] = { { 25, 22, 23 }, { 0x00000000000000000000035dc8daad2d,0x000000000000000000001c8266bb64f3 } },
    [ 48] = { { 22, 21, 26 }, { 0x0000000000000000000043a812d4ed35,0x00000000000000000000a8f9d21b4457 } },
    [ 49] = { { 25, 24, 25 }, { 0x000000000000000000006aeb5bc6ad33,0x0000000000000000000029f195a9c4d5 } },
    [ 50] = { { 25, 22, 25 }, { 0x000000000000000000017350165a8ceb,0x00000000000000000000154a30da53cf } },
    [ 51] = { { 26, 22, 24 }, { 0x00000000000000000000c038a795a55b,0x000000000000000000048dbcfb291a87 } },
    [ 52] = { { 28, 22, 25 }, { 0x0000000000000000000c1b201deaadab,0x0000000000000000000011db051adadf } },
    [ 53] = { { 27, 24, 24 }, { 0x0000000000000000000d51d9086a5e6b,0x0000000000000000000e1baa371d131d } },
    [ 54] = { { 30, 22, 29 }, { 0x0000000000000000000099442a48b48b,0x000000000000000000220df7b4a5dad7 } },
    [ 55] = { { 28, 24, 26 }, { 0x0000000000000000006ad29a13c684e5,0x0000000000000000004e583917858539 } },
    [ 56] = { { 31, 25, 30 }, { 0x0000000000000000008cbd48536544a7,0x000000000000000000f0e9029a39197d } },
    [ 57] = { { 30, 25, 30 }, { 0x00000000000000000117016da974e395,0x000000000000000000fb5b9b563418dd } },
    [ 58] = { { 29, 29, 33 }, { 0x00000000000000000116e5cf9872c477,0x000000000000000000e2a28c5123359d } },
    [ 59] = { { 30, 21, 30 }, { 0x000000000000000007470543fd56dcb9,0x0000000000000000039a3759502b38c5 } },
    [ 60] = { { 31, 26, 28 }, { 0x00000000000000000d67eb4dd862849d,0x000000000000000008b2611528f3bd7b } },
    [ 61] = { { 30, 24, 29 }, { 0x00000000000000000db38b0d9843951d,0x00000000000000000dfb620d2de29d33 } },
    [ 62] = { { 32, 30, 35 }, { 0x00000000000000002bfba303554aa5ad,0x00000000000000003b52aa1f019b9869 } },
    [ 63] = { { 30, 24, 32 }, { 0x00000000000000001e2699283f56170d,0x00000000000000007573d08f69f3795d } },
    [ 64] = { { 30, 26, 34 }, { 0x0000000000000000ee291b1b5f61cc4d,0x0000000000000000c2d00d8e4dfb2929 } },
//    [ 65] = { { 37, 33, 34 }, { 0x0000000000000000c1aee1a724694555,0x000000000000000007abfb7c6bf23d27 } },
//    [ 66] = { { 39, 32, 34 }, { 0x0000000000000000fc63bae7675b3455,0x0000000000000000d86ed3dcbd2f3e5d } },
//    [ 67] = { { 39, 33, 36 }, { 0x0000000000000006c24c6bc44278b29b,0x0000000000000006e9fed344898e0c87 } },
//    [ 68] = { { 34, 36, 35 }, { 0x000000000000000df39187ac92331377,0x000000000000000c91e95946c1a3675d } },
//    [ 69] = { { 39, 38, 37 }, { 0x000000000000000d4281b76e33c2bab3,0x00000000000000047583c3c6a2d36ff9 } },
//    [ 70] = { { 40, 36, 38 }, { 0x000000000000003c3c52022eb5438e9d,0x0000000000000002f34fe60b75b31645 } },
//    [ 71] = { { 37, 39, 37 }, { 0x000000000000000362331d48ccd6b011,0x0000000000000049f3dd354b46e8c45b } },
//    [ 72] = { { 35, 41, 36 }, { 0x00000000000000aef9c1f7e6b5d25591,0x0000000000000084104c8dc5da669f45 } },
//    [ 73] = { { 35, 41, 36 }, { 0x00000000000001aef9c1f7e6b5d25591,0x0000000000000084104c8dc5da669f45 } },
//    [ 74] = { { 39, 36, 32 }, { 0x00000000000001d83e407ab6899d7bb9,0x00000000000000b01545a8f5e8593e65 } },
//    [ 75] = { { 42, 41, 41 }, { 0x0000000000000538a38f3ee4d9accebb,0x0000000000000536318ce6a46cfab793 } },
//    [ 76] = { { 40, 33, 39 }, { 0x0000000000000f182a7f0ee8e296e49b,0x000000000000068052fff8fce35a54b5 } },
//    [ 77] = { { 41, 38, 36 }, { 0x000000000000032c08af1628a7f1f691,0x00000000000010b193ca9d37f4582da5 } },
//    [ 78] = { { 34, 34, 39 }, { 0x000000000000253490bfa022798552bb,0x0000000000003720deee842dff19b9e9 } },
//    [ 79] = { { 35, 36, 41 }, { 0x00000000000050210b852f76ab286715,0x00000000000049c3f31929ab460f12c5 } },
//    [ 80] = { { 53, 43, 45 }, { 0x00000000000066929b82016ef349661b,0x000000000000bc460978539a0680f6d3 } },
//    [ 81] = { { 40, 42, 31 }, { 0x00000000000025634f00a9b36c61267f,0x0000000000009ee3c49bf65322a759a9 } },
//    [ 82] = { { 41, 35, 46 }, { 0x0000000000013d3f9926b52b492364bb,0x0000000000024847a8b20406152986c3 } },
//    [ 83] = { { 40, 40, 37 }, { 0x0000000000057db66b647e7e0142934d,0x00000000000311e19869a722dda0e0d5 } },
//    [ 84] = { { 42, 42, 37 }, { 0x0000000000091592247ec8d51011697b,0x000000000003e9b9675ac85290355681 } },
//    [ 85] = { { 45, 48, 41 }, { 0x00000000000f0e79c3fa954adbaef427,0x00000000000d1af528e8e5cb3fe1e6a5 } },
//    [ 86] = { { 42, 49, 35 }, { 0x00000000000cbd821d6d1c9f5daa8ba9,0x0000000000217a4544a2ea5974e934f7 } },
//    [ 87] = { { 44, 42, 40 }, { 0x000000000007964b9a9c1ac665f4cb6b,0x00000000003b20fbe810a8cc0866dbab } },
//    [ 88] = { { 51, 44, 44 }, { 0x0000000000cbaf12db5a4aeda8c25cbf,0x0000000000237272f0388c2c45e9eaf7 } },
//    [ 89] = { { 50, 43, 46 }, { 0x0000000001a9874c69481ced67649e29,0x00000000006fa7103538f9c5850be999 } },
//    [ 90] = { { 51, 44, 44 }, { 0x000000000270b71aeafa42e62264aafb,0x0000000000698694b972cc4887abe3e9 } },
//    [ 91] = { { 43, 46, 61 }, { 0x0000000003c0a231da7fb9b366a7cb8f,0x000000000359aafd28c56cc02126a563 } },
//    [ 92] = { { 53, 44, 43 }, { 0x0000000009fd8c0e1c0c1a4d65546c25,0x00000000060a237594f2302c81315a85 } },
//    [ 93] = { { 43, 46, 62 }, { 0x0000000013c1b404886f397b22538f4f,0x00000000154ba99e215b8868a328d443 } },
//    [ 94] = { { 49, 54, 46 }, { 0x000000003d8b8cedfcb376f515c8d5d7,0x000000001e96b89aa3ab6a690419d1d5 } },
//    [ 95] = { { 45, 51, 42 }, { 0x000000005fc70c5c34f4cceac26efc6f,0x000000002d9711fcc67be973eaedfa85 } },
//    [ 96] = { { 41, 50, 56 }, { 0x000000005f9a077b0e30784b6f5585a9,0x00000000070a841004bb30fc18d90dcd } },
//    [ 97] = { { 51, 52, 46 }, { 0x00000000b3ab046ea0362e6647e16e57,0x000000015eb44a5fcaa058fee709c657 } },
//    [ 98] = { { 58, 55, 36 }, { 0x00000003ebab06243ba8397f9ded8cc3,0x000000021b85517f0d277fcf595e4b25 } },
//    [ 99] = { { 52, 59, 54 }, { 0x00000001ece3976ae9333e3dcba1094f,0x0000000275ad977787a576fc4d7b688b } },
//    [100] = { { 64, 59, 45 }, { 0x000000050b482142aa60920ed1729371,0x00000002a3d491e0a48ee1f46bfcd12f } },
//    [101] = { { 41, 59, 34 }, { 0x0000000aa9ee406c4c80736dd6917c91,0x0000000d10ae3c90dea64633b73aa52f } },
//    [102] = { { 55, 60, 63 }, { 0x00000003b94fa386fc14cb2bcea9d111,0x0000000703c5fc63f96dca3ed629360d } },
//    [103] = { { 57, 61, 33 }, { 0x000000265de894e5396f75eec35597b9,0x0000005eefc16a24b91e1a5fdfa64dad } },
//    [104] = { { 48, 62, 51 }, { 0x0000009e22783b07851878fac33dd281,0x0000004e92a86b98b352de06a6594099 } },
//    [105] = { { 56, 45, 64 }, { 0x0000019073e5fc45bf5ee53ae5cf9217,0x0000002d77d0ba7cfb2b994b8675252f } },
//    [106] = { { 57, 70, 29 }, { 0x0000029e318585576aac4eaa3ab63911,0x000000377bac3250144c2b72d2ef3b9b } },
//    [107] = { { 58, 63, 47 }, { 0x00000090dc65314a8f107d584cfbb0d1,0x000005e2d47a86a57132d2b0a5a0fdf3 } },
//    [108] = { { 46, 68, 45 }, { 0x0000099296e66b9c45e0a05b6d16b18f,0x000009f0d9d0e11e0d48708d5f58f28f } },
//    [109] = { { 47, 65, 65 }, { 0x000015e0969f321d84e86494d414f863,0x000006044d48a389cea068db5108fda9 } },
//    [110] = { { 41, 69, 46 }, { 0x000008c73e6449674e589d9045c96d15,0x0000191cd47486b4e9d77d33ec50dfe7 } },
//    [111] = { { 58, 71, 44 }, { 0x00002102d49d19626d044d5c1d9ae103,0x00002938f920e6944efa5caef7878e63 } },
//    [112] = { { 69, 69, 66 }, { 0x0000d071ea7d70b922652c9f4efda8cb,0x0000e52504c30c4096fe91d069e46121 } },
//    [113] = { { 51, 72, 54 }, { 0x000193990ced7b35ba733f67199e35bd,0x0000b5d19749518fb15cbdafd9a7d465 } },
//    [114] = { { 41, 64, 56 }, { 0x0001dfa95a9c990692a392fe97a4f315,0x0001d560652911c558586eed3231d675 } },
//    [115] = { { 59, 61, 51 }, { 0x0005daeb94781e99eaead2570c08bd71,0x000765307428ff9d13285a852adad497 } },
//    [116] = { { 67, 63, 52 }, { 0x00081e9e8474ae769818815f2bcd98c7,0x000b01661ba448ef05e66cbbc9207431 } },
//    [117] = { { 53, 61, 59 }, { 0x001389e30cd07e0cd3c99dec23f7b341,0x001518d799822036676494a13b405a63 } },
//    [118] = { { 63, 46, 55 }, { 0x00138e781c31e618cf8d5a8689e93e3d,0x003da8a10e2da973b6de54b9e569f85b } },
//    [119] = { { 55, 58, 46 }, { 0x005060d63017eff64942abeecad0faef,0x0059de7110354cb25de9e440aae1b7bd } },
//    [120] = { { 45, 44, 55 }, { 0x00d18e88d849b5d62bc57a82c0c5ad37,0x00adb0989a25e823b306d0eb61661977 } },
//    [121] = { { 64, 70, 85 }, { 0x00c5c3462bdc06418fc661c45b6abe1b,0x002878d6a58fa193fa98242623f1537d } },
//    [122] = { { 55, 70, 85 }, { 0x03f58fd2254a5c438ba0324e7372ba31,0x000f315ee584e91379fae40e2938472d } },
//    [123] = { { 54, 56, 85 }, { 0x043054cbbcf482ffc6a20ccf3a430715,0x00dcacdfa4fd86e90bdef329c674a3d5 } },
//    [124] = { { 61, 67, 82 }, { 0x0d0139acecf086eaa2345a21d0d20b87,0x0d698c3802ed56e01774322836b610b5 } },
//    [125] = { { 63, 68, 52 }, { 0x0e20947e484fe27ec73bbfb2ff2229d1,0x104e5b22226f7df8250de82e703004dd } },
//    [126] = { { 62, 67, 54 }, { 0x2402f676b82cd3d745a36180dac22a5b,0x1f7d66480e35477c37e8a5c215e073ef } },
//    [127] = { { 62, 68, 55 }, { 0x6909aa5a669e5ab525289c15a2942f97,0x6e40aacb80adb3c037e0a73531f2d0c3 } },
//    [128] = { { 59, 60, 84 }, { 0xecfb1b9bc1f0564fc68dd22b9302d18d,0x4a4cf0348b717188e2aead7d60f8a0df } },
  }; /*}}}*/
  int bits;
  for (bits = 1; (max >> (bits - 1)) > 1; bits++) continue;

  for (int i = 0; i < NSTAGES; i++) {
    invertiblematrix(state->stage[i].mat, state->istage[i].mat, bits);
    state->stage[i].mult = rand64() | 1;
    state->istage[i].mult = multinv(state->stage[i].mult);
  }
  state->z = 0;
  state->max = max;
  state->m0 = config[bits].m[0];
  state->m1 = config[bits].m[1];
  state->im0 = multinv(state->m0);
  state->im1 = multinv(state->m1);
  state->s0 = config[bits].s[0];
  state->s1 = config[bits].s[1];
  state->s2 = config[bits].s[2];
  state->bits = bits;
}

// Return values `0 <= x < state->max` in random order without repetition over
// state->max values.
uint64_t next(state_t* state) {
  uint64_t z = state->z;
  uint64_t m = ((uint64_t)2 << (state->bits - 1)) - 1;
  state->z = state->z < state->max ? state->z + 1 : 0;
  do {
    for (int i = 0; i < NSTAGES; i++) {
      uint64_t zz = 0;
      // One step of a Weyl generator
      // (constant is arbitrary, but happens to be phi)
      z += 0x9e3779b97f4a7c15;

      // Matrix by vector multiply
      for (int j = 0; j < state->bits; j++) {
        if (z & 1) zz ^= state->stage[i].mat[j];
        z >>= 1;
      }

      // One step of a Lehmer generator
      // (multiplicand must be odd to be co-prime to modulo)
      z = zz * state->stage[i].mult;

      // And crop back to the necessary number of bits
      z &= m;
    }

    // murmur3-style mix function
    z ^= z >> state->s0;
    z *= state->m0;
    z &= m;
    z ^= z >> state->s1;
    z *= state->m1;
    z &= m;
    z ^= z >> state->s2;

    // Now the result may be out-of-range, in which case we use it as the input
    // for another round of the same thing.  Another approach would be to use
    // the next state->z value, but this way is easier to reverse and the
    // bijective property ensures that it can't get stuck in an out-of-range
    // cycle (we started in-range, so we can't be on a cycle that is entirely
    // out-of-range, and we can't fall into such a cycle using bijective
    // operations).
  } while (z > state->max);
  return z;
}


// Figure out what position in the squence a result from next() was.
uint64_t undo(state_t *state, uint64_t x) {
  uint64_t m = ((uint64_t)2 << (state->bits - 1)) - 1;
  do {
    x ^= x >> state->s2 ^ (x >> state->s2 >> state->s2);
    x *= state->im1;
    x &= m;
    x ^= x >> state->s1 ^ (x >> state->s1 >> state->s1);
    x *= state->im0;
    x &= m;
    x ^= x >> state->s0 ^ (x >> state->s0 >> state->s0);
    for (int i = NSTAGES - 1; i >= 0; i--) {
      uint64_t xx = 0;
      x *= state->istage[i].mult;
      x &= m;
      for (int j = 0; j < state->bits; j++) {
        if (x & 1) xx ^= state->istage[i].mat[j];
        x >>= 1;
      }
      x = xx - 0x9e3779b97f4a7c15;
      x &= m;
    }
  } while (x > state->max);
  return x;
}


int main(int argc, char *argv[]) {
  state_t state;
  uint64_t max = ~(uint64_t)0;
  uint32_t seed = time(NULL);

  if (argc >= 2) max = strtoull(argv[1], NULL, 0);
  if (argc >= 3) seed = strtoull(argv[2], NULL, 0);
  srand(seed);
  init(&state, max);
  if (argc <= 3 && strcmp(argv[3], "-b") == 0) {
    for (;;) {
      uint64_t r = next(&state);
      fwrite(&r, 1, state.bits >> 3, stdout);
    }
  } else {
    for (int i = 0; i < 20; i++) {
      uint64_t n = next(&state);
      uint64_t m = undo(&state, n);
      printf("%016" PRIx64 "  %016" PRIx64 "\n", n, m);
    }
  }
  return 0;
}

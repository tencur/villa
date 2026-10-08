// Sign-aware trilinear blend of Lasagna's hemisphere-encoded (nx, ny) normal codes
// for the sparse chunk-cache kernels.
//
// The predict3d writer stores (nx, ny) * sign(nz), so neighbouring voxels of one
// sheet can hold codes of opposite sign where the normal is close to the z = 0
// plane; blending those codes linearly cancels the in-plane part.  These helpers
// mirror lasagna/normal_sampling.py step by step (see its module docstring) and
// must be kept in sync with it:
//   1. read the 8 corner codes of both channels,
//   2. decode each corner to (nx, ny, nz = sqrt(max(0, 1 - nx^2 - ny^2))),
//   3. reference = corner with the largest trilinear weight (first on ties);
//      a corner whose dot product with the reference is negative is flipped
//      (a corner with code (0, 0), i.e. zero padding, is never the reference
//      and never flipped),
//   4. blend with the ordinary weights, using the mirrored code 256 - c for flipped
//      corners; if the blended nz is negative, mirror the result back to nz >= 0.
// When all corners agree, steps 3-4 are the identity and the blend is the same
// arithmetic, in the same order, as the plain per-channel kernel loop.
//
// Corner order everywhere: k = dx + 2*dy + 4*dz = 000,100,010,110,001,101,011,111.
#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Read the 8 corner codes of one channel of a padded 34^3 chunk.
__device__ __forceinline__ void lasagna_normal_pair_load_corners(
    const uint8_t* __restrict__ ch,
    int ix0, int ix1, int iy0, int iy1, int iz0, int iz1,
    float* c)
{
    const int S_z = 34 * 34;
    const int S_y = 34;
    c[0] = (float)ch[iz0 * S_z + iy0 * S_y + ix0];
    c[1] = (float)ch[iz0 * S_z + iy0 * S_y + ix1];
    c[2] = (float)ch[iz0 * S_z + iy1 * S_y + ix0];
    c[3] = (float)ch[iz0 * S_z + iy1 * S_y + ix1];
    c[4] = (float)ch[iz1 * S_z + iy0 * S_y + ix0];
    c[5] = (float)ch[iz1 * S_z + iy0 * S_y + ix1];
    c[6] = (float)ch[iz1 * S_z + iy1 * S_y + ix0];
    c[7] = (float)ch[iz1 * S_z + iy1 * S_y + ix1];
}

// Trilinear weights in corner order (same expressions as the kernels' w000..w111).
__device__ __forceinline__ void lasagna_trilinear_weights(float fx, float fy, float fz, float* w)
{
    w[0] = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    w[1] = fx          * (1.0f - fy) * (1.0f - fz);
    w[2] = (1.0f - fx) * fy          * (1.0f - fz);
    w[3] = fx          * fy          * (1.0f - fz);
    w[4] = (1.0f - fx) * (1.0f - fy) * fz;
    w[5] = fx          * (1.0f - fy) * fz;
    w[6] = (1.0f - fx) * fy          * fz;
    w[7] = fx          * fy          * fz;
}

// Per-corner sign (+1 keep / -1 flip) relative to the heaviest corner, and the
// final hemisphere flip (+1 / -1) applied to the blended codes.
// Mirrors normal_sampling.align_corner_signs.
__device__ __forceinline__ void lasagna_normal_pair_signs(
    const float* ca,   // [8] corner codes of nx
    const float* cb,   // [8] corner codes of ny
    const float* w,    // [8] trilinear weights
    float* sgn,        // [8] out
    float* flip)       // out
{
    // code (0, 0) = zero padding / no data (no valid normal encodes to it):
    // never the reference, never flipped.
    bool empty[8];
    for (int k = 0; k < 8; k++) empty[k] = (ca[k] == 0.0f) && (cb[k] == 0.0f);
    // Reference: heaviest non-empty corner, first maximum on ties.
    int ref = -1;
    for (int k = 0; k < 8; k++) {
        if (!empty[k] && (ref < 0 || w[k] > w[ref])) ref = k;
    }
    if (ref < 0) ref = 0;
    float ax[8], ay[8], az[8];
    for (int k = 0; k < 8; k++) {
        ax[k] = (ca[k] - 128.0f) / 127.0f;
        ay[k] = (cb[k] - 128.0f) / 127.0f;
        az[k] = sqrtf(fmaxf(0.0f, 1.0f - ax[k] * ax[k] - ay[k] * ay[k]));
    }
    float sz = 0.0f;
    for (int k = 0; k < 8; k++) {
        float dot = ax[k] * ax[ref] + ay[k] * ay[ref] + az[k] * az[ref];
        sgn[k] = (empty[k] || dot >= 0.0f) ? 1.0f : -1.0f;
        sz += w[k] * sgn[k] * az[k];
    }
    *flip = (sz < 0.0f) ? -1.0f : 1.0f;
}

// Mirrored codes: c'[k] = c[k] for kept corners, 256 - c[k] for flipped ones
// ((256 - c - 128) / 127 == -(c - 128) / 127).
__device__ __forceinline__ void lasagna_normal_pair_mirror(const float* c, const float* sgn, float* out)
{
    for (int k = 0; k < 8; k++) {
        out[k] = (sgn[k] > 0.0f) ? c[k] : 256.0f - c[k];
    }
}

// Blended code of one channel.  Mirrors normal_sampling.blend_codes.
__device__ __forceinline__ float lasagna_normal_pair_blend(
    const float* c, const float* w, const float* sgn, float flip)
{
    float val = 0.0f;
    for (int k = 0; k < 8; k++) {
        val += w[k] * ((sgn[k] > 0.0f) ? c[k] : 256.0f - c[k]);
    }
    return (flip < 0.0f) ? 256.0f - val : val;
}

// d(blend)/d(fx, fy, fz) of a plain trilinear blend of v[8]; same formulas as the
// per-channel backward loop.  For the normal pair, call with the mirrored codes and
// multiply the result by flip (signs and flip are piecewise constant in position).
__device__ __forceinline__ void lasagna_trilinear_grad(
    const float* v, float fx, float fy, float fz,
    float* dfx, float* dfy, float* dfz)
{
    *dfx = (1.0f - fy) * (1.0f - fz) * (v[1] - v[0])
         + fy          * (1.0f - fz) * (v[3] - v[2])
         + (1.0f - fy) * fz          * (v[5] - v[4])
         + fy          * fz          * (v[7] - v[6]);
    *dfy = (1.0f - fx) * (1.0f - fz) * (v[2] - v[0])
         + fx          * (1.0f - fz) * (v[3] - v[1])
         + (1.0f - fx) * fz          * (v[6] - v[4])
         + fx          * fz          * (v[7] - v[5]);
    *dfz = (1.0f - fx) * (1.0f - fy) * (v[4] - v[0])
         + fx          * (1.0f - fy) * (v[5] - v[1])
         + (1.0f - fx) * fy          * (v[6] - v[2])
         + fx          * fy          * (v[7] - v[3]);
}

__device__ __forceinline__ uint8_t lasagna_round_u8(float val)
{
    int ival = (int)roundf(val);
    if (ival < 0) ival = 0;
    if (ival > 255) ival = 255;
    return (uint8_t)ival;
}

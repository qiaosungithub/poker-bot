// Row-wise CUDA kernels for the Deep CFR fit step (one warp per row, lane owns 8 columns).
// GEMMs are done by cuBLAS (torch.mm / addmm) in fast_step.py; these kernels do the
// per-row work: gather/targets, LayerNorm fwd/bwd, output layer + loss + its backward,
// clip-norm and Adam. Small-parameter gradients (biases, LN params, output layer) are
// reduced per block (registers -> shared memory) and stored WITHOUT atomics into row
// blockIdx of a scratch matrix S [NB, n_small]; norm_kernel sums the rows.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void load8(const bf16* p, float* f) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) { float2 t = __bfloat1622float2(h[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
__device__ __forceinline__ void store8(bf16* p, const float* f) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void loadf8(const float* p, float* f) {
  float4 a = *reinterpret_cast<const float4*>(p);
  float4 b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// ---------------------------------------------------------------------------------------
// gather rows, build bf16 inputs, normalised targets, sample weights
// aux[row*16 + j] = legal mask (j<5), aux[row*16 + 8 + j] = target, wrow[row] = weight
__global__ void prep_kernel(const float* __restrict__ data, const int64_t* __restrict__ idx,
                            const float* __restrict__ itmax_p, bf16* __restrict__ xb,
                            float* __restrict__ aux, float* __restrict__ wrow,
                            float* __restrict__ ctrl, int B, int strat) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (blockIdx.x == 0 && threadIdx.x == 0) { ctrl[0] = 0.f; ctrl[1] += 1.f; }
  if (row >= B) return;
  const float* r = data + idx[row] * 64;
  float2 v = *reinterpret_cast<const float2*>(r + 2 * lane);
  int c = 2 * lane;
  float x0 = c < 51 ? v.x : 0.f, x1 = c + 1 < 51 ? v.y : 0.f;
  reinterpret_cast<__nv_bfloat162*>(xb + (int64_t)row * 64)[lane] = __floats2bfloat162_rn(x0, x1);
  float mk = 0.f, y = 0.f;
  if (lane < 5) { mk = r[46 + lane]; y = r[51 + lane]; }
  float itmax = fmaxf(*itmax_p, 1.f);
  float q = r[56] / itmax;
  float w = q * sqrtf(q);
  float yn;
  if (strat) {
    w *= r[57];
    float yy = y * mk;
    float s = warp_sum(yy);
    yn = yy / fmaxf(s, 1e-12f);
  } else {
    float n = fmaxf(warp_sum(mk), 1.f);
    float ym = y * mk;
    float mean = warp_sum(ym) / n;
    float d = ym - mean * mk;
    float var = warp_sum(d * d * mk) / n;
    yn = (y - mean) / sqrtf(var + 1e-6f) * mk;
  }
  if (lane < 8) { aux[row * 16 + lane] = mk; aux[row * 16 + 8 + lane] = yn; }
  if (lane == 0) wrow[row] = w;
}

// ---------------------------------------------------------------------------------------
// a = lrelu(LN(h) * g + be), st = (mu, rstd)
__global__ void ln_fwd_kernel(const bf16* __restrict__ h, const float* __restrict__ g,
                              const float* __restrict__ be, float* __restrict__ st,
                              bf16* __restrict__ a, int B) {
  int lane = threadIdx.x & 31;
  int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int nw = (gridDim.x * blockDim.x) >> 5;
  int c0 = lane * 8;
  float gg[8], bb[8];
  loadf8(g + c0, gg); loadf8(be + c0, bb);
  for (int row = gw; row < B; row += nw) {
    float f[8];
    load8(h + (int64_t)row * 256 + c0, f);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += f[i];
    float mu = warp_sum(s) * (1.f / 256.f);
    float s2 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { f[i] -= mu; s2 += f[i] * f[i]; }
    float r = 1.f / sqrtf(warp_sum(s2) * (1.f / 256.f) + 1e-5f);
#pragma unroll
    for (int i = 0; i < 8; ++i) { float z = f[i] * r * gg[i] + bb[i]; f[i] = z >= 0.f ? z : 0.01f * z; }
    store8(a + (int64_t)row * 256 + c0, f);
    if (lane == 0) { st[row * 2] = mu; st[row * 2 + 1] = r; }
  }
}

// ---------------------------------------------------------------------------------------
// LayerNorm + LeakyReLU backward of one row given da (grad wrt the lrelu output).
// Accumulates per-lane column partials: acc[0] += dh (bias), acc[1] += dz*xh (gain),
// acc[2] += dz (LN bias).  dz = lrelu'(z) * da, xh = normalised h.
__device__ __forceinline__ void ln_bwd_row(const float* da, const float* hf, float mu, float r,
                                           const float* gg, const float* bb, bf16* dhrow,
                                           float (&acc)[3][8]) {
  float xh[8], dz[8], dxh[8], s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    xh[i] = (hf[i] - mu) * r;
    float z = xh[i] * gg[i] + bb[i];
    dz[i] = z >= 0.f ? da[i] : 0.01f * da[i];
    dxh[i] = dz[i] * gg[i];
    s1 += dxh[i]; s2 += dxh[i] * xh[i];
    acc[1][i] += dz[i] * xh[i]; acc[2][i] += dz[i];
  }
  float m1 = warp_sum(s1) * (1.f / 256.f);
  float m2 = warp_sum(s2) * (1.f / 256.f);
  float o[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { o[i] = r * (dxh[i] - m1 - xh[i] * m2); acc[0][i] += o[i]; }
  store8(dhrow, o);
}

// per-warp partial vectors live in smw[warp * stride + j]; sum over warps, store the
// first `count` sums contiguously to dst[blockIdx.x * ns + j] (plain stores).
__device__ __forceinline__ void block_sum_store(const float* smw, int stride, int count,
                                                float* dst, int ns) {
  int nwarps = blockDim.x >> 5;
  float* out = dst + (int64_t)blockIdx.x * ns;
  for (int j = threadIdx.x; j < count; j += blockDim.x) {
    float s = 0.f;
    for (int w = 0; w < nwarps; ++w) s += smw[w * stride + j];
    out[j] = s;
  }
}

__device__ __forceinline__ void put8(float* p, const float* v) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// dst -> S[., b|g|be] (768 contiguous values)
template <int NR>
__global__ void ln_bwd_kernel(const bf16* __restrict__ da, const bf16* __restrict__ h,
                              const float* __restrict__ st, const float* __restrict__ g,
                              const float* __restrict__ be, bf16* __restrict__ dh,
                              float* __restrict__ dst, int ns, int B) {
  extern __shared__ float smw[];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int nw = (gridDim.x * blockDim.x) >> 5;
  int c0 = lane * 8;
  float gg[8], bb[8], acc[3][8] = {{0}};
  loadf8(g + c0, gg); loadf8(be + c0, bb);
  for (int row0 = gw * NR; row0 < B; row0 += nw * NR) {
    float d[NR][8], hf[NR][8], mu[NR], r[NR];
#pragma unroll
    for (int q = 0; q < NR; ++q) {
      int row = row0 + q;
      load8(da + (int64_t)row * 256 + c0, d[q]);
      load8(h + (int64_t)row * 256 + c0, hf[q]);
      mu[q] = st[row * 2]; r[q] = st[row * 2 + 1];
    }
#pragma unroll
    for (int q = 0; q < NR; ++q)
      ln_bwd_row(d[q], hf[q], mu[q], r[q], gg, bb, dh + (int64_t)(row0 + q) * 256 + c0, acc);
  }
#pragma unroll
  for (int v = 0; v < 3; ++v) put8(smw + warp * 768 + v * 256 + c0, acc[v]);
  __syncthreads();
  block_sum_store(smw, 768, 768, dst, ns);
}

// ---------------------------------------------------------------------------------------
// advantage head: a3 = lrelu(h3); v = a3.Wo[0] + bo[0]; A_j = a3.Wo[1+j] + bo[1+j];
// dueling combine, weighted MSE vs normalised targets, backward -> dh3.
// dst -> S[., b3 (256) | Wo (8x256) | bo (8)] = 2312 contiguous values; lossp[blockIdx].
#define HA_STRIDE 2320
__global__ void head_adv_kernel(const bf16* __restrict__ h3, const float* __restrict__ Wo,
                                const float* __restrict__ bo, const float* __restrict__ aux,
                                const float* __restrict__ wrow, bf16* __restrict__ dh3,
                                float* __restrict__ dst, float* __restrict__ lossp, int ns,
                                float inv5b, int B) {
  extern __shared__ float smw[];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int nw = (gridDim.x * blockDim.x) >> 5;
  int c0 = lane * 8;
  float wo[6][8];
#pragma unroll
  for (int k = 0; k < 6; ++k) loadf8(Wo + k * 256 + c0, wo[k]);
  float bo0 = bo[0];
  float boa[5];
#pragma unroll
  for (int j = 0; j < 5; ++j) boa[j] = bo[1 + j];
  float adb3[8] = {0}, awo[6][8] = {{0}}, abo[6] = {0}, aloss = 0.f;
  for (int row = gw; row < B; row += nw) {
    float a3[8], hraw[8];
    load8(h3 + (int64_t)row * 256 + c0, hraw);
#pragma unroll
    for (int i = 0; i < 8; ++i) a3[i] = hraw[i] >= 0.f ? hraw[i] : 0.01f * hraw[i];
    float o[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) s += a3[i] * wo[k][i];
      o[k] = s;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
      for (int k = 0; k < 6; ++k) o[k] += __shfl_xor_sync(0xffffffffu, o[k], off);
    const float* ar = aux + row * 16;
    float mk[5], yn[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) { mk[j] = ar[j]; yn[j] = ar[8 + j]; }
    float w = wrow[row];
    float v = o[0] + bo0, A[5], n = 0.f, asum = 0.f;
#pragma unroll
    for (int j = 0; j < 5; ++j) { A[j] = (o[1 + j] + boa[j]) * mk[j]; n += mk[j]; asum += A[j]; }
    n = fmaxf(n, 1.f);
    float amean = asum / n, l = 0.f, gj[5], dv = 0.f, gm = 0.f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      float pred = (v + A[j] - amean * mk[j]) * mk[j];
      float diff = pred - yn[j];
      l += diff * diff;
      gj[j] = diff * (2.f * inv5b) * w * mk[j];
      dv += gj[j]; gm += gj[j] * mk[j];
    }
    gm /= n;
    float dA[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) dA[j] = (gj[j] - gm) * mk[j];
    aloss += l * w;
    float d[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float dt = dv * wo[0][i];
#pragma unroll
      for (int j = 0; j < 5; ++j) dt += dA[j] * wo[1 + j][i];
      d[i] = hraw[i] >= 0.f ? dt : 0.01f * dt;
      adb3[i] += d[i];
      awo[0][i] += dv * a3[i];
#pragma unroll
      for (int j = 0; j < 5; ++j) awo[1 + j][i] += dA[j] * a3[i];
    }
    store8(dh3 + (int64_t)row * 256 + c0, d);
    abo[0] += dv;
#pragma unroll
    for (int j = 0; j < 5; ++j) abo[1 + j] += dA[j];
  }
  float* my = smw + warp * HA_STRIDE;
  put8(my + c0, adb3);
#pragma unroll
  for (int k = 0; k < 6; ++k) put8(my + 256 + k * 256 + c0, awo[k]);
  float z8[8] = {0};
  put8(my + 256 + 6 * 256 + c0, z8);
  put8(my + 256 + 7 * 256 + c0, z8);
  if (lane < 8) my[2304 + lane] = lane < 6 ? abo[lane] : 0.f;
  if (lane == 0) my[2312] = aloss;
  __syncthreads();
  block_sum_store(smw, HA_STRIDE, 2312, dst, ns);
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += smw[w * HA_STRIDE + 2312];
    lossp[blockIdx.x] = s * inv5b;
  }
}

// ---------------------------------------------------------------------------------------
// strategy head: logits_j = a2.Wo[j] + bo[j] (masked), weighted CE, backward -> da2,
// then LN2 backward -> dh2.  dst1 -> S[., b2|g2|be2] (768), dst2 -> S[., Wo (2048) | bo (8)].
#define HS_STRIDE 2832
__global__ void head_strat_kernel(const bf16* __restrict__ a2, const float* __restrict__ Wo,
                                  const float* __restrict__ bo, const float* __restrict__ aux,
                                  const float* __restrict__ wrow, const bf16* __restrict__ h2,
                                  const float* __restrict__ st2, const float* __restrict__ g2,
                                  const float* __restrict__ be2, bf16* __restrict__ dh2,
                                  float* __restrict__ dst1, float* __restrict__ dst2,
                                  float* __restrict__ lossp, int ns, float invb, int B) {
  extern __shared__ float smw[];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int nw = (gridDim.x * blockDim.x) >> 5;
  int c0 = lane * 8;
  float wo[5][8], gg[8], bb[8];
#pragma unroll
  for (int k = 0; k < 5; ++k) loadf8(Wo + k * 256 + c0, wo[k]);
  loadf8(g2 + c0, gg); loadf8(be2 + c0, bb);
  float bov[5];
#pragma unroll
  for (int j = 0; j < 5; ++j) bov[j] = bo[j];
  float awo[5][8] = {{0}}, abo[5] = {0}, acc[3][8] = {{0}}, aloss = 0.f;
  for (int row = gw; row < B; row += nw) {
    float x[8], hf[8];
    load8(a2 + (int64_t)row * 256 + c0, x);
    load8(h2 + (int64_t)row * 256 + c0, hf);
    float mu = st2[row * 2], r = st2[row * 2 + 1];
    const float* ar = aux + row * 16;
    float mk[5], yn[5], lg[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) s += x[i] * wo[j][i];
      lg[j] = s;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
      for (int j = 0; j < 5; ++j) lg[j] += __shfl_xor_sync(0xffffffffu, lg[j], off);
    float mx = -3.0e38f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      mk[j] = ar[j]; yn[j] = ar[8 + j];
      lg[j] = mk[j] > 0.f ? lg[j] + bov[j] : -1e9f;
      mx = fmaxf(mx, lg[j]);
    }
    float se = 0.f, ys = 0.f, e[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) { e[j] = __expf(lg[j] - mx); se += e[j]; ys += yn[j]; }
    float lse = logf(se), w = wrow[row], l = 0.f, dl[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      l -= yn[j] * (lg[j] - mx - lse);
      dl[j] = mk[j] > 0.f ? (e[j] / se * ys - yn[j]) * w * invb : 0.f;
      abo[j] += dl[j];
    }
    aloss += l * w;
    float da[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 5; ++j) { s += dl[j] * wo[j][i]; awo[j][i] += dl[j] * x[i]; }
      da[i] = s;
    }
    ln_bwd_row(da, hf, mu, r, gg, bb, dh2 + (int64_t)row * 256 + c0, acc);
  }
  float* my = smw + warp * HS_STRIDE;
#pragma unroll
  for (int v = 0; v < 3; ++v) put8(my + v * 256 + c0, acc[v]);
#pragma unroll
  for (int k = 0; k < 5; ++k) put8(my + 768 + k * 256 + c0, awo[k]);
  float z8[8] = {0};
  for (int k = 5; k < 8; ++k) put8(my + 768 + k * 256 + c0, z8);
  if (lane < 8) my[768 + 2048 + lane] = lane < 5 ? abo[lane] : 0.f;
  if (lane == 0) my[768 + 2056] = aloss;
  __syncthreads();
  block_sum_store(smw, HS_STRIDE, 768, dst1, ns);
  block_sum_store(smw + 768, HS_STRIDE, 2056, dst2, ns);
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += smw[w * HS_STRIDE + 768 + 2056];
    lossp[blockIdx.x] = s * invb;
  }
}

// ---------------------------------------------------------------------------------------
// Blocks [0, nbig): elementwise over the big region (4 elements / thread).
// Blocks [nbig, nbig + ceil(ns/32)): column sums of S [NB, ns] for 32 small columns each,
// written to G[small_off + col]. Last block also adds sum(lossp) into loss_acc.
// Every block adds its ||G*mask||^2 partial into ctrl[0].
__global__ void norm_kernel(float* __restrict__ G, const float* __restrict__ S,
                            const float* __restrict__ mask, float* __restrict__ ctrl,
                            const float* __restrict__ lossp, float* __restrict__ loss_acc,
                            int small_off, int ns, int NB, int nbig) {
  __shared__ float red[8][33];
  float contrib = 0.f;
  int tid = threadIdx.x;
  if ((int)blockIdx.x < nbig) {
    int i0 = (blockIdx.x * 256 + tid) * 4;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      int i = i0 + q;
      if (i < small_off) { float g = G[i] * mask[i]; contrib += g * g; }
    }
  } else {
    int sb = blockIdx.x - nbig;
    int col = sb * 32 + (tid & 31);
    int rg = tid >> 5;
    float s = 0.f;
    if (col < ns)
      for (int r = rg; r < NB; r += 8) s += S[(int64_t)r * ns + col];
    red[rg][tid & 31] = s;
    __syncthreads();
    if (rg == 0 && col < ns) {
      float t = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) t += red[q][tid];
      G[small_off + col] = t;
      float gm = t * mask[small_off + col];
      contrib = gm * gm;
    }
    if (blockIdx.x == gridDim.x - 1 && rg == 1) {
      float l = 0.f;
      for (int r = tid & 31; r < NB; r += 32) l += lossp[r];
      l = warp_sum(l);
      if ((tid & 31) == 0) *loss_acc += l;
    }
  }
  __syncthreads();
  float v = warp_sum(contrib);
  __shared__ float wsum[8];
  if ((tid & 31) == 0) wsum[tid >> 5] = v;
  __syncthreads();
  if (tid < 32) {
    v = tid < 8 ? wsum[tid] : 0.f;
    v = warp_sum(v);
    if (tid == 0) atomicAdd(ctrl, v);
  }
}

// clip_grad_norm_(1.0) + torch Adam step; writes fp32 master + bf16 copy; G := 0
__global__ void adam_kernel(float* __restrict__ P, bf16* __restrict__ Pb, float* __restrict__ G,
                            float* __restrict__ m, float* __restrict__ v,
                            const float* __restrict__ mask, const float* __restrict__ ctrl,
                            const float* __restrict__ lr_p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float norm = sqrtf(ctrl[0]);
  float t = ctrl[1];
  float lr = *lr_p;
  float coef = fminf(1.f / (norm + 1e-6f), 1.f);
  float g = G[i] * mask[i] * coef;
  float mi = 0.9f * m[i] + 0.1f * g;
  float vi = 0.999f * v[i] + 0.001f * g * g;
  float bc1 = 1.f - powf(0.9f, t);
  float bc2 = 1.f - powf(0.999f, t);
  float denom = sqrtf(vi) / sqrtf(bc2) + 1e-8f;
  float p = P[i] - (lr / bc1) * mi / denom;
  P[i] = p; Pb[i] = __float2bfloat16_rn(p); m[i] = mi; v[i] = vi; G[i] = 0.f;
}

// ---------------------------------------------------------------------------------------
#define F(t) (t).data_ptr<float>()
#define H(t) reinterpret_cast<bf16*>((t).data_ptr<at::BFloat16>())
#define STREAM at::cuda::getCurrentCUDAStream()

void fk_prep(torch::Tensor data, torch::Tensor idx, torch::Tensor itmax, torch::Tensor xb,
             torch::Tensor aux, torch::Tensor wrow, torch::Tensor ctrl, int64_t strat) {
  int B = idx.size(0);
  prep_kernel<<<(B * 32 + 255) / 256, 256, 0, STREAM>>>(F(data), idx.data_ptr<int64_t>(),
      F(itmax), H(xb), F(aux), F(wrow), F(ctrl), B, (int)strat);
}
void fk_ln_fwd(torch::Tensor h, torch::Tensor g, torch::Tensor be, torch::Tensor st,
               torch::Tensor a, int64_t blocks, int64_t threads) {
  ln_fwd_kernel<<<blocks, threads, 0, STREAM>>>(H(h), F(g), F(be), F(st), H(a), h.size(0));
}
void fk_ln_bwd(torch::Tensor da, torch::Tensor h, torch::Tensor st, torch::Tensor g,
               torch::Tensor be, torch::Tensor dh, torch::Tensor dst, int64_t ns,
               int64_t blocks, int64_t threads, int64_t nr) {
  size_t sm = (threads / 32) * 768 * sizeof(float);
  if (nr == 2)
    ln_bwd_kernel<2><<<blocks, threads, sm, STREAM>>>(H(da), H(h), F(st), F(g), F(be), H(dh),
        F(dst), (int)ns, da.size(0));
  else
    ln_bwd_kernel<1><<<blocks, threads, sm, STREAM>>>(H(da), H(h), F(st), F(g), F(be), H(dh),
        F(dst), (int)ns, da.size(0));
}
void fk_head_adv(torch::Tensor h3, torch::Tensor Wo, torch::Tensor bo, torch::Tensor aux,
                 torch::Tensor wrow, torch::Tensor dh3, torch::Tensor dst, torch::Tensor lossp,
                 int64_t ns, double inv5b, int64_t blocks, int64_t threads) {
  size_t sm = (threads / 32) * HA_STRIDE * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(head_adv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024); attr = true; }
  head_adv_kernel<<<blocks, threads, sm, STREAM>>>(H(h3), F(Wo), F(bo), F(aux), F(wrow), H(dh3),
      F(dst), F(lossp), (int)ns, (float)inv5b, h3.size(0));
}
void fk_head_strat(torch::Tensor a2, torch::Tensor Wo, torch::Tensor bo, torch::Tensor aux,
                   torch::Tensor wrow, torch::Tensor h2, torch::Tensor st2, torch::Tensor g2,
                   torch::Tensor be2, torch::Tensor dh2, torch::Tensor dst1, torch::Tensor dst2,
                   torch::Tensor lossp, int64_t ns, double invb, int64_t blocks,
                   int64_t threads) {
  size_t sm = (threads / 32) * HS_STRIDE * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(head_strat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024); attr = true; }
  head_strat_kernel<<<blocks, threads, sm, STREAM>>>(H(a2), F(Wo), F(bo), F(aux), F(wrow),
      H(h2), F(st2), F(g2), F(be2), H(dh2), F(dst1), F(dst2), F(lossp), (int)ns, (float)invb,
      a2.size(0));
}
void fk_norm(torch::Tensor G, torch::Tensor S, torch::Tensor mask, torch::Tensor ctrl,
             torch::Tensor lossp, torch::Tensor loss_acc, int64_t small_off) {
  int ns = S.size(1), NB = S.size(0);
  int nbig = (small_off + 1023) / 1024;
  int nsb = (ns + 31) / 32;
  norm_kernel<<<nbig + nsb, 256, 0, STREAM>>>(F(G), F(S), F(mask), F(ctrl), F(lossp),
      F(loss_acc), (int)small_off, ns, NB, nbig);
}
void fk_adam(torch::Tensor P, torch::Tensor Pb, torch::Tensor G, torch::Tensor m, torch::Tensor v,
             torch::Tensor mask, torch::Tensor ctrl, torch::Tensor lr) {
  int n = P.size(0);
  adam_kernel<<<(n + 255) / 256, 256, 0, STREAM>>>(F(P), H(Pb), F(G), F(m), F(v), F(mask),
      F(ctrl), F(lr), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("prep", &fk_prep); mod.def("ln_fwd", &fk_ln_fwd); mod.def("ln_bwd", &fk_ln_bwd);
  mod.def("head_adv", &fk_head_adv); mod.def("head_strat", &fk_head_strat);
  mod.def("norm", &fk_norm); mod.def("adam", &fk_adam);
}

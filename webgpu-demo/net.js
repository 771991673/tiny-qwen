// net.js — Qwen3.5-0.8B WebGPU Runtime
// WGSL compute shader templates + dispatch harness
// Operations are dispatched sequentially, each with templated WGSL

// ─── Shader Templates ────────────────────────────────────────────────────

const WGSL = {};

// Generic tiled MatMul: C[M,N] = A[M,K] @ B[K,N]
// Tile size 16 for good occupancy
WGSL.matmul = (M, N, K) => `
const TILE = 16u;
var<workgroup> sh_a: array<array<f32, TILE>, TILE>;
var<workgroup> sh_b: array<array<f32, TILE>, TILE>;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(TILE, TILE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let row = wgid.y * TILE + lid.y;
    let col = wgid.x * TILE + lid.x;
    var sum = 0.0;
    let tiles = (${K}u + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        let a_idx = row * ${K}u + t * TILE + lid.x;
        sh_a[lid.y][lid.x] = (row < ${M}u && t * TILE + lid.x < ${K}u) ? a[a_idx] : 0.0;
        let b_idx = (t * TILE + lid.y) * ${N}u + col;
        sh_b[lid.y][lid.x] = (t * TILE + lid.y < ${K}u && col < ${N}u) ? b[b_idx] : 0.0;
        workgroupBarrier();
        for (var k = 0u; k < TILE; k++) { sum += sh_a[lid.y][k] * sh_b[k][lid.x]; }
        workgroupBarrier();
    }
    if (row < ${M}u && col < ${N}u) { c[row * ${N}u + col] = sum; }
}`;

// Batched MatMul for attention: C[B,H,N,D] = A[B,H,N,D] @ B[B,H,D,N]
// Each head gets its own workgroup
WGSL.batched_matmul_nt = (B, H, M, N, K) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x; let row = gid.y; let batch_head = gid.z;
    if (row >= ${M}u || col >= ${N}u) { return; }
    let stride = ${M}u * ${K}u;
    let a_base = batch_head * stride + row * ${K}u;
    let b_base = batch_head * ${N}u * ${K}u;
    var sum = 0.0;
    for (var k = 0u; k < ${K}u; k++) {
        sum += a[a_base + k] * b[b_base + k * ${N}u + col];
    }
    c[batch_head * ${M}u * ${N}u + row * ${N}u + col] = sum;
}`;

// Batched MatMul TN: C[B,H,M,N] = A[B,H,M,K] @ B[B,H,K,N]
WGSL.batched_matmul = (B, H, M, N, K) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

const TILE = 16u;
var<workgroup> sh_a: array<array<f32, TILE>, TILE>;
var<workgroup> sh_b: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let bh = gid.z;
    if (bh >= ${B}u * ${H}u) { return; }
    let row = wgid.y * TILE + lid.y;
    let col = wgid.x * TILE + lid.x;
    let b_stride = ${M}u * ${K}u;
    let a_base = bh * b_stride;
    let b_base = bh * ${K}u * ${N}u;
    var sum = 0.0;
    let tiles = (${K}u + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        let ak = t * TILE + lid.x;
        sh_a[lid.y][lid.x] = (row < ${M}u && ak < ${K}u) ? a[a_base + row * ${K}u + ak] : 0.0;
        let bk = t * TILE + lid.y;
        sh_b[lid.y][lid.x] = (bk < ${K}u && col < ${N}u) ? b[b_base + bk * ${N}u + col] : 0.0;
        workgroupBarrier();
        for (var k = 0u; k < TILE; k++) { sum += sh_a[lid.y][k] * sh_b[k][lid.x]; }
        workgroupBarrier();
    }
    if (row < ${M}u && col < ${N}u) {
        c[bh * ${M}u * ${N}u + row * ${N}u + col] = sum;
    }
}`;

// SDPA with causal mask: O[B,H,N,D] = softmax(Q[B,H,N,D] @ K[B,H,D,N] / sqrt(D)) @ V[B,H,N,D]
WGSL.sdpa_causal = (B, H, N, D) => `
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;

var<workgroup> sh: array<f32, 1024>;

const N = ${N}u;
const D = ${D}u;
const SCALE = ${(1.0 / Math.sqrt(256)).toFixed(10)}f32;

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let bh = gid.z;
    let qrow = gid.x;
    if (qrow >= N) { return; }
    let qbase = bh * N * D + qrow * D;
    let kbase = bh * N * D;
    let vbase = bh * N * D;
    let obase = bh * N * D + qrow * D;

    // Compute scores: Q[row] @ K^T (causal)
    // Use tile-based to reduce register pressure
    var max_score = -3.40282e+38;
    var scores: array<f32, 64>; // max N = 64 for tiling
    
    // Actually for general N, we tile over K dimension
    // Simple approach: compute one score per thread
    for (var kv = 0u; kv <= qrow; kv++) {
        var s = 0.0;
        for (var d = 0u; d < D; d += 16u) {
            // Manual dot product loop
            for (var dd = 0u; dd < 16u && d + dd < D; dd++) {
                s += q[qbase + d + dd] * k[kbase + kv * D + d + dd];
            }
        }
        s = s * SCALE;
        // store partial in workgroup shared, but kv can be large... 
        // For small N we can use registers
        // We'll use a two-pass approach: compute scores, then softmax, then weighted sum
    }
}`;

// Element-wise RMSNorm (Qwen3.5 uses GemmaRMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + weight))
WGSL.gemma_rmsnorm = (rows, dim, eps) => `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

var<workgroup> sq_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let row = wgid.x;
    if (row >= ${rows}u) { return; }
    let base = row * ${dim}u;
    var local_sq = 0.0;
    for (var j = lid.x; j < ${dim}u; j += 256u) {
        let v = x[base + j]; local_sq += v * v;
    }
    sq_sum[lid.x] = local_sq;
    workgroupBarrier();
    var i = 128u;
    while (i > 0u) {
        if (lid.x < i) { sq_sum[lid.x] += sq_sum[lid.x + i]; }
        workgroupBarrier(); i >>= 1u;
    }
    let rstd = 1.0 / sqrt(sq_sum[0] / f32(${dim}u) + ${eps}f32);
    for (var j = lid.x; j < ${dim}u; j += 256u) {
        y[base + j] = x[base + j] * rstd * (1.0 + w[j]);
    }
}`;

// VisionEncoder LayerNorm
WGSL.layernorm = (rows, dim, eps) => `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let row = wgid.x;
    if (row >= ${rows}u) { return; }
    let base = row * ${dim}u;
    var s = 0.0;
    for (var j = lid.x; j < ${dim}u; j += 256u) { s += x[base + j]; }
    sh[lid.x] = s; workgroupBarrier();
    var i = 128u; while (i > 0u) {
        if (lid.x < i) { sh[lid.x] += sh[lid.x + i]; }
        workgroupBarrier(); i >>= 1u;
    }
    let mean = sh[0] / f32(${dim}u);
    var v = 0.0;
    for (var j = lid.x; j < ${dim}u; j += 256u) { let d = x[base + j] - mean; v += d * d; }
    sh[lid.x] = v; workgroupBarrier();
    i = 128u; while (i > 0u) {
        if (lid.x < i) { sh[lid.x] += sh[lid.x + i]; }
        workgroupBarrier(); i >>= 1u;
    }
    let rstd = 1.0 / sqrt(sh[0] / f32(${dim}u) + ${eps}f32);
    for (var j = lid.x; j < ${dim}u; j += 256u) {
        y[base + j] = (x[base + j] - mean) * rstd * w[j] + b[j];
    }
}`;

// GELU (tanh approx)
WGSL.gelu = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    let x = a[i]; let p = x * (0.7978845608 + 0.044715 * x * x);
    b[i] = 0.5 * x * (1.0 + tanh(p));
}`;

// SiLU
WGSL.silu = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    b[i] = a[i] / (1.0 + exp(-a[i]));
}`;

// GELU (tanh approx) with numerical stability for extreme inputs
WGSL.gelu = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    let x = a[i];
    // Asymptotic: for large |x| avoid 0*Inf NaN from tanh saturation
    if (x > 5.0) { b[i] = x; }
    else if (x < -5.0) { b[i] = 0.0; }
    else {
        let p = x * (0.7978845608 + 0.044715 * x * x);
        b[i] = 0.5 * x * (1.0 + tanh(p));
    }
}`;

// Sigmoid
WGSL.sigmoid = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    b[i] = 1.0 / (1.0 + exp(-a[i]));
}`;

// Element-wise copy
WGSL.copy = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    b[i] = a[i];
}`;

// Element-wise add: c[i] = a[i] + b[i]
WGSL.add = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    c[i] = a[i] + b[i];
}`;

// In-place add: b[i] += a[i]
WGSL.add_inplace = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    b[i] += a[i];
}`;

// Bias broadcast add: y[i] += bias[i % dim]
WGSL.bias_add = (n, dim) => `
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    y[i] += bias[i % ${dim}u];
}`;

// Element-wise mul: c[i] = a[i] * b[i]  
WGSL.mul = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    c[i] = a[i] * b[i];
}`;

// Scalar multiply: c[i] = a[i] * s
WGSL.scalar_mul = (n) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<uniform> s: f32;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${n}u) { return; }
    c[i] = a[i] * s;
}`;

// Reshape + transpose for attention: [B,T,d] -> [B,H,T,D] where d=H*D
WGSL.reshape_qkv = (B, T, H, D) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= ${B}u * ${T}u * ${H}u * ${D}u) { return; }
    // input layout: [B, T, H*D]
    // output layout: [B, H, T, D]
    let total = i;
    let d = total % ${D}u;
    let h = (total / ${D}u) % ${H}u;
    let t = (total / (${D}u * ${H}u)) % ${T}u;
    let b_ = total / (${D}u * ${H}u * ${T}u);
    let in_idx = b_ * ${T}u * ${H}u * ${D}u + t * ${H}u * ${D}u + h * ${D}u + d;
    let out_idx = b_ * ${H}u * ${T}u * ${D}u + h * ${T}u * ${D}u + t * ${D}u + d;
    b[out_idx] = a[in_idx];
}`;

// Softmax row-wise: [B, H, N, D] softmax over last dim
// Actually for attention softmax: [B, H, N, N] softmax over last dim
WGSL.softmax = (B, H, N) => `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>) {
    let row = wgid.x;
    if (row >= ${B}u * ${H}u * ${N}u) { return; }
    let base = row * ${N}u;
    // max
    var maxv = -3.40282e+38;
    for (var j = lid.x; j < ${N}u; j += 256u) { maxv = max(maxv, x[base + j]); }
    sh[lid.x] = maxv; workgroupBarrier();
    var i = 128u; while (i > 0u) {
        if (lid.x < i) { sh[lid.x] = max(sh[lid.x], sh[lid.x + i]); }
        workgroupBarrier(); i >>= 1u;
    }
    let row_max = sh[0];
    // sum
    var sumv = 0.0;
    for (var j = lid.x; j < ${N}u; j += 256u) { sumv += exp(x[base + j] - row_max); }
    sh[lid.x] = sumv; workgroupBarrier();
    i = 128u; while (i > 0u) {
        if (lid.x < i) { sh[lid.x] += sh[lid.x + i]; }
        workgroupBarrier(); i >>= 1u;
    }
    for (var j = lid.x; j < ${N}u; j += 256u) {
        y[base + j] = exp(x[base + j] - row_max) / sh[0];
    }
}`;

// ─── Verified WGSL Shaders (from dawn-python tests) ─────────────────────

// VisionAttention with cu_seqlens block-diagonal mask
// Q, K, V each [total_tokens, n_heads, head_dim]
// cu_seqlens: [num_blocks+1] cumulative sequence lengths as u32
// Output: O [total_tokens, n_heads, head_dim]
WGSL.vision_attention = (T, H, D) => {
  const SCALE = (1.0 / Math.sqrt(D)).toFixed(10);
  return `
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> cu: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;

const T = ${T}u;
const H = ${H}u;
const D = ${D}u;
const SCALE = ${SCALE};

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    if (flat_idx >= T * H * D) { return; }
    let d_idx = flat_idx % D;
    let h_idx = (flat_idx / D) % H;
    let t_idx = flat_idx / (D * H);

    // Find which block this token belongs to
    var block_start = 0u;
    var block_end = T;
    for (var b = 0u; b < 99u; b++) {
        if (cu[b + 1u] == 0u) { break; }
        if (t_idx < cu[b + 1u]) {
            block_start = cu[b];
            block_end = cu[b + 1u];
            break;
        }
    }

    let q_base = t_idx * H * D + h_idx * D;

    // First pass: find max score (for numerical stability)
    var max_score = -3.40282e+38;
    for (var kv = block_start; kv < block_end; kv++) {
        var s = 0.0;
        let k_base = kv * H * D + h_idx * D;
        for (var dd = 0u; dd < D; dd++) {
            s += q[q_base + dd] * k[k_base + dd];
        }
        s = s * SCALE;
        if (s > max_score) { max_score = s; }
    }

    // Second pass: sum of exp
    var sum_exp = 0.0;
    for (var kv = block_start; kv < block_end; kv++) {
        var s = 0.0;
        let k_base = kv * H * D + h_idx * D;
        for (var dd = 0u; dd < D; dd++) {
            s += q[q_base + dd] * k[k_base + dd];
        }
        s = exp(s * SCALE - max_score);
        sum_exp += s;
    }

    // Third pass: weighted sum for this specific d_idx
    var result = 0.0;
    for (var kv = block_start; kv < block_end; kv++) {
        var s = 0.0;
        let k_base = kv * H * D + h_idx * D;
        for (var dd = 0u; dd < D; dd++) {
            s += q[q_base + dd] * k[k_base + dd];
        }
        let attn = exp(s * SCALE - max_score) / sum_exp;
        result += attn * v[kv * H * D + h_idx * D + d_idx];
    }

    o[flat_idx] = result;
}`;
};

// RoPE for vision: [seq_len, head_dim] with frequencies [seq_len, head_dim/2]
// Applies in-place on output buffer y
WGSL.rope_seq = (seq_len, D) => {
  const half = D / 2;
  return `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> freqs: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

const N = ${seq_len}u;
const D = ${D}u;
const HALF = ${half}u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    if (flat_idx >= N * D) { return; }
    let d = flat_idx % D;
    let t = flat_idx / D;
    if (d < HALF) {
        let other_d = d + HALF;
        let x_cur = x[t * D + d];
        let x_other = x[t * D + other_d];
        let cs = cos(freqs[t * HALF + d]);
        let sn = sin(freqs[t * HALF + d]);
        y[t * D + d] = x_cur * cs - x_other * sn;
        y[t * D + other_d] = x_cur * sn + x_other * cs;
    }
}`;
};

// Conv1d (grouped, used in GatedDeltaNet)
// Input: [B, T, C] -> Conv1d groups=C, kernel=K -> Output: [B, T, C]
WGSL.conv1d_groups = (B, T, C, K) => `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;  // [C, K]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ${B}u * ${T}u * ${C}u) { return; }
    let c = i % ${C}u;
    let t = (i / ${C}u) % ${T}u;
    let b = i / (${C}u * ${T}u);
    var sum = 0.0;
    for (var k = 0u; k < ${K}u; k++) {
        let tp = i32(t) + i32(k) - i32(${K}u - 1u);
        if (tp >= 0 && tp < i32(${T}u)) {
            let idx = b * ${T}u * ${C}u + u32(tp) * ${C}u + c;
            sum += x[idx] * w[c * ${K}u + k];
        }
    }
    y[i] = sum;
}`;

// GatedDeltaNet delta rule scan
// State S[H,D_k,D_v], iterate over T, update: S = S*exp(g_t) + k_t * (v_t - S@k_t)*beta_t
// Output O[t,h,d_v] = S[h,:,d_v] @ q[t,h,:]
WGSL.delta_scan = (B, H, Dk, Dv, T) => `
@group(0) @binding(0) var<storage, read> q: array<f32>; // [B, T, H, Dk]
@group(0) @binding(1) var<storage, read> k: array<f32>; // [B, T, H, Dk]
@group(0) @binding(2) var<storage, read> v: array<f32>; // [B, T, H, Dv]
@group(0) @binding(3) var<storage, read> g: array<f32>; // [B, T, H]
@group(0) @binding(4) var<storage, read> beta: array<f32>; // [B, T, H]
@group(0) @binding(5) var<storage, read_write> o: array<f32>; // [B, T, H, Dv]

const B = ${B}u; const H = ${H}u; const Dk = ${Dk}u; const Dv = ${Dv}u; const T = ${T}u;
var<workgroup> S: array<array<f32, 8>, 8>; // small shared for tile

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = gid.y; let dv = gid.x;
    let b = gid.z;
    if (h >= H || dv >= Dv) { return; }
    // Initialize S as zeros (per head)
    // Simplified: iterate T, update state, write output
    var state: array<f32, 16>; // state[Dk]
    for (var dk = 0u; dk < Dk; dk++) { state[dk] = 0.0; }
    for (var t = 0u; t < T; t++) {
        let g_t = exp(g[b * T * H + t * H + h]);
        let beta_t = beta[b * T * H + t * H + h];
        // Read k[t, h, :] and v[t, h, dv]
        let k_idx = b * T * H * Dk + t * H * Dk + h * Dk;
        let v_val = v[b * T * H * Dv + t * H * Dv + h * Dv + dv];
        // S = S * g_t
        for (var dk = 0u; dk < Dk; dk++) { state[dk] = state[dk] * g_t; }
        // delta = (v - S@k) * beta
        var sdotk = 0.0;
        for (var dk = 0u; dk < Dk; dk++) {
            sdotk += state[dk] * k[k_idx + dk];
        }
        let delta = (v_val - sdotk) * beta_t;
        // S += k * delta
        for (var dk = 0u; dk < Dk; dk++) {
            state[dk] += k[k_idx + dk] * delta;
        }
        // o[t, h, dv] = S @ q[t, h, :]
        let q_idx = b * T * H * Dk + t * H * Dk + h * Dk;
        var out_val = 0.0;
        for (var dk = 0u; dk < Dk; dk++) {
            out_val += state[dk] * q[q_idx + dk];
        }
        o[b * T * H * Dv + t * H * Dv + h * Dv + dv] = out_val;
    }
}`;

// RoPE (partial rotary): applies cos/sin to first rotary_dim elements
WGSL.rope = (B, T, H, D, rotary_dim) => `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_sin: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ${B}u * ${T}u * ${H}u * ${D}u) { return; }
    let d = i % ${D}u;
    let h = (i / ${D}u) % ${H}u;
    let pos = (i / (${D}u * ${H}u)) % ${T}u;
    let rd = ${rotary_dim}u;
    if (d < rd) {
        let half = rd / 2u;
        let pair = d % half;
        let idx_in = pos * half + pair;
        let cs = cos_sin[idx_in * 2u];
        let sn = cos_sin[idx_in * 2u + 1u];
        let x_other = (d < half) 
            ? x[i + half] 
            : x[i - half];
        let x_cur = x[i];
        let rot = (d < half) 
            ? x_cur * cs - x_other * sn
            : x_cur * cs + x_other * sn;
        y[i] = rot;
    } else {
        y[i] = x[i];
    }
}`;

// Embedding lookup: indices [B,T] -> embeddings [B,T,D]
WGSL.embedding = (B, T, D, vocab) => `
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ${B}u * ${T}u * ${D}u) { return; }
    let d = i % ${D}u;
    let t = (i / ${D}u) % ${T}u;
    let b = i / (${D}u * ${T}u);
    let token = indices[b * ${T}u + t];
    if (token < ${vocab}u) {
        out[i] = table[token * ${D}u + d];
    } else {
        out[i] = 0.0;
    }
}`;


// ─── WebGPU Runtime ──────────────────────────────────────────────────────

class WebGPURuntime {
  constructor() {
    this.device = null;
    this.moduleCache = new Map();
    this.pipelineCache = new Map();
    this.bufferPool = [];
    this.nextId = 1;
    this.weightBuffer = null;
    this.weightOffset = {};
  }

  async init() {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU not supported');
    this.device = await adapter.requestDevice({
      maxBufferSize: 1 << 30,
      requiredLimits: {
        maxStorageBufferBindingSize: 1 << 30,
        maxBufferSize: 1 << 30,
        maxComputeWorkgroupSizeX: 256,
        maxComputeInvocationsPerWorkgroup: 256,
      }
    });
    console.log('[WebGPU] Device initialized');
  }

  // Create a GPU buffer
  buffer(size, usage, data = null) {
    const s = align(size, 4);
    const buf = this.device.createBuffer({
      size: s,
      usage,
      label: `buf_${this.nextId++}`,
    });
    if (data) {
      this.device.queue.writeBuffer(buf, 0, data, 0, size);
    }
    return buf;
  }

  // Load weights from ArrayBuffer into a GPU storage buffer
  loadWeights(weightData) {
    this.weightBuffer = this.buffer(
      weightData.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      weightData
    );
    console.log(`[WebGPU] Weights: ${weightData.byteLength} bytes`);
  }

  // Create a shader module from WGSL source (cached)
  getModule(src) {
    const key = hashStr(src);
    if (this.moduleCache.has(key)) return this.moduleCache.get(key);
    const mod = this.device.createShaderModule({ code: src });
    this.moduleCache.set(key, mod);
    return mod;
  }

  // Dispatch a compute shader
  dispatch(code, bindings, workgroupCount, label = 'op') {
    const mod = this.getModule(code);
    const entries = bindings.map((b, i) => ({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: b.type || 'storage' },
    }));
    const bglKey = hashStr(JSON.stringify(entries));
    let bgl = this.pipelineCache.get(`bgl_${bglKey}`);
    if (!bgl) {
      bgl = this.device.createBindGroupLayout({ entries });
      this.pipelineCache.set(`bgl_${bglKey}`, bgl);
    }
    const plKey = hashStr(code) + '_' + bglKey;
    let pipeline = this.pipelineCache.get(plKey);
    if (!pipeline) {
      const plLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      pipeline = this.device.createComputePipeline({
        layout: plLayout,
        compute: { module: mod, entryPoint: 'main' },
      });
      this.pipelineCache.set(plKey, pipeline);
    }
    const bg = this.device.createBindGroup({
      layout: bgl,
      entries: bindings.map((b, i) => ({
        binding: i,
        resource: { buffer: b.resource },
      })),
    });
    const encoder = this.device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroupCount[0], workgroupCount[1] || 1, workgroupCount[2] || 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // Read buffer back to CPU
  async readBuffer(buf, size) {
    const s = align(size, 4);
    const staging = this.device.createBuffer({
      size: s,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buf, 0, staging, 0, s);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = staging.getMappedRange(0, size).slice();
    staging.unmap();
    staging.destroy();
    return data;
  }

  // Synchronize all pending operations
  async sync() {
    await this.device.queue.onSubmittedWorkDone();
  }

  // Create a uniform buffer with a single float
  uniformBuffer(value) {
    const buf = this.buffer(4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(buf, 0, new Float32Array([value]));
    return buf;
  }

  // Get offset helper for weight lookup
  weightOff(name) {
    const w = this.weightOffset;
    if (!(name in w)) throw new Error(`Weight not found: ${name}`);
    return w[name];
  }
}

// Helper: quick string hash for caching
function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return h;
}

function align(x, a) { return Math.ceil(x / a) * a; }


// ─── Exports ─────────────────────────────────────────────────────────────

export { WebGPURuntime, WGSL, hashStr, align };

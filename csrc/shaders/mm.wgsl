// A(M,N) x B(N,K) = C(M,K), where M,N,K are dims
const MAX_DIMS: u32 = 8u;
const BLOCKSIZE: u32 = 16u;
const TILESIZE: u32 = 4u;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,

    self_offset: u32,
    mat2_offset: u32,
    out_offset: u32,
    _pad2: u32,

    self_strides: array<u32, MAX_DIMS>,
    mat2_strides: array<u32, MAX_DIMS>,
    out_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0)
var<storage, read> A: array<f32>; // self

@group(0) @binding(1)
var<storage, read> B: array<f32>; // mat2

@group(0) @binding(2)
var<storage, read_write> C: array<f32>; // out

@group(0) @binding(3)
var<uniform> params: Params;

// one thread = one output element C[column, row]
@compute @workgroup_size(BLOCKSIZE, TILESIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x * TILESIZE; // 0..K-1
    let row = gid.y; // 0..M-1
    if (col >= params.K || row >= params.M) { return; }

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var k: u32 = 0u; k < params.N; k = k + 1u) {
        let a_idx = params.self_offset + row * params.N + k;
        let b_idx = params.mat2_offset + k * params.K + col;
        acc0 = acc0 + A[a_idx] * B[b_idx];
        acc1 = acc1 + A[a_idx] * B[b_idx + 1u];
        acc2 = acc2 + A[a_idx] * B[b_idx + 2u];
        acc3 = acc3 + A[a_idx] * B[b_idx + 3u];
    }
    let c_idx = params.out_offset + row * params.K + col;
    C[c_idx] = acc0;
    C[c_idx+1u] = acc1;
    C[c_idx+2u] = acc2;
    C[c_idx+3u] = acc3;
}
// Tiled matrix multiplication with shared memory
// A(M,N) x B(N,K) = C(M,K)
// Each workgroup computes a TILE_M x TILE_N block of the output
// Uses shared memory to cache tiles for better memory bandwidth utilization

const MAX_DIMS: u32 = 8u;
const TILE_SIZE: u32 = 16u;

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
var<storage, read> A: array<f32>; // self (M x N)

@group(0) @binding(1)
var<storage, read> B: array<f32>; // mat2 (N x K)

@group(0) @binding(2)
var<storage, read_write> C: array<f32>; // out (M x K)

@group(0) @binding(3)
var<uniform> params: Params;

// Shared memory for tiles
var<workgroup> tile_A: array<f32, 256>;  // TILE_SIZE * TILE_SIZE = 16 * 16
var<workgroup> tile_B: array<f32, 256>;  // TILE_SIZE * TILE_SIZE = 16 * 16

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tx = lid.x;  // Thread x within workgroup (0-15)
    let ty = lid.y;  // Thread y within workgroup (0-15)

    // Global row and column this thread computes
    let row = wid.y * TILE_SIZE + ty;  // Row in C
    let col = wid.x * TILE_SIZE + tx;  // Col in C

    // Accumulator for dot product
    var acc: f32 = 0.0;

    // Number of tiles along the N dimension
    let num_tiles = (params.N + TILE_SIZE - 1u) / TILE_SIZE;

    // Iterate over tiles
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        // A[row, t*TILE_SIZE + tx]
        let a_row = row;
        let a_col = t * TILE_SIZE + tx;
        var a_val: f32 = 0.0;
        if (a_row < params.M && a_col < params.N) {
            let a_idx = params.self_offset + a_row * params.self_strides[0] + a_col * params.self_strides[1];
            a_val = A[a_idx];
        }
        tile_A[ty * TILE_SIZE + tx] = a_val;

        // Load tile of B into shared memory
        // B[t*TILE_SIZE + ty, col]
        let b_row = t * TILE_SIZE + ty;
        let b_col = col;
        var b_val: f32 = 0.0;
        if (b_row < params.N && b_col < params.K) {
            let b_idx = params.mat2_offset + b_row * params.mat2_strides[0] + b_col * params.mat2_strides[1];
            b_val = B[b_idx];
        }
        tile_B[ty * TILE_SIZE + tx] = b_val;

        // Synchronize to make sure tiles are loaded
        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + tile_A[ty * TILE_SIZE + k] * tile_B[k * TILE_SIZE + tx];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result to output
    if (row < params.M && col < params.K) {
        let c_idx = params.out_offset + row * params.K + col;
        C[c_idx] = acc;
    }
}
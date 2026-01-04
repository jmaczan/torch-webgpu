// TODO: improve performance, currently it's a native impl
// based on my first CUDA matrix multiplication kernel
// A(M,N) x B(N,K) = C(M,K), where M,N,K are dims

/*
A - dim (M, N), let's say (3, 2) = 3 rows 3 cols
a0 a1
a2 a3
a4 a5
------------------------
B - dim (N, K), say (2, 4) = 2 rows 4 cols
b0 b1 b2 b3
b4 b5 b6 b7
------------------------
C - dim (M, K) = (3, 4) = 3 rows 4 cols
c0 c1 c2 c3
c4 c5 c6 c7
c8 c9 c10 c11

assuming row major

numel C = 3 * 4 = 12
ndim = 2
// mathematically; as coordinates
c[0, 0] = row 0 of A * col 0 of B = a0b0 + a1b4
c[0, 1] = row 0 of A * col 1 of B = a0b1 + a1b5
c[0, 2] = row 0 of A * col 2 of B = a0b2 + a1b6
c[0, 3] = row 0 of A * col 3 of B = a0b3 + a1b7
c[1, 0] = row 1 of A * col 0 of B = a2b0 + a3b4
c[1, 1] = row 1 of A * col 1 of B = a2b1 + a3b5
c[1, 2] = row 1 of A * col 2 of B = a2b2 + a3b6
c[1, 3] = row 1 of A * col 3 of B = a2b3 + a3b7
...

// programmatically; as a memory
C[0] = c[0, 0]
C[1] = c[0, 1]
C[2] = c[0, 2]
C[3] = c[0, 3]
C[4] = c[1, 0]
...
*/
const MAX_DIMS: u32 = 8u;

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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // say I dispatched M * K workgroups and workgroup_size is 1
    // and now say gid.x is 4, then how do I pick correct row of A and col of B to compute C[gid.x]?
    // I know M is 3 and K is 4
    // I want to compute coordinates of c[c_x, c_y]. c_x rows, c_y cols.
    // row major, so c_y changes faster - by 1 - and c_x changes every K elements
    // I computed by hand that C[4] = c[1, 0], so let's try to reverse that
    // c_x = gid.x / M and rounded to lower int? // changes slower
    // c_y = gid.x % K // changes faster
    if (gid.x < params.M * params.K) {
        var c_x: u32 = gid.x / params.K;
        var c_y: u32 = gid.x % params.K; 
        // now I know c[c_x, c_y] and I can iterate over c_x..c_x+N (?) and c_y..c_y+N (not sure if correct?)
        var output: f32 = 0.0;
        for (var i: u32 = 0; i < params.N; i = i + 1u) {
            output = output + A[c_x*params.N + i] * B[i * params.K + c_y]; // + A[c_x + i] * B[c_y + i];
        }
        C[gid.x] = output;
    }
}
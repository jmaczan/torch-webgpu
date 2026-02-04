
import wgpu
import numpy as np
import time
import json
from pathlib import Path

def benchmark_matmul(device, m, k, n, n_warmup=5, n_runs=30, tile_size=16):
    """Benchmark matrix multiply at given dimensions."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = adapter.request_device_sync()

    # Create matrices
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    # Create buffers
    a_buf = dev.create_buffer_with_data(data=a.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    b_buf = dev.create_buffer_with_data(data=b.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    c_buf = dev.create_buffer(size=m * n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    # Tiled matmul shader
    shader_code = f"""
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> C: array<f32>;

    const M: u32 = {m}u;
    const K: u32 = {k}u;
    const N: u32 = {n}u;
    const TILE: u32 = {tile_size}u;

    var<workgroup> tileA: array<f32, {tile_size * tile_size}>;
    var<workgroup> tileB: array<f32, {tile_size * tile_size}>;

    @compute @workgroup_size({tile_size}, {tile_size})
    fn main(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {{
        let row = gid.x;
        let col = gid.y;
        let localRow = lid.x;
        let localCol = lid.y;

        var sum: f32 = 0.0;
        let numTiles = (K + TILE - 1u) / TILE;

        for (var t: u32 = 0u; t < numTiles; t = t + 1u) {{
            let tiledK = t * TILE + localCol;
            let tiledRow = t * TILE + localRow;

            if (row < M && tiledK < K) {{
                tileA[localRow * TILE + localCol] = A[row * K + tiledK];
            }} else {{
                tileA[localRow * TILE + localCol] = 0.0;
            }}

            if (tiledRow < K && col < N) {{
                tileB[localRow * TILE + localCol] = B[tiledRow * N + col];
            }} else {{
                tileB[localRow * TILE + localCol] = 0.0;
            }}

            workgroupBarrier();

            for (var i: u32 = 0u; i < TILE; i = i + 1u) {{
                sum = sum + tileA[localRow * TILE + i] * tileB[i * TILE + localCol];
            }}

            workgroupBarrier();
        }}

        if (row < M && col < N) {{
            C[row * N + col] = sum;
        }}
    }}
    """

    shader = dev.create_shader_module(code=shader_code)

    bind_group_layout = dev.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
    ])

    pipeline_layout = dev.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

    pipeline = dev.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "main"}
    )

    bind_group = dev.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": a_buf}},
            {"binding": 1, "resource": {"buffer": b_buf}},
            {"binding": 2, "resource": {"buffer": c_buf}},
        ]
    )

    # Calculate workgroup counts
    wg_x = (m + tile_size - 1) // tile_size
    wg_y = (n + tile_size - 1) // tile_size

    # Warmup
    for _ in range(n_warmup):
        encoder = dev.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(wg_x, wg_y)
        compute_pass.end()
        dev.queue.submit([encoder.finish()])
        dev.queue.read_buffer(c_buf)  # Sync

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()

        encoder = dev.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(wg_x, wg_y)
        compute_pass.end()
        dev.queue.submit([encoder.finish()])
        dev.queue.read_buffer(c_buf)  # Sync

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "dimensions": f"{m}x{k}x{n}",
        "workgroups": f"{wg_x}x{wg_y}",
        "tile_size": tile_size,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "all_times_ms": times
    }


if __name__ == "__main__":
    print("Testing mega-kernel at production dimensions...")

    results = {
        "timestamp": str(np.datetime64("now")),
        "experiments": {}
    }

    # Test at toy scale (256x256) - baseline
    print("\n1. Toy scale (256x256x256)...")
    results["experiments"]["toy_256"] = benchmark_matmul(None, 256, 256, 256)
    print(f"   Mean: {results['experiments']['toy_256']['mean_ms']:.3f} ms")

    # Test at production scale (896x896 -> 896x4864 for MLP up projection)
    print("\n2. Production MLP up (896x896x4864)...")
    try:
        results["experiments"]["prod_mlp_up"] = benchmark_matmul(None, 896, 896, 4864)
        print(f"   Mean: {results['experiments']['prod_mlp_up']['mean_ms']:.3f} ms")
    except Exception as e:
        print(f"   Failed: {e}")
        results["experiments"]["prod_mlp_up"] = {"error": str(e)}

    # Test at production scale (896x4864x896 for MLP down projection)
    print("\n3. Production MLP down (896x4864x896)...")
    try:
        results["experiments"]["prod_mlp_down"] = benchmark_matmul(None, 896, 4864, 896)
        print(f"   Mean: {results['experiments']['prod_mlp_down']['mean_ms']:.3f} ms")
    except Exception as e:
        print(f"   Failed: {e}")
        results["experiments"]["prod_mlp_down"] = {"error": str(e)}

    # Save results
    output_file = Path("results/reviewer_round3") / "production_scale_mega_kernel.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

kernel void reduce(global float *a, global float *sums,
                   const uint n)
{
    uint x = get_global_id(0);
    uint t = get_local_id(0);
    uint g = get_group_id(0);
    uint offset = 1;
    uint local_i = 2 * t;
    uint global_i = 2 * x;
    local float b[LOCAL_WORK_SIZE * 2];

    b[local_i] = global_i >= n ? 0.0f : a[global_i];
    b[local_i + 1] = global_i + 1 >= n ? 0.0f : a[global_i + 1];

    for (uint i = LOCAL_WORK_SIZE; i > 0; i >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (t < i)
        {
            int ai = offset * (local_i + 1) - 1;
            int bi = offset * (local_i + 2) - 1;

            b[bi] += b[ai];
        }
        offset <<= 1;
    }

    local float last;
    if (t == 0)
    {
        sums[g] = b[LOCAL_WORK_SIZE * 2 - 1];
    }
}

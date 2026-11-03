# octex design

octex (Octa Explosion) is billion-scale approximate nearest-neighbor search on NVIDIA GPUs by precomputed lattice
neighbor lists. In hundreds of randomly projected copies of the dataset, every vector rounds to a product of
eight-dimensional E8 lattice cells (its fragment), and every populated fragment stores the list of vectors nearest to its
center (a fragment ripple). A query reads the lists of its own fragments in all copies at once; in a fixed number of
waves, its best hits then read their rows of the neighbor table (vector ripples). An int8 sketch ranks every candidate
before one exact fp16 rerank. The whole search of a batch runs as CUDA graphs over all GPUs of a server.

The code is laid out as a cuVS module (`cuvs::neighbors::octex`, `cpp/include/cuvs/neighbors/octex.hpp`,
`cpp/src/neighbors/octex/`), builds against the libcuvs pip wheel, and has a nanobind Python binding (package
`octex`). No PyTorch anywhere in the library. Search is pure CUDA C++ (cuBLAS for the two projections).

## Terms

| name | meaning |
|---|---|
| N, D | dataset size and dimension |
| C | copies (random projections) |
| b | E8 blocks per copy (default 4), so a copy projects into 8b dimensions |
| L | spacings (levels), fine to coarse, default 2.0, 2.5, 3.2, 4.0, 5.0 (projected standard deviations) |
| l | list length: entries stored per populated cell |
| T = 2^logT | hash buckets per (copy, level), logT = 20 |
| S | sketch width (int8 coordinates per vector, top-S principal directions) |
| L2 | neighbor-table row length |

## Index directory (format version 1)

An index is a directory of `.npy` files (little endian, C order) plus `meta.json`. Builders write it; `load()` reads
it (memory mapped) and places it on the GPUs. `meta.json` may give an absolute path for any array in a `"files"`
map (large arrays such as `vectors` or `sketch` are then not copied).

| file | dtype, shape | content |
|---|---|---|
| `meta.json` | | `{"format": 1, "n": N, "dim": D, "copies": C, "blocks": b, "spacings": [...], "list_len": l, "log_buckets": 20, "sketch_width": S, "table_len": L2 or 0, "metric": "sqeuclidean", "files": {}}` |
| `mean.npy` | f32 [D] | data mean mu |
| `proj.npy` | f32 [C, D, 8b] | projection of copy c, already divided by the projected standard deviation |
| `shift.npy` | f32 [C, 8b] | random shift of copy c (uniform in [0, 2)) |
| `lists.npy` | i32 [C, Lmax] | per copy, the lists of all levels concatenated (cell rows of l ids, nearest first), zero padded |
| `keys.npy` | i64 [C, L, Mmax] | per copy and level, the 64-bit hashes of the populated cells in (bucket, hash) order, zero padded |
| `offsets.npy` | i32 [C, L, T + 1] | per copy and level, CSR over buckets into `keys` |
| `base.npy` | i32 [C, L] | per copy, first cell row of every level within `lists` |
| `cells.npy` | i64 [C, L] | populated cells per copy and level (statistics) |
| `sketch_basis.npy` | f32 [D, S] | top-S principal directions |
| `sketch_scale.npy` | f32 [S] | per coordinate scale (amax / 127) |
| `sketch.npy` | i8 [N, S] | round(((x - mu) @ basis) / scale) |
| `table.npy` | i32 [N, L2] | neighbor table: per vector, L2 near neighbors (reverse edges merged in), -1 padded |
| `vectors.npy` | f16 [N, D] | the vectors, for the exact rerank |

Cell hash: for a projected, shifted vector y (8b floats) and spacing f, each block of eight coordinates x = y / f
decodes to the nearest point of E8 (the nearer of the nearest D8 point of x and the nearest D8 point of x - 1/2,
plus 1/2), and the 64-bit hash is the one in `detail/e8.cuh` (`e8b_hash64`; primes PR, mixer 0x9E3779B97F4A7C15,
final xor-shift 31 / multiply / xor-shift 29, signed arithmetic shifts). Bucket = hash & (T - 1). Builder and search
must use the same device function.

## Search pipeline (per GPU, one batch of m queries, captured once as a CUDA graph and replayed)

1. Center: qc = q - mu (fp32), qh = half(qc).
2. Project: Y = qh @ Ucat + shift (cuBLAS fp16 GEMM, fp16 out; Ucat [D, C_g * 8b] = this GPU's copies) and
   qs = qc @ basis (fp32 GEMM, [m, S]).
3. Probe: per (query, copy) the finest level whose cell has a list; segment = (row * l, read); per-query read total.
4. Hash-set size per query: next power of two >= 1.5 x total, clamped to [2^11, W]; W <= 32768.
5. Gather (balanced): distinct ids of all segments with their int8 sketch distances.
6. First-round selection: exact fp16 rerank of the R sketch-nearest, keep the best KO = starts (ids and distances).
   Sketch-rank mode (billion scale): keep the KO sketch-nearest instead (fused sketch merge), no exact rerank.
7. Cross-GPU merge (sharded placement): each GPU owns a contiguous slice of the batch for the waves; it reads the
   kept lists of its slice from every GPU through peer pointers and merges them (distinct ids, best KO).
8. Waves (W rounds): pick the best S kept entries not expanded before (fused pick-starts kernel); gather their table
   rows (rows sharded over GPUs, read through peer pointers, or one host pointer); exact rerank of the R2
   sketch-nearest merged into the kept list (fused sketch merge in sketch-rank mode). The last wave merges into the
   best k.
9. Sketch-rank mode: final exact rerank of the best `final_rerank` kept entries into the best k.

Every step is a kernel or a cuBLAS call on one stream: no host synchronization inside a batch. Batches larger than
the captured size are split; the last partial batch is padded with copies of its first query.

Placements:
- `replicate`: every GPU holds the whole index and answers its own slice of the queries (no peer traffic).
- `shard`: copies split over the GPUs (round robin by saved part), table rows and optionally vectors split by row
  ranges and read through peer pointers; each GPU runs the first round for all queries over its copies, the waves
  for its slice.
- vectors and table can stay in pinned host memory (`vectors_on_host`, `table_on_host`), read in place over PCIe.

## Source

The kernels come from the research prototype (raw-pointer CUDA, no framework types). Kernel names in the prototype
and their role here: `probe_kernel` (step 3), `gather9_kernel` (step 5), `select_rerank3_kernel` (keep, steps 6 and
8), `select_rerank2_kernel` (best k), `peer_gather_kernel` (step 7), `gather10_kernel` (step 8, rows through
pointers), `gather7_kernel` (step 8, one local table), `sketch_merge_kernel`, `pick_starts_kernel`. New kernels
replace the framework ops of the prototype: centering, hash-set sizes, merge of the per-GPU kept lists (sort by id,
drop duplicates, top KO by distance), fills.

## Build

Per copy c (seeded): random rotation of the top-`pca_dims` principal subspace into 8b dimensions, divided by the
projected standard deviation, uniform shift in [0, 2). Per level: hash every vector, unique cells with population,
drop cells below `min_cell` (all levels but the coarsest), cell center = E8 point of a member times f, list = the l
vectors nearest to the center in the projection (cuVS IVF-Flat on the projection, or brute force when
`exact_lists`), sort cells by (bucket, hash), CSR offsets. Neighbor table: exact (cuVS brute force) up to
`exact_table_max` vectors, else built by octex itself (a search over the dataset with the lists only, then waves);
reverse edges merged in. Sketch: top-S principal directions, scale = amax / 127.

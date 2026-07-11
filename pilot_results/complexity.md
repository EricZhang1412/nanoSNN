# Gate-path complexity

Counts are per sample and exclude shared Q/K/V/proj/MLP work.

| label | T | depth | H | D | FP-mults/block/step | FP-mults/model/step | FP-mults/model/seq | gate params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dvs128/c0_sdla | 16 | 2 | 4 | 64 | 0 | 0 | 0 | 0 |
| dvs128/c1_lowrank | 16 | 2 | 4 | 64 | 24576 | 49152 | 786432 | 4096 |
| dvs128/c2_oneminusk | 16 | 2 | 4 | 64 | 16384 | 32768 | 524288 | 0 |
| dvs128/c3_mga | 16 | 2 | 4 | 64 | 0 | 0 | 0 | 1304 |
| shd/c0_sdla | 100 | 2 | 4 | 64 | 0 | 0 | 0 | 0 |
| shd/c1_lowrank | 100 | 2 | 4 | 64 | 24576 | 49152 | 4915200 | 4096 |
| shd/c2_oneminusk | 100 | 2 | 4 | 64 | 16384 | 32768 | 3276800 | 0 |
| shd/c3_mga | 100 | 2 | 4 | 64 | 0 | 0 | 0 | 1304 |

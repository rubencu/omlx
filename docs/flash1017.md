# DeepSeek Flash1017 profile

Flash1017 is an opt-in oMLX profile for the pinned
[`mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp` checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`. It runs the unchanged
weights through oMLX's web interface and OpenAI-compatible API. It does not
produce a new model conversion.

The package is `0.7.1rc1+flash1017.2`, on branch `flash1017-preview`, based on
public oMLX `v0.7.1rc1` (`7a5a3d75ce7463300bdfa5fedccd34348f60ff46`).
No fork release has been published yet. Follow the
[installation guide](flash1017-install.md) to build from this checkout and
launch an isolated server.

## Three active optimizations

| Optimization | Behavior |
|---|---|
| BF16 vocabulary-weight reuse | [head.py](../omlx/patches/deepseek_v41/head.py) shares weight loads across two through five short verification rows in target and draft/Markov projections. Each row retains the stock reduction structure. Row six and unsupported shapes use the stock fallback. |
| Bounded Engram row cache | [rows.py](../omlx/patches/deepseek_v41/flash1017/rows.py) integrates [HotRows](../omlx/patches/deepseek_v41/flash1017/hot_rows.py) with the two disk-backed tables. It retains up to 256 MiB per table, 512 MiB combined for payload, full-ID tags and the frequency sketch. Allocation failure bypasses the cache. This cache reuses model data; it is separate from conversation KV/prefix caching. |
| Engram prefetch | [loading.py](../omlx/patches/deepseek_v41/loading.py) and [storage.py](../omlx/patches/deepseek_v41/storage.py) use the native background disk reader with a model-specific 64 MiB admission budget, compared with upstream's 16 MiB. One pending read is drained at the native boundaries. |

The profile retains stock `0.7.1rc1` attention arithmetic, prompt chunks of up
to 2,048 tokens and the native acceptance-based MTP controller. MTP starts at
three drafts and adjusts between one and three from token acceptance. Its
configured maximum is not a fixed depth for every verification cycle.

## Requirements and serving limits

The evaluated machine is an **M5 Ultra Mac Studio with 80 GPU cores and
256 GiB unified memory**. Capacity checks allow a Mac with at least 256 GiB,
but other hardware has not been evaluated. The wheel requires Apple Silicon,
CPython 3.12 and macOS 15 or later.

- **At least 210 GiB available before loading**, plus sufficient recommended
  MLX working-set capacity. Unload other models before starting this workload.
- **131,072 tokens total context**, including prompt and generated tokens.
  A smaller explicit context setting remains smaller. The cap does not
  guarantee that every request fits; the performance suite reaches 64K inputs.
- **One active request**, with additional requests queued. The profile is for
  one Mac and has no parallel-request throughput qualification.
- **2,048-token maximum prompt chunks**. The native memory guard may shrink
  chunks or reject admission.
- **32 GiB physical-memory reserve** in the MLX limit calculation,
  **8 GiB allocator-cache guideline** and a **12 GiB available-memory floor**
  checked around shard loading. These are separate from the Engram row cache
  and conversation caches; they do not cap total process RSS.

MLX wired-memory and allocator guidelines affect the entire server process.
On the evaluated 256 GiB machine, the profile sets a 224 GiB limit. These
guidelines do not guarantee a particular amount of free system memory.
Temporary CPU gather arrays and in-flight prefetch data are additional to the
512 MiB retained row-cache budget.

## Activation and lifecycle

Use `omlx flash1017 --model /path/to/model` as shown in the installation guide.
The command creates an isolated server configuration and a symlink to the
existing checkpoint. `--check` verifies four pinned metadata files without
loading weights; it does not rehash every weight shard. `--configure-only`
prepares settings without starting the server.

The model setting `deepseek_v41_flash1017_enabled` defaults to false and is
excluded from cross-model settings profiles. Checkpoint JSON cannot activate
the profile. The loader checks the checkpoint before allocation and refuses
unsupported combinations, including distributed execution, external engines,
expert offload, quantized KV caches, SpecPrefill and other speculative decoders.
The prefix-cache namespace is `flash1017-v2`.

[runtime.py](../omlx/patches/deepseek_v41/flash1017/runtime.py) owns the row
cache and coordinates shutdown with the native reader. The loader attaches
controllers before serving and closes partial loads on the inference worker.
Use normal server unload/shutdown paths so the worker is drained and
synchronized before model resources are released. CPU lifecycle locks alone
do not establish GPU completion.

## Tested results

The evaluated package is `0.7.1rc1+flash1017.2`, from runtime commit
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`. The tested CPython 3.12
Apple Silicon wheel has SHA-256:

```text
0fbdb77ca86ce35b4dca0275b534c1fafde7d70f0f7f6a89d5f964821f7f7f06
```

Results apply to that binary and its recorded dependencies. Rebuilding the
checkout can produce a different wheel checksum, including when only
documentation or build metadata changes.

### Output preservation

All **618 fixed greedy responses** matched public oMLX `v0.7.1rc1` in text and
finish reason, with zero task-score regressions. Both builds scored:

| Evaluation | Passing tasks |
|---|---:|
| HumanEval+ | 148 / 164 |
| MBPP+ | 299 / 378 |
| Fixed GSM8K test subset | 61 / 64 |
| Authored long-context controls | 11 / 12 |

The same four coding responses reached their output budgets in both builds.
This is finite observed preservation, not universal numerical or answer
equivalence. The long-context cases are engineering controls, not a standard
long-context benchmark; the GSM8K result covers only the selected 64 questions.

Coding used a pinned macOS EvalPlus adapter with isolated, seeded workers,
not an unmodified leaderboard run. Its canonical check passed 540/542 tasks:
the `HumanEval/32` completeness check and `Mbpp/630` sanitizer behavior remain
known evaluator limitations in both denominators. Seeds do not control every
possible source of nondeterminism.

### Performance

The installed-wheel comparison used **32 measured requests and four excluded
warmups**: eight workloads in two paired blocks, with fresh baseline/candidate
processes ordered ABBA and reversed workload order in the second block.

| Workload group | Higher generation throughput | Lower first-visible-token latency |
|---|---:|---:|
| All eight: two synthetic 1K and six natural-text cases | **7.9%** | **4.5%** |
| Six natural-text cases: four 16K and two 64K | **8.0%** | **5.5%** |

These are equal-case geometric means of paired ratios. All **16 pairs** matched
in text, finish reason and token counts. Generation throughput improved in all
16 pairs; first-token latency improved in 11 of 16 and varied between blocks.

Ordinary EOS stopping was enabled with a 512-token output budget. Five
workloads reached that budget in both builds. Completed-answer latency applies
only to the three other workloads, whose six paired observations measured a
7.6% reduction.

Models were loaded before timing, conversation-prefix reuse was disabled, and
the identical warmup preceded each measured sequence. Engram row-cache state
was retained within a process and OS caches were not purged. These results
describe warm loaded-model operation, not startup or cold-storage performance.
Two blocks on one Mac provide no confidence interval, tail-latency estimate
or independent replication.

Observed cumulative MLX allocator peaks included loading and warmup. Candidate
peaks were about 77.8 MiB higher in one block and equal in the other; these are
not full process/system memory measurements or evidence of memory savings.
The candidate retained 512 MiB of CPU row-cache data. Effective wired-memory
policies also differed: upstream raised its startup limit to 243.2 GiB, while
the profile set 224 GiB before loading.

The companion `flash1017-research` checkout contains the protocols, raw
responses, grades, dependency and source identities, and analyses under
`validation/quality-v2-01/` and `validation/performance-v2-01/`. That companion
repository has not yet been published.

## Source, packaging and credit

The runtime is contained in this package and does not load scripts from a
research directory. [provenance.json](../omlx/patches/deepseek_v41/flash1017/provenance.json)
records source identities. The [wheel helper](../scripts/build_flash1017_wheel.py)
packages the checkout's Python, web and data files with ten hash-pinned public
oMLX native assets. It verifies their native source/build inputs against the
public base, preserves licenses and records build provenance. See the
installation guide for the build contract and compatibility tag.

DeepSeek supplies the model and DSpark; mlx-community supplies the unchanged
quantized checkpoint. Apple MLX, MLX-LM and oMLX provide the inference
foundation. Related BF16 row reuse appears in
[tacos8me/m5-ultra](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/head.py).
HotRows is **copied verbatim** from that project's pinned implementation,
credited in [NOTICE](../omlx/patches/deepseek_v41/flash1017/NOTICE), with its
[Apache-2.0 license](../omlx/patches/deepseek_v41/flash1017/HOT_ROWS_LICENSE)
retained. Per-file MIT/Apache notices also apply. Integration, lifecycle
handling and validation of this exact checkpoint are separate contributions
from those underlying techniques.

## Maintenance checks

Focused tests cover [inference contracts](../tests/test_flash1017_early.py),
[runtime ownership and caching](../tests/test_flash1017_runtime.py),
[prefetch budgets](../tests/test_flash1017_prefetch.py),
[engine integration](../tests/test_flash1017_pool.py),
[profile settings](../tests/test_flash1017_profile.py) and
[packaging](../tests/test_flash1017_packaging.py). CPU fixtures test contracts;
they do not replace installed-artifact numerical and serving validation.

Keep opt-in defaults, checkpoint checks, fallback and worker-drain requirements
when changing the runtime. The wheel helper rejects changes to native sources,
their build setup or the MLX ABI pin; those require new compiled assets and
validation. Bind any new performance claim to its actual source, wheel and
dependency identities.

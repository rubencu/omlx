# The Flash1017 profile

This fork is the maintained implementation for ongoing development and an
installable experimental distribution. The separate
[planned `rubencu/flash1017-research` repository](https://github.com/rubencu/flash1017-research)
holds the immutable historical source/evidence and article; that repository
has not yet been published. Its local sibling is `../flash1017-research`.
Runtime changes belong here, not in the historical research snapshot.

The active development branch is `flash1017-preview`, initially based on public
oMLX `v0.7.1rc1`, commit `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`.
The package version is `0.7.1rc1+flash1017.1`. New installation, serving and
performance evidence must identify the actual fork commit and wheel hash.
Source integration does not transfer an older benchmark to a different build.

## Intended use and activation

The exact checkpoint is
[`mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`. Its weights and quantization
are unchanged. This work does not produce another Hugging Face model artifact.

Use the isolated `omlx flash1017 --model /path/to/model` workflow in the
[installation guide](flash1017-install.md). `--check` validates the four pinned
metadata files without loading weights. `--configure-only` prepares the server
settings without starting it. The CLI uses a separate base directory, creates
a reference to the existing checkpoint and enables the model-local
`deepseek_v41_flash1017_enabled` setting. Metadata hashes do not substitute for
hashing every weight shard.

The setting defaults to false and is excluded from cross-model settings
profiles. Checkpoint JSON cannot activate the whole profile. Loading rejects
the wrong architecture/checkpoint and unsupported combinations: distributed
execution, bypassing the intended engine, external engines, expert offload,
quantized KV caches, SpecPrefill and other speculative decoders. Attention
rounding also has a separate, explicit metadata compatibility flag; that flag
alone does not enable Flash1017.

## What is included

| Component | Behavior and implementation |
|---|---|
| BF16 vocabulary projection | [head.py](../omlx/patches/deepseek_v41/head.py) shares weight loads across two through six short verification rows, used by both target and draft/Markov projections when opted in. Rows two through five preserve the previous per-row reduction structure. Row six replaces a different fallback and has a separate numerical scope; unsupported shapes use the existing path. |
| Attention-output rounding | [language.py](../omlx/patches/deepseek_v41/language.py), [dspark.py](../omlx/patches/deepseek_v41/dspark.py) and [config.py](../omlx/patches/deepseek_v41/config.py) apply the reference activation-quantization boundary before the grouped output projection in target and draft attention. The complete profile explicitly enables it. This is a compatibility change, not a general answer-quality claim. |
| Frozen confidence admission | [policy.py](../omlx/patches/deepseek_v41/flash1017/policy.py) and [batch_generator.py](../omlx/patches/mlx_lm_mtp/batch_generator.py) use DSpark confidence and fixed calibration/costs to admit a causal verification prefix. Draft width is five; admission stops at the first non-improving prefix and may select zero draft tokens. That still pays for the fixed-width proposal. Live request timings never recalibrate the policy. Context-copy cycles retain their existing path. |
| Decoder replay | [cone.py](../omlx/patches/deepseek_v41/flash1017/cone.py) and [cone_arithmetic.py](../omlx/patches/deepseek_v41/flash1017/cone_arithmetic.py) compute the decoder suffixes needed for the pinned dependency layout. Eligibility checks and fallback remain part of the implementation. Native CED is disabled for this profile so two replay strategies are not layered together. |
| Shape-sensitive numerical repairs | [shape_arithmetic.py](../omlx/patches/deepseek_v41/flash1017/shape_arithmetic.py) preserves projection/split-K, gate, hyper-connection and local WKV behavior where shrinking shapes changes arithmetic. It restores the final expert-dispatch boundary when a short suffix would select a different kernel. Temporary padding is removed from the result; it is not another attention token. These are specific shape repairs, not a general equivalence theorem. |
| Draft-state priming | [prime_arithmetic.py](../omlx/patches/deepseek_v41/flash1017/prime_arithmetic.py) preserves the relevant arithmetic in all three draft-stage projections, crops before normalization/cache append and verifies the expected priming path completes. Matching the target state alone is insufficient to establish matching draft state. |
| Engram prefetch | [loading.py](../omlx/patches/deepseek_v41/loading.py) uses oMLX's existing disk-backed Engram reader and [storage.py](../omlx/patches/deepseek_v41/storage.py) accepts a model-specific admission budget. The profile raises it from the upstream 16 MiB default to 64 MiB; there is one pending background read, drained at the existing boundaries. |
| Bounded row reuse | [rows.py](../omlx/patches/deepseek_v41/flash1017/rows.py) integrates the credited [HotRows implementation](../omlx/patches/deepseek_v41/flash1017/hot_rows.py) with the two disk tables: at most 256 MiB per table, 512 MiB combined retained cache payload/tags/sketch. It coordinates with the native reader, handles cache-allocation failure by bypassing the cache, and retains fallback and cleanup paths. This is distinct from KV/prefix caching. |
| Runtime ownership | [runtime.py](../omlx/patches/deepseek_v41/flash1017/runtime.py) owns replay/cache installation, per-model enable/disable, bounded diagnostic snapshots and shutdown. Controls require drained request boundaries. The integration rejects overlapping/foreign ownership and restores owned attributes while releasing references; inference methods are not patched process-wide. |
| Loader and engine integration | [loading.py](../omlx/patches/deepseek_v41/loading.py), [engine_pool.py](../omlx/engine_pool.py) and [engine/vlm.py](../omlx/engine/vlm.py) resolve effective opt-in settings before sizing and engine selection, check metadata before allocation, attach controllers before serving, and close partially started models on the inference worker. |
| Serving and memory profile | [profile.py](../omlx/patches/deepseek_v41/flash1017/profile.py) and [server.py](../omlx/server.py) enforce the context and request limits below, use a separate prefix-cache namespace and retain normal scheduler memory admission. Loader headroom checks and process-level MLX memory guidelines are explicit parts of this profile. |
| Standalone fork packaging | The [wheel helper](../scripts/build_flash1017_wheel.py) packages current fork Python/web/data sources with ten hash-pinned public native assets. It checks that their native source/build inputs still match the public base, preserves licenses and generates correct CPython 3.12/arm64 metadata. It does not copy an installed application or depend on laboratory helper paths. |

## Supported boundary and memory behavior

The historical machine was one **M5 Ultra Mac Studio, 80 GPU cores, 256 GiB
unified memory**. The loader requires at least 256 GiB and enough recommended
working-set capacity for its configured limit. Other hardware has not inherited
the historical qualification simply because it passes that capacity check.

- **131,072 tokens total context**, including prompt and generated tokens.
  A smaller explicit context limit remains smaller.
- **One active request**, with further requests queued. This is a single-machine
  profile, not evidence for parallel-request throughput or cluster inference.
- **8,192-token maximum prompt chunks**. The existing memory guard may shrink
  chunks or reject admission. The configured maximum is not a promise that every
  measured or future request uses identical chunk geometry.
- **32 GiB physical-memory reserve** in the MLX limit calculation, an
  **8 GiB MLX allocator-cache guideline**, and a **12 GiB available-memory floor**
  checked around shard loading. These are separate from the 512 MiB Engram row
  cache and the server's KV/prefix caches. They do not cap total process RSS or
  guarantee that every request fits.

The MLX wired/memory/cache guidelines affect the server process. Use the
isolated launch workflow rather than assuming these are per-tensor settings.
Runtime controllers protect their model instances, but GPU completion still
requires the normal worker drain and synchronization before unload. Python
locks alone do not synchronize asynchronous GPU execution.

## Evidence and claims

The archived October 18 prototype measured **20.06% less geometric-mean time
to first token**, 19.554 to 15.631 seconds, against installed oMLX 0.7.1rc1
with native adaptive MTP enabled. The fixed comparison used eight workload
definitions, two process blocks and three configurations: 48 measured requests
and six excluded warmups. It includes configuration and code changes together;
it does not isolate the contribution of every row in the feature map.

Those were warm-runtime measurements with models loaded and OS caches retained,
fresh conversation state, and candidate row-cache history retained within each
declared block. They exclude loading and HTTP transport. The longest timed input
was 64K. One 64K CPython case regressed 2.32%, and sampled peak process footprint
increased from 208.35 to 216.10 GiB. Two blocks on one machine are not a confidence
interval or independent replication.

The original fixed-512 useful-answer qualification **failed** because nine
requests reached EOS early, three per arm. First-token/prefill timing was
qualified separately. The 128-token timing supplement is post hoc, taken from
the original runs; it is not a new 128-token request benchmark. Output streams
differed: **equivalent answer quality and faster equivalent task completion were
not established**. The earlier 40.3% replay/cache prefill result used a different,
already corrected local reference and must not be added to the overall result.

Longer offline state/output checks do not qualify optimized million-token chat.
The separate long-HTTP gate failed after an unexpected newline in its recovery
response; the cause was not established and the remaining long cases were not
run. These observations and the unselected/failed experiments remain part of
the research record.

**None of those measurements is a benchmark of this newly packaged fork.**
Its public-native wheel also differs from the measured application's native
binary identity. Packaging/source checks, controller contract tests, actual
serving checks, numerical comparisons and performance measurements establish
different things and should be recorded separately against the relevant artifact.

## Maintaining the fork

The arithmetic is now in ordinary package modules with source provenance in
[provenance.json](../omlx/patches/deepseek_v41/flash1017/provenance.json).
The runtime does not AST-load scripts from the research directory. The
controller adapters keep policy, CPU caches and ownership attributes outside
the model's parameter tree; installation checks parameter names and tensor
identity. Changes to arithmetic require their own numerical validation rather
than assuming that an ownership test proves equivalent model outputs.

The relevant tests are
[early inference contracts](../tests/test_flash1017_early.py),
[runtime ownership/cache contracts](../tests/test_flash1017_runtime.py),
[prefetch budgets](../tests/test_flash1017_prefetch.py),
[engine-pool integration](../tests/test_flash1017_pool.py),
[profile configuration](../tests/test_flash1017_profile.py) and
[packaging](../tests/test_flash1017_packaging.py). Some use CPU/fake fixtures;
they are not a replacement for testing the installed artifact's GPU and
serving paths. The installation guide gives the standalone packaging-test
command that avoids the broad upstream GPU test setup.

Preserve the explicit opt-in, checkpoint scope, fallback and shutdown contracts
when updating upstream. Rebase/cherry-pick small components with reviewable
diffs. The public-native helper intentionally refuses changes to the native
source subtree, build setup or MLX ABI pin: such a change requires compiling and
qualifying new native assets. Archive new benchmark source/build identities in
the research repository rather than rewriting the existing evidence.

## Credit and upstream candidates

DeepSeek supplies the model and DSpark method; mlx-community supplies the
unchanged quantized checkpoint. Apple MLX, MLX-LM and oMLX supply the inference
foundation. Dependency replay and numerical partition preservation have prior
implementations in [antirez/ds4](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L42408)
and [tacos8me's MLX replay](https://github.com/tacos8me/m5-ultra/blob/4e4d5a81f8135249815a3f284c0ed5bbd1faee58/kernels/omlx/src/omlx/patches/deepseek_v41/encoder_replay.py).
Earlier tacos8me code also contains analogous
[BF16 row reuse](https://github.com/tacos8me/m5-ultra/blob/fb233e22762f11e23123437efd4cbfc400a59f07/kernels/omlx/src/omlx/patches/deepseek_v41/head.py)
and a [frozen confidence/cost policy](https://github.com/tacos8me/m5-ultra/blob/4e4d5a81f8135249815a3f284c0ed5bbd1faee58/kernels/omlx/src/omlx/patches/deepseek_v41/mtp.py).
These are related prior work; similarity alone establishes neither direct
derivation nor independent invention. The causal admission rule follows
[DSpark section 3.2.2](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS2).
[gnahzhang's reference-rounding documentation](https://huggingface.co/gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX/blob/c98bd97baca052f484c1d6ea6b6e8a8eced404ef/README.md)
also informs the compatibility boundary.

HotRows is explicitly **copied verbatim** from tacos8me/m5-ultra at the pinned
source in [NOTICE](../omlx/patches/deepseek_v41/flash1017/NOTICE), with its
[Apache-2.0 license](../omlx/patches/deepseek_v41/flash1017/HOT_ROWS_LICENSE)
retained. Per-file MIT/Apache notices remain applicable. Local contribution
claims concern concrete numerical repairs, integration, ownership/lifecycle
handling and exact-checkpoint validation, not invention of all these techniques.

The likely upstream units are separate reviewable changes, rather than one
large profile PR:

1. **BF16 reuse for rows two through five**, credited to the existing work,
   with production-path tests against the actual target upstream revision.
   Keep row six and its different numerical baseline a separate decision.
2. **Optional reference attention rounding** in both target/draft paths, with
   explicit metadata/configuration behavior and compatibility tests.
3. **Prefetch budget and lifecycle integration** where maintainers want those
   interfaces, with ordinary-default behavior and cleanup/cancellation tests.
4. **Confidence admission and decoder replay** only with their calibration,
   state/numerical boundaries and scoped integration reviewed. The exact
   checkpoint's fitted costs should not silently become all-model defaults.

The research snapshot's narrow upstream packet is a historical draft against
a different upstream revision, not a submitted PR or a current-main benchmark.
Actual branches and PRs belong in this runtime fork. MLX or DeepSeek submissions
would need a concrete library/model issue or reusable change; this integration
alone does not justify parallel PRs to every project. No repository, release,
article or upstream submission is published by preparing these files.

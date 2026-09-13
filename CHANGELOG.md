# Changelog

## [0.12.0](https://github.com/fg-labs/bwa-mem3/compare/v0.11.0...v0.12.0) (2026-09-13)


### Features

* **cli:** add --rescue-skip=false opt-out form ([#472](https://github.com/fg-labs/bwa-mem3/issues/472)) ([ba8422f](https://github.com/fg-labs/bwa-mem3/commit/ba8422f76e9993bcd2e1babd6158c12b90587cbc))
* **compat:** centralize the --compat divergence guard and add --compat-allow-divergent ([#477](https://github.com/fg-labs/bwa-mem3/issues/477)) ([b0be7d3](https://github.com/fg-labs/bwa-mem3/commit/b0be7d3ffd639ae05589c3782d6aacc41c421595))
* **fast:** fold the score-gated chain-extension cap into --fast (--extend-tie-frac 0.95 + --extend-csub) ([#269](https://github.com/fg-labs/bwa-mem3/issues/269)) ([e5bcb2f](https://github.com/fg-labs/bwa-mem3/commit/e5bcb2f6c277d5910a6319324e4e6aa8434930aa))
* **index:** bwamem3 index -u for runtime SA sampling ([#445](https://github.com/fg-labs/bwa-mem3/issues/445)) ([c55f3d9](https://github.com/fg-labs/bwa-mem3/commit/c55f3d966254dbeb62b3ea448df649709e4611da))
* **index:** fmi_seed_api facade for external seeding consumers ([#446](https://github.com/fg-labs/bwa-mem3/issues/446)) ([0e6ed0f](https://github.com/fg-labs/bwa-mem3/commit/0e6ed0fdfca9e91a2aeb0b74401f8790ebb3cf27))


### Bug Fixes

* **bam:** clamp nst_nt4_table index in meth SEQ restore; harden header-filter guard ([#463](https://github.com/fg-labs/bwa-mem3/issues/463)) ([314e66d](https://github.com/fg-labs/bwa-mem3/commit/314e66d8f81f5aad8135f588507bd9ae99aac841))
* **bandedSWA:** add symmetric ref/query length bounds guards to all wrappers ([#467](https://github.com/fg-labs/bwa-mem3/issues/467)) ([17ce8bc](https://github.com/fg-labs/bwa-mem3/commit/17ce8bcb82d10158540408cb1309f536f1a9e393))
* **bandedSWA:** weight the 16-bit z-drop drift by the gap-extend penalty ([#471](https://github.com/fg-labs/bwa-mem3/issues/471)) ([b8e8572](https://github.com/fg-labs/bwa-mem3/commit/b8e85720ea902f895bbb79e7560527d104ea5e06))
* **bandedSWA:** widen the qlen band-clamp reach so it can't overflow ([#468](https://github.com/fg-labs/bwa-mem3/issues/468)) ([c1e8e9e](https://github.com/fg-labs/bwa-mem3/commit/c1e8e9e06e0778add32d450c225da7ceba0c5d9e))
* **bwamem:** eliminate srtg realloc-aliasing bug + harden unchecked reallocs ([#457](https://github.com/fg-labs/bwa-mem3/issues/457)) ([6d4a858](https://github.com/fg-labs/bwa-mem3/commit/6d4a8586dbdbbc5dd3b496013df2d9795345b42a))
* **chain:** default path leaves the read unmapped when the weight filter drops every chain; docs state --compat per target ([#489](https://github.com/fg-labs/bwa-mem3/issues/489)) ([3954d65](https://github.com/fg-labs/bwa-mem3/commit/3954d6572d5f3204f5743ca1e5b7547af8e14d27))
* clamp -t to MAX_THREADS; concurrency and API hardening ([#454](https://github.com/fg-labs/bwa-mem3/issues/454)) ([c651aa5](https://github.com/fg-labs/bwa-mem3/commit/c651aa5d02ac33497c24f191cfbce81cd50d73b2))
* **cli:** memory-safety and input-handling correctness in the CLI/reader paths ([#459](https://github.com/fg-labs/bwa-mem3/issues/459)) ([741ffbe](https://github.com/fg-labs/bwa-mem3/commit/741ffbe9d31d26a9c6f6e6e59f4fa274462325ca))
* fail-closed guards on reader/parser/IO error paths ([#451](https://github.com/fg-labs/bwa-mem3/issues/451)) ([c5dc86b](https://github.com/fg-labs/bwa-mem3/commit/c5dc86bbca815b9512686c978c563dbfb8769cfe))
* **fmi:** correct the SA sentinel offset in the prefetch lookup; fence the uncompressed SA accessors ([#469](https://github.com/fg-labs/bwa-mem3/issues/469)) ([9d2190f](https://github.com/fg-labs/bwa-mem3/commit/9d2190fce2d2466515b9fdbf95caac0fbd722572))
* **fmi:** fail-closed index validation, OOM and env-parse hardening ([#461](https://github.com/fg-labs/bwa-mem3/issues/461)) ([258de49](https://github.com/fg-labs/bwa-mem3/commit/258de4930b7e507fe327f076320ba36c067da72a))
* **index:** don't exit() from inside the emit_0123 OpenMP region ([#462](https://github.com/fg-labs/bwa-mem3/issues/462)) ([268cf78](https://github.com/fg-labs/bwa-mem3/commit/268cf78f0e067aa2cc4d055fe20ce4a9045c47c0))
* **index:** harden OOM and EOF error paths in index build and I/O wrappers ([#466](https://github.com/fg-labs/bwa-mem3/issues/466)) ([6234fb5](https://github.com/fg-labs/bwa-mem3/commit/6234fb52ace64357a8681cf2978c9b97dd743bea))
* **kopen:** harden pipe/URL input opener and reap producer exit status ([#458](https://github.com/fg-labs/bwa-mem3/issues/458)) ([ed61b5e](https://github.com/fg-labs/bwa-mem3/commit/ed61b5ecd5ef9e04d7568a5126881327a0f52b42))
* **mem:** free leaked input readers and read-memo scratch ([#502](https://github.com/fg-labs/bwa-mem3/issues/502)) ([4a93c7e](https://github.com/fg-labs/bwa-mem3/commit/4a93c7ebf51dcd7091f1711aa98daa812e7650c2))
* **nt4:** guard signed-char high-bit bytes; unify the nt4 re-encoding idiom ([#456](https://github.com/fg-labs/bwa-mem3/issues/456)) ([0de026e](https://github.com/fg-labs/bwa-mem3/commit/0de026e8a848444827bfd38d71d112eec63e046b))
* **numeric:** reject and guard divide-by-zero on -E 0 and -I mean,0 ([#460](https://github.com/fg-labs/bwa-mem3/issues/460)) ([c47de17](https://github.com/fg-labs/bwa-mem3/commit/c47de179191e1bac2df2fe5478b1cd86d4c03e6f))
* overflow / negative-size / OOM-false-positive guards ([#453](https://github.com/fg-labs/bwa-mem3/issues/453)) ([167f643](https://github.com/fg-labs/bwa-mem3/commit/167f643c330e72f961fee577ef1cd706cb93ae55))
* **reader:** fail loudly on a truncated gzip stream and guard over-long reads ([#464](https://github.com/fg-labs/bwa-mem3/issues/464)) ([5ea3150](https://github.com/fg-labs/bwa-mem3/commit/5ea3150dd295a50e2490208868df58ae877973ad))


### Performance

* **bam_rec_scratch:** grow scratch buffers geometrically; share the policy ([#455](https://github.com/fg-labs/bwa-mem3/issues/455)) ([de02336](https://github.com/fg-labs/bwa-mem3/commit/de0233633748a4022366c969b775d6ffe103693a))
* **bam:** parallelize --bam BGZF compression, auto-scaled from -t (--bam-threads) ([#474](https://github.com/fg-labs/bwa-mem3/issues/474)) ([378d231](https://github.com/fg-labs/bwa-mem3/commit/378d231bf4c8cca07bcd4b26fb66dd3e0b905d8a))
* **bandedSWA:** cache scoring-matrix LUTs in the constructor ([#476](https://github.com/fg-labs/bwa-mem3/issues/476)) ([6010149](https://github.com/fg-labs/bwa-mem3/commit/60101497ff8ed8ac22df03e0008a243fea8d227f))
* **bandedSWA:** NEON extension kernel throughput (tiled SoA transpose, 16-bit tidy-up, Apple-only epilogue gate) ([#480](https://github.com/fg-labs/bwa-mem3/issues/480)) ([0310628](https://github.com/fg-labs/bwa-mem3/commit/0310628fba7975884671144d1665ea444c5ed192))
* **chain:** prefetch the next chain's packed reference window ([#486](https://github.com/fg-labs/bwa-mem3/issues/486)) ([a8350aa](https://github.com/fg-labs/bwa-mem3/commit/a8350aa81c525ce61b263618a316e36041c4ccdd))
* **dedup:** incremental by-score sort for the mate-rescue dedup calls ([#498](https://github.com/fg-labs/bwa-mem3/issues/498)) ([99034b4](https://github.com/fg-labs/bwa-mem3/commit/99034b41c98a4695c9b5ba8166fa6c9385801818))
* **fmi:** branchless bit-plane test in the compressed-SA LF step (-0.6% CPU on Graviton4) ([#485](https://github.com/fg-labs/bwa-mem3/issues/485)) ([79d309a](https://github.com/fg-labs/bwa-mem3/commit/79d309a5bd71ed34491dfffc2a45c7572028f772))
* **fmi:** count the k-only backward step's occurrence bits in NEON registers; gate backwardExt against a scalar reference ([#482](https://github.com/fg-labs/bwa-mem3/issues/482)) ([33c57ac](https://github.com/fg-labs/bwa-mem3/commit/33c57ac0516bbf9701d05b06da6f540dd12f9482))
* **fmi:** default the compressed-SA resolver to 40 lanes on x86_64 ([#491](https://github.com/fg-labs/bwa-mem3/issues/491)) ([876cdab](https://github.com/fg-labs/bwa-mem3/commit/876cdab38656395f7871dd206180cdc0654b99fb))
* **ksw:** Apple-gated incremental ring indexing in the NEON global kernels ([#484](https://github.com/fg-labs/bwa-mem3/issues/484)) ([792cd9e](https://github.com/fg-labs/bwa-mem3/commit/792cd9ec25ef5e67585a9c1c965f47c27d391648))
* **mem:** look-ahead prefetch of the next read/pair's CIGAR-emission reference window ([#452](https://github.com/fg-labs/bwa-mem3/issues/452)) ([eb0c8c1](https://github.com/fg-labs/bwa-mem3/commit/eb0c8c17d28d2f4fe11178b152a49a279da99c11))
* **meth:** prune 3-letter over-seeding before SA resolution (~30% faster --meth) ([#441](https://github.com/fg-labs/bwa-mem3/issues/441)) ([f3048f0](https://github.com/fg-labs/bwa-mem3/commit/f3048f0b178f0cf0d63c4f6817fbae4dbcf60556))
* **ref:** SIMD 2-bit-&gt;byte reference unpack (fwd+rev, all arches) + prefetch short-ext/mate-rescue fetches ([#448](https://github.com/fg-labs/bwa-mem3/issues/448)) ([6955c8e](https://github.com/fg-labs/bwa-mem3/commit/6955c8e86dcae7f2ea9cd1e89abbe73ae539ad72))
* **rescue:** NEON mate-rescue kernel throughput (lazy query end, pad gather, 16-bit port, tiled SoA, epilogue trims, block checkpointing) ([#479](https://github.com/fg-labs/bwa-mem3/issues/479)) ([38ff9bb](https://github.com/fg-labs/bwa-mem3/commit/38ff9bbba4134b5a4c76779992db4d831177010c))
* **sam_encode:** finish the SSSE3 SEQ/QUAL encoders with one overlapped vector instead of a scalar tail ([#490](https://github.com/fg-labs/bwa-mem3/issues/490)) ([9ed9bc6](https://github.com/fg-labs/bwa-mem3/commit/9ed9bc65ebe2a0708b9794805469bf025c66d801))
* **sam_encode:** NEON SEQ/QUAL encoders finish with one overlapped vector; single-lookup reverse ([#481](https://github.com/fg-labs/bwa-mem3/issues/481)) ([823c9ba](https://github.com/fg-labs/bwa-mem3/commit/823c9ba06c039c2e7630f185d41cebf94f0510e3))
* **seeding:** enable the third-pass bwtseed lockstep on x86 when every thread has its own physical core ([#497](https://github.com/fg-labs/bwa-mem3/issues/497)) ([63eb8a4](https://github.com/fg-labs/bwa-mem3/commit/63eb8a4a6d6617cbcf7391e0a638c62d9c5e85f3))
* **sort:** pdqsort for the pairing and primary-marking sorts, hold-and-shift insertion pass in ks_introsort ([#499](https://github.com/fg-labs/bwa-mem3/issues/499)) ([3b72796](https://github.com/fg-labs/bwa-mem3/commit/3b727965c3b5f4b091d973ffa3507cacbc6bd4da))
* **x86:** build the AVX2 / AVX-512BW 16-bit batch wrappers' SoA with a tiled transpose ([#495](https://github.com/fg-labs/bwa-mem3/issues/495)) ([b4a4e61](https://github.com/fg-labs/bwa-mem3/commit/b4a4e61f95533d021763ed431ea9d1339fa3f086))
* **x86:** build the AVX2 / AVX-512BW 8-bit batch wrappers' SoA with a tiled transpose ([#492](https://github.com/fg-labs/bwa-mem3/issues/492)) ([a609cc7](https://github.com/fg-labs/bwa-mem3/commit/a609cc7e6a621715daab9f5d52efef8fec0addbf))
* **x86:** lower the AVX2 int16 wavefront crossover for global alignment to w &gt;= 10 ([#496](https://github.com/fg-labs/bwa-mem3/issues/496)) ([f51752b](https://github.com/fg-labs/bwa-mem3/commit/f51752b2b256e77c61922deb8924114517f6444b))
* **x86:** skip the AVX2 / AVX-512BW 8-bit extension kernels' wide per-row epilogue when no lane can change ([#494](https://github.com/fg-labs/bwa-mem3/issues/494)) ([a445084](https://github.com/fg-labs/bwa-mem3/commit/a445084f9356d46a1ebb1577ae6316424bc53ccc))


### Refactoring

* remove dead legacy BWT-index API, SMEM iterator, and AFF blocks ([#465](https://github.com/fg-labs/bwa-mem3/issues/465)) ([1dbb072](https://github.com/fg-labs/bwa-mem3/commit/1dbb072710cefefe9381be7fbb48885e62306952))


### Documentation

* correct source-doc comments; add fr_fastq CRLF-in-quality test case ([#450](https://github.com/fg-labs/bwa-mem3/issues/450)) ([8ab82b6](https://github.com/fg-labs/bwa-mem3/commit/8ab82b6c0d41448d4335f35acda308797ea279ae))
* **equivalence:** drop an out-of-scope external cross-reference from the gap-from-M note ([#442](https://github.com/fg-labs/bwa-mem3/issues/442)) ([1a4a8bd](https://github.com/fg-labs/bwa-mem3/commit/1a4a8bd878ed140874ddb52090f31f6be2b87d4e))
* neutralize internal tooling names in source comments ([#449](https://github.com/fg-labs/bwa-mem3/issues/449)) ([426ea8d](https://github.com/fg-labs/bwa-mem3/commit/426ea8dcc8a962a75dcd6c473d784889c387e5ed))
* **readme:** update performance tables for v0.11.0 ([#443](https://github.com/fg-labs/bwa-mem3/issues/443)) ([bd42358](https://github.com/fg-labs/bwa-mem3/commit/bd423584fa114873d6bfa95d3812c19c4a0df649))

## [0.11.0](https://github.com/fg-labs/bwa-mem3/compare/v0.10.0...v0.11.0) (2026-09-01)


### Features

* **bsw:** byte-identical extension-DP job dedup with a self-calibrating on/off controller ([#415](https://github.com/fg-labs/bwa-mem3/issues/415)) ([5bfa754](https://github.com/fg-labs/bwa-mem3/commit/5bfa754afef722ee605937fad9e673496e036d5b))
* **mem:** --dedup-reads whole-read-pair memoization (panel speed lever, byte-identical) ([#433](https://github.com/fg-labs/bwa-mem3/issues/433)) ([a6bf411](https://github.com/fg-labs/bwa-mem3/commit/a6bf411798a44a98d35a444b5040287f8b7870e7))
* **mem:** add --no-adaptive-band to opt out of adaptive banded-SW under --fast ([#417](https://github.com/fg-labs/bwa-mem3/issues/417)) ([ca4f336](https://github.com/fg-labs/bwa-mem3/commit/ca4f3361a6d5c1aa3e9953e63417cbaccf6485b9))


### Bug Fixes

* **bsw:** zero padded idr/idq in the AVX2 8-bit banded-SW wrapper ([#434](https://github.com/fg-labs/bwa-mem3/issues/434)) ([7b7a7eb](https://github.com/fg-labs/bwa-mem3/commit/7b7a7eb4d9de9b421ab25d5781adbf4a6608fa31))
* **bsw:** zero padded len2 in the AVX-512 8-bit banded-SW wrapper ([#439](https://github.com/fg-labs/bwa-mem3/issues/439)) ([ab9159b](https://github.com/fg-labs/bwa-mem3/commit/ab9159bd28ccc870f6bf5d70760e26756834cfc3))
* **bsw:** zero padded-lane h0 in the banded-SW wrapper padding loops ([#435](https://github.com/fg-labs/bwa-mem3/issues/435)) ([082cd90](https://github.com/fg-labs/bwa-mem3/commit/082cd9035fb905ac6cc1fe8a18a063ecfeac602f))
* clamp pwrite_all requests to 1GiB so &gt;2GiB index writes work on macOS ([#419](https://github.com/fg-labs/bwa-mem3/issues/419)) ([7bd35c3](https://github.com/fg-labs/bwa-mem3/commit/7bd35c3fdaf22af298d5870664a07b6b822d4712))
* **extend:** cap the 8-bit banded-SW envelope at w=124 (int8 offset wrap) ([#422](https://github.com/fg-labs/bwa-mem3/issues/422)) ([6def77e](https://github.com/fg-labs/bwa-mem3/commit/6def77e631d4770772f7dfef074041c5184bfa78))
* **extend:** compute the 16-bit banded-SW band clamp in wide arithmetic ([#423](https://github.com/fg-labs/bwa-mem3/issues/423)) ([5ffab89](https://github.com/fg-labs/bwa-mem3/commit/5ffab8939aeed31ed38050612598fab4d51dff9c))
* **extend:** gate the vector banded-SW z-drop on zdrop &gt; 0 ([#424](https://github.com/fg-labs/bwa-mem3/issues/424)) ([90c3e55](https://github.com/fg-labs/bwa-mem3/commit/90c3e5554a750763a8e00dba11a0b7e81ee48b7e))
* **ksw:** free the int16 wavefront scratch on worker-thread exit (LSan) ([#431](https://github.com/fg-labs/bwa-mem3/issues/431)) ([b44cdf3](https://github.com/fg-labs/bwa-mem3/commit/b44cdf3cbffbfb2e66413577cc62318f75ea6a51))
* **kswv:** zero padded-lane idr/idq/h0 in the mate-rescue wrapper padding loops ([#436](https://github.com/fg-labs/bwa-mem3/issues/436)) ([81afa59](https://github.com/fg-labs/bwa-mem3/commit/81afa59c1557ba4eb7319687b8e5465bef8cdd42))
* **rescue:** correct and tighten the u8 mate-rescue admission bound ([#421](https://github.com/fg-labs/bwa-mem3/issues/421)) ([12cb1a5](https://github.com/fg-labs/bwa-mem3/commit/12cb1a5ede80334d2447d437e34610056a95748e))


### Performance

* **bsw:** prefetch next batch's ref/query in the NEON 8-bit wrapper ([#190](https://github.com/fg-labs/bwa-mem3/issues/190)) ([7131a2b](https://github.com/fg-labs/bwa-mem3/commit/7131a2b730829ce794e8f51e34a96fcfb00d0992))
* **bsw:** use SYM 16-bit prepass on NEON, keep byte-LUT on x86 ([#411](https://github.com/fg-labs/bwa-mem3/issues/411)) ([ada4885](https://github.com/fg-labs/bwa-mem3/commit/ada488548bfbcaa2883e5d6fbbea2e0508ea15e5))
* **chain:** reuse the per-read chaining B-tree instead of re-allocating it ([#416](https://github.com/fg-labs/bwa-mem3/issues/416)) ([4b01ff2](https://github.com/fg-labs/bwa-mem3/commit/4b01ff2fa79fdff171c17b5ccf16554bb6b19492))
* **extend:** extend the certified probe rung to the 16-bit tier ([#429](https://github.com/fg-labs/bwa-mem3/issues/429)) ([c5e8ef8](https://github.com/fg-labs/bwa-mem3/commit/c5e8ef82cdeb7710c6c0bdde08f4588a3793108a))
* **extend:** raise ungapped-scan cap to 512 for long-read fast-path (byte-identical) ([#426](https://github.com/fg-labs/bwa-mem3/issues/426)) ([ba7a497](https://github.com/fg-labs/bwa-mem3/commit/ba7a497ee6999d35a1c6653a6af8878ec511b9b2))
* **extend:** route proven-narrow 8-bit extensions through a certified probe rung ([#427](https://github.com/fg-labs/bwa-mem3/issues/427)) ([9461c0d](https://github.com/fg-labs/bwa-mem3/commit/9461c0ddd46fd4aff19896952b0bc615448230de))
* **extend:** widen the certified adaptive-band probe past the fixed 20-cell start ([#428](https://github.com/fg-labs/bwa-mem3/issues/428)) ([672d69b](https://github.com/fg-labs/bwa-mem3/commit/672d69bee57d9527bfc085e6502018e2f66785d3))
* **ksw:** vectorize ksw_global2 with a byte-identical anti-diagonal SIMD kernel ([#418](https://github.com/fg-labs/bwa-mem3/issues/418)) ([d815814](https://github.com/fg-labs/bwa-mem3/commit/d815814233e570db2672c30a0fa873fa9444ad30))
* **mem:** prefetch the CIGAR-emission reference-window cache line ([#425](https://github.com/fg-labs/bwa-mem3/issues/425)) ([cad4b4c](https://github.com/fg-labs/bwa-mem3/commit/cad4b4c7d2d3d35c4c728bb8c4f2ce5192f57d86))
* **seed:** K-only backward extension in the SMEM pass, with micro-specializations ([#432](https://github.com/fg-labs/bwa-mem3/issues/432)) ([ef6c324](https://github.com/fg-labs/bwa-mem3/commit/ef6c32425410c955bd7cfe8b09907bda37f38bc5))
* **seed:** make the phase-2 SMEM lockstep probe opt-in, not default ([#414](https://github.com/fg-labs/bwa-mem3/issues/414)) ([397e1a6](https://github.com/fg-labs/bwa-mem3/commit/397e1a6851dcadcc41d5fa125ea7b7704a160945))
* sound byte-identical adaptive extension band (default on) ([#420](https://github.com/fg-labs/bwa-mem3/issues/420)) ([7733cf5](https://github.com/fg-labs/bwa-mem3/commit/7733cf569d5e3a9be053022091f6b36f334cd152))


### Documentation

* **mem:** document infer_bw as the ungapped-CIGAR fast path; drop stale FIXME ([#430](https://github.com/fg-labs/bwa-mem3/issues/430)) ([fb43e19](https://github.com/fg-labs/bwa-mem3/commit/fb43e19aa2ed855b4fb023e3d107f6247915f9e3))
* **readme:** add performance section with current-release + full history ([#410](https://github.com/fg-labs/bwa-mem3/issues/410)) ([d1e9b43](https://github.com/fg-labs/bwa-mem3/commit/d1e9b43bc6d80bf017034e4667540cf3ba1ba479))

## [0.10.0](https://github.com/fg-labs/bwa-mem3/compare/v0.9.0...v0.10.0) (2026-08-21)


### Features

* **cli:** add --hic as an alias for -5SP ([#372](https://github.com/fg-labs/bwa-mem3/issues/372)) ([d00d97d](https://github.com/fg-labs/bwa-mem3/commit/d00d97dbeec4646862f97ebcfb125e678a40ea5f)), closes [#368](https://github.com/fg-labs/bwa-mem3/issues/368)
* **mem:** --huge-pages to back the index with 1 GB pages when available ([#405](https://github.com/fg-labs/bwa-mem3/issues/405)) ([371a181](https://github.com/fg-labs/bwa-mem3/commit/371a1819802c2962b768c3b165f0d1319a6a75b3))


### Bug Fixes

* **chain:** clamp mem_chain_weight to the width of mem_chain_t::w ([#376](https://github.com/fg-labs/bwa-mem3/issues/376)) ([ad3045f](https://github.com/fg-labs/bwa-mem3/commit/ad3045fe9b371dd8523451ccf9a5083e471aac13)), closes [#309](https://github.com/fg-labs/bwa-mem3/issues/309)
* **compat:** model the all-chains-dropped divergence per --compat target ([#374](https://github.com/fg-labs/bwa-mem3/issues/374)) ([a73ada0](https://github.com/fg-labs/bwa-mem3/commit/a73ada0318ff2f19146183ad4fedd0a96b0ab384)), closes [#310](https://github.com/fg-labs/bwa-mem3/issues/310)
* **kvec:** abort on realloc failure instead of leaking + NULL-deref ([#398](https://github.com/fg-labs/bwa-mem3/issues/398)) ([d199db1](https://github.com/fg-labs/bwa-mem3/commit/d199db15b0e1e15e15dcab67e7e69fb2ba13be3f))
* **reader:** prevent a paired-end out-of-bounds write in bseq_read_fast ([#395](https://github.com/fg-labs/bwa-mem3/issues/395)) ([6fb093c](https://github.com/fg-labs/bwa-mem3/commit/6fb093c98824ce95483dc292a1bebd6e74101d0f))


### Performance

* **bandedSWA:** unmasked fast-regime for the fully-in-band extension columns ([#408](https://github.com/fg-labs/bwa-mem3/issues/408)) ([70c3baa](https://github.com/fg-labs/bwa-mem3/commit/70c3baa78b19f79a17c33c2e73ff160a7f7ed7da))
* **bsw,kswv:** drop sse2neon translation overhead in the NEON banded-SW kernels ([#378](https://github.com/fg-labs/bwa-mem3/issues/378)) ([aecc30d](https://github.com/fg-labs/bwa-mem3/commit/aecc30d166a63e711d710f6b24186a74f9e35501))
* **bsw:** drop a redundant per-cell argmax compare the AVX2 twin already dropped ([#379](https://github.com/fg-labs/bwa-mem3/issues/379)) ([0c68c84](https://github.com/fg-labs/bwa-mem3/commit/0c68c84ac8b038a816a4252c1378e05f49ac0cb5))
* **bsw:** skip sse2neon's index-mask in the NEON score-LUT gathers ([#381](https://github.com/fg-labs/bwa-mem3/issues/381)) ([9f7514c](https://github.com/fg-labs/bwa-mem3/commit/9f7514c4dced93db91b3f68165be674b61200a36))
* **classa:** stacked byte-identical local-opts (F3,L3,L7,L8,L17,L27) ([#387](https://github.com/fg-labs/bwa-mem3/issues/387)) ([f8bc444](https://github.com/fg-labs/bwa-mem3/commit/f8bc444d52169b125a9f65fd61b3378e8fe63e5a))
* **dedup:** sort a (key,index) permutation, dropping the per-call save-copy ([#399](https://github.com/fg-labs/bwa-mem3/issues/399)) ([937c2fa](https://github.com/fg-labs/bwa-mem3/commit/937c2fac9a5996fb1a48f7cf54353527c4353957))
* **kswv:** use andnot for the AVX2 16-bit rescue boundary-zero ([#401](https://github.com/fg-labs/bwa-mem3/issues/401)) ([b798992](https://github.com/fg-labs/bwa-mem3/commit/b7989929e1b8ab28e8601a1fa354aeb951042ba7))
* **rescue:** fuse the u8 mate-rescue diagonal with USQADD ([#406](https://github.com/fg-labs/bwa-mem3/issues/406)) ([1d2e20f](https://github.com/fg-labs/bwa-mem3/commit/1d2e20fb8eb6618c8921f9bcfa6dd02ee5c5311c))
* **rescue:** process the u8 mate-rescue kernel two target rows per pass ([#407](https://github.com/fg-labs/bwa-mem3/issues/407)) ([0020016](https://github.com/fg-labs/bwa-mem3/commit/0020016cf8ebe06605b6b576e9e00fe6703def05))
* **seed:** auto-tune the phase-2 SMEM lockstep width at startup ([#393](https://github.com/fg-labs/bwa-mem3/issues/393)) ([6a7f146](https://github.com/fg-labs/bwa-mem3/commit/6a7f1466864a49e84f9638f72851e507eddd81a3))


### Refactoring

* **simd:** make simd_compat.h explicitly ARM-only; drop dead x86 branches ([#394](https://github.com/fg-labs/bwa-mem3/issues/394)) ([8cc29e3](https://github.com/fg-labs/bwa-mem3/commit/8cc29e3b73125679e9d4bc19c2c1870e6b203795))


### Documentation

* **bwamem:** document that the no-extension seedcov recompute is load-bearing ([#386](https://github.com/fg-labs/bwa-mem3/issues/386)) ([c0526e5](https://github.com/fg-labs/bwa-mem3/commit/c0526e55c86b46dda8fc231ec9cab9208f356111))
* document 1 GB huge pages for the index as a Linux deployment lever ([#404](https://github.com/fg-labs/bwa-mem3/issues/404)) ([017925a](https://github.com/fg-labs/bwa-mem3/commit/017925a2381a1b4eede5f9400a963b127c32a7d8)), closes [#377](https://github.com/fg-labs/bwa-mem3/issues/377)
* document plain, --compat, and --fast alignment modes ([#392](https://github.com/fg-labs/bwa-mem3/issues/392)) ([a6c3c86](https://github.com/fg-labs/bwa-mem3/commit/a6c3c861ed1cdb3d950ca04e58d8b210880fe52a))
* note that -I leaves non-FR orientations without a distribution ([#370](https://github.com/fg-labs/bwa-mem3/issues/370)) ([236b270](https://github.com/fg-labs/bwa-mem3/commit/236b27099c8f7596a12aaae3de8e11a6797a8f33)), closes [#369](https://github.com/fg-labs/bwa-mem3/issues/369)
* retire drifted PR catalog for a verified upstream-disposition record ([#400](https://github.com/fg-labs/bwa-mem3/issues/400)) ([593ca59](https://github.com/fg-labs/bwa-mem3/commit/593ca5959ed9a679666dd7dd34db8c7a2665e606))

## [0.9.0](https://github.com/fg-labs/bwa-mem3/compare/v0.8.0...v0.9.0) (2026-08-06)


### Features

* **mem:** make --compat=bwa-mem selectable ([#360](https://github.com/fg-labs/bwa-mem3/issues/360)) ([22d604d](https://github.com/fg-labs/bwa-mem3/commit/22d604d98a9d55d982dbb5e121bb96c9ef3f61b6))


### Bug Fixes

* **bam:** propagate bam_aux_append failures instead of dropping tags ([#367](https://github.com/fg-labs/bwa-mem3/issues/367)) ([9e7c600](https://github.com/fg-labs/bwa-mem3/commit/9e7c6001b866ec3f044c8f54a5f63d0c6af28c60))
* **bam:** render pa:f from the same definition as the SAM writer ([#366](https://github.com/fg-labs/bwa-mem3/issues/366)) ([fd7e3fe](https://github.com/fg-labs/bwa-mem3/commit/fd7e3fe04a5345d68bd60d73095602f3493c8848)), closes [#365](https://github.com/fg-labs/bwa-mem3/issues/365)
* **pair:** derive FLAG 0x2 from a[0] by default, matching both upstreams ([#363](https://github.com/fg-labs/bwa-mem3/issues/363)) ([a9a96cc](https://github.com/fg-labs/bwa-mem3/commit/a9a96ccfaf4cd7e6f74aa6a510c4ecf97ab3013e))


### Documentation

* **changelog:** backtick SAM header tags to stop @-mentioning users ([#359](https://github.com/fg-labs/bwa-mem3/issues/359)) ([769baec](https://github.com/fg-labs/bwa-mem3/commit/769baeccd5341eb2b30f3f83be5c22b4186bf023))
* **readme:** state bwa-mem2 parity and surface --compat ([#364](https://github.com/fg-labs/bwa-mem3/issues/364)) ([668b21e](https://github.com/fg-labs/bwa-mem3/commit/668b21ecfa2645f9a37eba63dd6a9f7e093794dd))

## [0.8.0](https://github.com/fg-labs/bwa-mem3/compare/v0.7.0...v0.8.0) (2026-08-03)


### ⚠ BREAKING CHANGES

* **meth:** `bwa-mem3 mem --meth` now writes SAM text instead of BAM. Add `--bam` to any script that depended on the old behavior. Pipelines that pipe into samtools need no change — samtools autodetects SAM text.

### Features

* **bam:** warn that in-process compressed BAM is single-threaded ([#287](https://github.com/fg-labs/bwa-mem3/issues/287)) ([9cd414d](https://github.com/fg-labs/bwa-mem3/commit/9cd414dcfd4f98b75f1fb71f570a769143455923))
* **mem:** --compat target enum with record + header byte-identity to bwa-mem2 ([#277](https://github.com/fg-labs/bwa-mem3/issues/277)) ([265b427](https://github.com/fg-labs/bwa-mem3/commit/265b42769731b09d8cca1221148c5f2be98faeb6))
* **mem:** band mate rescue to a k-mer anchor diagonal (--rescue-kmer) ([#335](https://github.com/fg-labs/bwa-mem3/issues/335)) ([6ce2b70](https://github.com/fg-labs/bwa-mem3/commit/6ce2b70509446c874fb7f6bd037bd6553e0a1e23))
* **meth:** add --meth-tags to select which Bismark tags are emitted ([#333](https://github.com/fg-labs/bwa-mem3/issues/333)) ([3ec4500](https://github.com/fg-labs/bwa-mem3/commit/3ec4500fc3e26c9e3140db1c0d4aa81fd84d1f8d)), closes [#331](https://github.com/fg-labs/bwa-mem3/issues/331)
* **meth:** let --bam choose the output container under --meth ([#341](https://github.com/fg-labs/bwa-mem3/issues/341)) ([ed4400c](https://github.com/fg-labs/bwa-mem3/commit/ed4400c5f09302160366403789eddbcf7d41a5f5))


### Bug Fixes

* **bntseq:** don't poison the pac-fetch buffer in release builds ([#263](https://github.com/fg-labs/bwa-mem3/issues/263)) ([1252f1b](https://github.com/fg-labs/bwa-mem3/commit/1252f1bf4245fa4f2245c64157109e6cc17bd88e))
* **bsw:** compute the 8-bit band clamp in wide arithmetic ([#270](https://github.com/fg-labs/bwa-mem3/issues/270)) ([00ade6d](https://github.com/fg-labs/bwa-mem3/commit/00ade6d9930bf4d5bf0609dd3480b58b4fee7509))
* **bsw:** correct 8-bit banded-SW z-drop and seed clamp at high seed scores (all tiers) ([#273](https://github.com/fg-labs/bwa-mem3/issues/273)) ([e722ed0](https://github.com/fg-labs/bwa-mem3/commit/e722ed06dd71d0186c401ce2f0e163d32b0a7ad7))
* **build:** generate header dependencies instead of a hand-maintained list ([#299](https://github.com/fg-labs/bwa-mem3/issues/299)) ([414ac7e](https://github.com/fg-labs/bwa-mem3/commit/414ac7e614db17a3564ae5f9cfdab4f5e6d54ad4))
* **chain:** keep sizeof(mem_chain_t) at 48 B to preserve bwa-mem2 chaining parity ([#268](https://github.com/fg-labs/bwa-mem3/issues/268)) ([a9d9445](https://github.com/fg-labs/bwa-mem3/commit/a9d9445cb984115887184e5bdb0a5ddc41e74708))
* check debug_macro_flag_lint's directory argument before cd ([#352](https://github.com/fg-labs/bwa-mem3/issues/352)) ([14d8096](https://github.com/fg-labs/bwa-mem3/commit/14d8096fec6e657e6008044267354f57c42c3fea))
* **coderabbit:** list main in auto_review.base_branches ([#358](https://github.com/fg-labs/bwa-mem3/issues/358)) ([63add93](https://github.com/fg-labs/bwa-mem3/commit/63add93a2a2cc2f02f216ff4ddbf1cb07ede634d))
* **header:** emit one default `@HD` on every output path ([#291](https://github.com/fg-labs/bwa-mem3/issues/291)) ([56889e1](https://github.com/fg-labs/bwa-mem3/commit/56889e1f5733b72f333fa12faf0ea82157f92d3f)), closes [#288](https://github.com/fg-labs/bwa-mem3/issues/288)
* **index:** cap each pread() at 1GiB so large indexes load on macOS ([#259](https://github.com/fg-labs/bwa-mem3/issues/259)) ([1db6697](https://github.com/fg-labs/bwa-mem3/commit/1db6697cb5456a5905c1e7ce736f4f4d63431334))
* **index:** resolve the auto memory budget from the host, not a 32 GiB cap ([#300](https://github.com/fg-labs/bwa-mem3/issues/300)) ([a8ac368](https://github.com/fg-labs/bwa-mem3/commit/a8ac3683c521bd7f388bb201f75b9a23af1280d4))
* **kswv:** zero query padding in the NEON and AVX-512BW 8-bit mate-rescue kernels ([#290](https://github.com/fg-labs/bwa-mem3/issues/290)) ([b8d5aa3](https://github.com/fg-labs/bwa-mem3/commit/b8d5aa38856d7149bcdf9161a1d5ca38ec5a5bdc))
* **mem:** guard allocations with xassert so OOM checks survive NDEBUG ([#312](https://github.com/fg-labs/bwa-mem3/issues/312)) ([8898782](https://github.com/fg-labs/bwa-mem3/commit/88987823179e8aa179a53d15b8743fa17f22b947))
* **mem:** make the batch-size cap opt-in so default batching matches bwa-mem2 ([#298](https://github.com/fg-labs/bwa-mem3/issues/298)) ([5a85305](https://github.com/fg-labs/bwa-mem3/commit/5a85305fee874ba16aa3e47d10eedbf4e7ea248e))
* **meth:** derive NM/MD from the scoring matrix so conversions are not counted ([#332](https://github.com/fg-labs/bwa-mem3/issues/332)) ([6aa4675](https://github.com/fg-labs/bwa-mem3/commit/6aa4675636673d94fdbbe768671e4ff56e4cad07)), closes [#327](https://github.com/fg-labs/bwa-mem3/issues/327)
* **meth:** diagnose orphaned --meth/--meth-tags values instead of aligning to them ([#334](https://github.com/fg-labs/bwa-mem3/issues/334)) ([a18b19b](https://github.com/fg-labs/bwa-mem3/commit/a18b19bf48a45d1bfb303645b04e73f458365743)), closes [#331](https://github.com/fg-labs/bwa-mem3/issues/331)
* **meth:** emit MQ:i and HN:i from the --meth BAM writer ([#304](https://github.com/fg-labs/bwa-mem3/issues/304)) ([70d1270](https://github.com/fg-labs/bwa-mem3/commit/70d1270666b3d50d5a51c94b795a7570cd28531a)), closes [#296](https://github.com/fg-labs/bwa-mem3/issues/296)
* **meth:** reject mate rescues whose SW alignment runs past the read ([#258](https://github.com/fg-labs/bwa-mem3/issues/258)) ([6ecc61b](https://github.com/fg-labs/bwa-mem3/commit/6ecc61b30a3a38c225cd7641de4ae7ed372e25fa))
* **sam:** make the SAM-A9 rid check opt-in so the recompute is skipped ([#330](https://github.com/fg-labs/bwa-mem3/issues/330)) ([87f3bd6](https://github.com/fg-labs/bwa-mem3/commit/87f3bd6b56eac5b01b97e496fd5730106f52a6b0))
* **simd:** map _MM_HINT_T0 to L1, not L3, on arm64 ([#262](https://github.com/fg-labs/bwa-mem3/issues/262)) ([34cb389](https://github.com/fg-labs/bwa-mem3/commit/34cb3891d5afd773279900b8428d0b5bf956a921))


### Performance

* **bntseq:** O(1) contig bucket table for bns_pos2rid ([#275](https://github.com/fg-labs/bwa-mem3/issues/275)) ([e0291ec](https://github.com/fg-labs/bwa-mem3/commit/e0291ec3a352e79b3dcd16a254f4ea6ce451313c))
* **bntseq:** unpack the 2-bit reference with a byte-&gt;4-base LUT ([#274](https://github.com/fg-labs/bwa-mem3/issues/274)) ([a9b570f](https://github.com/fg-labs/bwa-mem3/commit/a9b570fb302b5b365612164b85ddad6339b1b28c))
* **bsw:** admit high-h0 pairs to the 8-bit banded-SW tier ([#321](https://github.com/fg-labs/bwa-mem3/issues/321)) ([35f3238](https://github.com/fg-labs/bwa-mem3/commit/35f3238c1f4d52d2b0e94fc037db68a22d6314f6))
* **bsw:** drop a redundant compare in the 8-bit row argmax ([#266](https://github.com/fg-labs/bwa-mem3/issues/266)) ([ed34d5d](https://github.com/fg-labs/bwa-mem3/commit/ed34d5d9787313aab2ed012dcce1a8ec04870f76))
* **bsw:** fuse the SBT pre-pass into the 8-bit DP loop with two LUTs ([#280](https://github.com/fg-labs/bwa-mem3/issues/280)) ([94cbdf2](https://github.com/fg-labs/bwa-mem3/commit/94cbdf271a08c70ebf20abe645b412df33bc8ff5))
* **bsw:** pack the int8 lane groups by max(len1,len2) ([#283](https://github.com/fg-labs/bwa-mem3/issues/283)) ([7898254](https://github.com/fg-labs/bwa-mem3/commit/78982548c6b306daaea945a9333a70390ea34fb2))
* **chain:** hoist chain bounds and memoize the per-chain log() ([#265](https://github.com/fg-labs/bwa-mem3/issues/265)) ([7703634](https://github.com/fg-labs/bwa-mem3/commit/7703634884c5cc7c70f1479b121c2782ce527932))
* **chain:** skip side-effect-free comparisons in mem_chain_flt ([#326](https://github.com/fg-labs/bwa-mem3/issues/326)) ([6ca3c7e](https://github.com/fg-labs/bwa-mem3/commit/6ca3c7ef6bea30f09f927867bcf145c8cd20b9de))
* **io:** pool per-read string fields in a per-chunk arena ([#293](https://github.com/fg-labs/bwa-mem3/issues/293)) ([c63eba0](https://github.com/fg-labs/bwa-mem3/commit/c63eba080791c864e95000aa979b9733cfb7ec02))
* **kswv:** recover the query end after the row, via block checkpoints ([#328](https://github.com/fg-labs/bwa-mem3/issues/328)) ([f099e79](https://github.com/fg-labs/bwa-mem3/commit/f099e79d1c4e0cf2f06b00fe23ee60f013f20ba2))
* **kswv:** stop testing for query padding on columns that cannot have any ([#324](https://github.com/fg-labs/bwa-mem3/issues/324)) ([90be8df](https://github.com/fg-labs/bwa-mem3/commit/90be8df08bb2823b95dfa27897da92c08c17e7d9))
* **main:** calibrate proc_freq without sleeping a full second ([#295](https://github.com/fg-labs/bwa-mem3/issues/295)) ([052a84a](https://github.com/fg-labs/bwa-mem3/commit/052a84a20f229a915592f975181de9eaf4bf7ff2))
* **mem:** fuse the seed and extend kt_for passes ([#264](https://github.com/fg-labs/bwa-mem3/issues/264)) ([d4127e9](https://github.com/fg-labs/bwa-mem3/commit/d4127e9cb7a02b8b033402d622954d183c71d656))
* **mem:** read a pestat cohort in slices so compute starts sooner ([#305](https://github.com/fg-labs/bwa-mem3/issues/305)) ([212ed34](https://github.com/fg-labs/bwa-mem3/commit/212ed34d3a0254db78e7a7db3c029582f53e65d7))
* **mem:** reserve the cohort accumulator from task_size instead of doubling ([#342](https://github.com/fg-labs/bwa-mem3/issues/342)) ([80e40ad](https://github.com/fg-labs/bwa-mem3/commit/80e40ad351c41a1a23e4c7a8278cdee809e57b56))
* **mem:** size chaining scratch per thread instead of per read ([#297](https://github.com/fg-labs/bwa-mem3/issues/297)) ([3d66be4](https://github.com/fg-labs/bwa-mem3/commit/3d66be4da53acea114cf45de25c347c798e70f6a))
* **mem:** size the cohort ramp so it cannot outrun the reader ([#315](https://github.com/fg-labs/bwa-mem3/issues/315)) ([160adae](https://github.com/fg-labs/bwa-mem3/commit/160adae9b418adcf320247a016bb9caea3a186f0))
* **mem:** size the per-chunk read pool from the actual read count ([#279](https://github.com/fg-labs/bwa-mem3/issues/279)) ([3848339](https://github.com/fg-labs/bwa-mem3/commit/3848339b55706bcece1fc18ccbc884ed27651448))
* **meth:** share the generic writer's per-record scratch with the meth writer ([#306](https://github.com/fg-labs/bwa-mem3/issues/306)) ([8fade15](https://github.com/fg-labs/bwa-mem3/commit/8fade150e18195fd863d7bdda9b07a07684b5607))
* **rescue:** drop redundant mate-query copies in batched mate-SW ([#284](https://github.com/fg-labs/bwa-mem3/issues/284)) ([32f17e5](https://github.com/fg-labs/bwa-mem3/commit/32f17e5809fa744a0d74d20d24365d45b893d8bb))
* **rescue:** single-position anchor index, plus an opt-in --rescue-skip gate ([#349](https://github.com/fg-labs/bwa-mem3/issues/349)) ([e7e00af](https://github.com/fg-labs/bwa-mem3/commit/e7e00af4471a6f400a34346cf868016223f5e8a8))
* **sam,seed:** micro-cleanups in SAM formatting and seeding ([#286](https://github.com/fg-labs/bwa-mem3/issues/286)) ([82257ab](https://github.com/fg-labs/bwa-mem3/commit/82257abcf33a6056c49490e444165d48ee803973))
* **seed:** correct the FM-index checkpoint prefetch targets ([#267](https://github.com/fg-labs/bwa-mem3/issues/267)) ([6455cad](https://github.com/fg-labs/bwa-mem3/commit/6455cadea07e60859740b3174ebb79fbfd397ed2))
* **seed:** reuse sortSMEMs counting-sort scratch across batches ([#278](https://github.com/fg-labs/bwa-mem3/issues/278)) ([394f8f8](https://github.com/fg-labs/bwa-mem3/commit/394f8f8110f7d15be7ef2ca38c335590aa1e0284))
* **sort:** move the pdqsort/total-order dedup sort behind --fast ([#257](https://github.com/fg-labs/bwa-mem3/issues/257)) ([34b3446](https://github.com/fg-labs/bwa-mem3/commit/34b34466bce9412842d7eed1da3c772d38bad87d))
* **sort:** run pdqsort in the dedup sorts when the comparator sees no tie ([#261](https://github.com/fg-labs/bwa-mem3/issues/261)) ([7481c4f](https://github.com/fg-labs/bwa-mem3/commit/7481c4ffa7ac9a1a55b249fd0f7269b5e4c38f9b))


### Refactoring

* **bsw:** derive extension gaps from M again (bwa-mem2 compatibility) ([#256](https://github.com/fg-labs/bwa-mem3/issues/256)) ([771376e](https://github.com/fg-labs/bwa-mem3/commit/771376e3839275910704972d9a1a1915766b1eec))
* **bsw:** drop the unreachable 8-bit re-baselining in the AVX2/AVX-512 kernels ([#272](https://github.com/fg-labs/bwa-mem3/issues/272)) ([f1a0da6](https://github.com/fg-labs/bwa-mem3/commit/f1a0da65bab8f1cdca2192b9e56671767df0cabd))
* **bsw:** drop the unreachable 8-bit score re-baselining ([#271](https://github.com/fg-labs/bwa-mem3/issues/271)) ([ecebf2b](https://github.com/fg-labs/bwa-mem3/commit/ecebf2b34fb26bdddd212007bf58b25cf893032e))
* **chain:** fuse mem_chain_weight's two seed sweeps into one ([#322](https://github.com/fg-labs/bwa-mem3/issues/322)) ([500c9cb](https://github.com/fg-labs/bwa-mem3/commit/500c9cbcb3a62c023673c58fdc5cc985619dc4f8))
* **chain:** remove dead work from the chaining path ([#285](https://github.com/fg-labs/bwa-mem3/issues/285)) ([8de7e5b](https://github.com/fg-labs/bwa-mem3/commit/8de7e5bb91229b7b6efb9684d82f0cb6d22f4c4f))
* **chain:** tidy mem_chain_flt's filter loops ([#323](https://github.com/fg-labs/bwa-mem3/issues/323)) ([095b401](https://github.com/fg-labs/bwa-mem3/commit/095b401be04aa65348b5f65da7438f0347e3edc1))
* **header:** one generated-`@SQ` builder, one scanner, one record iterator ([#292](https://github.com/fg-labs/bwa-mem3/issues/292)) ([37e5e4a](https://github.com/fg-labs/bwa-mem3/commit/37e5e4a6a18c66ed3e4be5e818a4b57f5dafff86))
* **meth:** name the native-regen decision and document the hypothesis invariant ([#337](https://github.com/fg-labs/bwa-mem3/issues/337)) ([97eb3be](https://github.com/fg-labs/bwa-mem3/commit/97eb3be3a92c3f5963b41e1ad0cfef6a5aeaa55c))


### Documentation

* correct the developer guide's account of the source-only lints ([#354](https://github.com/fg-labs/bwa-mem3/issues/354)) ([a8825f5](https://github.com/fg-labs/bwa-mem3/commit/a8825f52a59a8abe4eed9673c1c3e77a33c046ff))
* **equivalence:** reflect restored bwa-mem2 byte-identity on the drop-in profile ([#276](https://github.com/fg-labs/bwa-mem3/issues/276)) ([8dbf218](https://github.com/fg-labs/bwa-mem3/commit/8dbf218adeeb1215e04619d921bbec18e0746d6f))
* **kswv:** the phase-1 boundary-mask cost is measured now, not unmeasurable ([#308](https://github.com/fg-labs/bwa-mem3/issues/308)) ([136cc9f](https://github.com/fg-labs/bwa-mem3/commit/136cc9ffac6ce857933b9bd7850549402e63de47))
* **memory:** measure the -t batch multiplier cost and correct the resident index figure ([#319](https://github.com/fg-labs/bwa-mem3/issues/319)) ([804f20b](https://github.com/fg-labs/bwa-mem3/commit/804f20bd41b806af4873325583681a0bc1bad1dc))
* **test:** describe the source-only lints' input contract ([#339](https://github.com/fg-labs/bwa-mem3/issues/339)) ([de77830](https://github.com/fg-labs/bwa-mem3/commit/de77830153e019b829d8f33ccd609e55703e149e))
* **test:** replace the dead phiX example for running a regression test ([#345](https://github.com/fg-labs/bwa-mem3/issues/345)) ([d3d9a53](https://github.com/fg-labs/bwa-mem3/commit/d3d9a5322123fc45320e017ae52650af19a94cf2))

## [0.7.0](https://github.com/fg-labs/bwa-mem3/compare/v0.6.0...v0.7.0) (2026-07-23)


### Features

* **bsw:** add BWAMEM3_DUMP_PAIRS to observe extension pairs ([#239](https://github.com/fg-labs/bwa-mem3/issues/239)) ([c4d7480](https://github.com/fg-labs/bwa-mem3/commit/c4d748084e4d844e1448f3c7f7b0c5039f9d8406))
* **meth:** add --meth=taps for TAPS methylation chemistry ([#228](https://github.com/fg-labs/bwa-mem3/issues/228)) ([0b8163d](https://github.com/fg-labs/bwa-mem3/commit/0b8163d2632bd168d2848617858176ac9ce5c127))
* **meth:** drop cross-chemistry seeds by read number ([#225](https://github.com/fg-labs/bwa-mem3/issues/225)) ([6e21a37](https://github.com/fg-labs/bwa-mem3/commit/6e21a37e40e0013916863673622b01fd694dbf58))
* **meth:** NEUTRAL --meth-scoring (TAPS default) + batched kswv rescue ([#243](https://github.com/fg-labs/bwa-mem3/issues/243)) ([f36dc5a](https://github.com/fg-labs/bwa-mem3/commit/f36dc5a05e820985ed3f9f802f5a88840c3ea978))


### Bug Fixes

* **ext:** tighten ungapped fast-path mismatch bound to the strict form ([#231](https://github.com/fg-labs/bwa-mem3/issues/231)) ([5c53540](https://github.com/fg-labs/bwa-mem3/commit/5c5354042b2036882fa8840b3ad62a78f7eed853))
* **fast:** enable --extend-mate-concordant and raise chain cap to 20 for non-meth --fast ([#218](https://github.com/fg-labs/bwa-mem3/issues/218)) ([50a3293](https://github.com/fg-labs/bwa-mem3/commit/50a329390753e77640449eafa564ecf13ade6ec3))
* **mem:** guard extension-buffer growth against int32 offset overflow ([#237](https://github.com/fg-labs/bwa-mem3/issues/237)) ([6acd6b4](https://github.com/fg-labs/bwa-mem3/commit/6acd6b43d4d1712b71f9244f85addc3ca8cd920b))
* **mem:** size sa_coord realloc in elements, not SMEM count ([#235](https://github.com/fg-labs/bwa-mem3/issues/235)) ([7fa501c](https://github.com/fg-labs/bwa-mem3/commit/7fa501c1a6e67d419b1a216f5c8b282385d518c6))
* **meth:** recover repeat placement via both-hypothesis mate rescue + PE MAPQ hardening ([#219](https://github.com/fg-labs/bwa-mem3/issues/219)) ([6705bbf](https://github.com/fg-labs/bwa-mem3/commit/6705bbfa89c2ab40f4fb81f2f6a6a9518c1cf35c))
* **meth:** scale --meth scoring defaults with the match score ([#227](https://github.com/fg-labs/bwa-mem3/issues/227)) ([9811767](https://github.com/fg-labs/bwa-mem3/commit/98117672a6bd0780a9a39ef8b4e50b4f1a6a539e))


### Performance

* **bandedSWA:** fewer ops in the AVX2/AVX-512 extend cores (x86-only) ([#255](https://github.com/fg-labs/bwa-mem3/issues/255)) ([d52842f](https://github.com/fg-labs/bwa-mem3/commit/d52842feef5b8c7cb6137e552c02c2234ddd206c))
* **bsw:** 128-bit (SSE/NEON) 16-bit prepass via byte-LUT + sign-extend ([#223](https://github.com/fg-labs/bwa-mem3/issues/223)) ([239a334](https://github.com/fg-labs/bwa-mem3/commit/239a334b0d6dcbc60ee4e76076353ea79e342479))
* **bsw:** AVX-512 16-bit prepass via 32-entry permutexvar LUT ([#222](https://github.com/fg-labs/bwa-mem3/issues/222)) ([0e8c0b6](https://github.com/fg-labs/bwa-mem3/commit/0e8c0b6674229243c26196e421caeb48580f2c23))
* **index:** parallel pread of the FM-index for faster startup ([#248](https://github.com/fg-labs/bwa-mem3/issues/248)) ([8d88fe5](https://github.com/fg-labs/bwa-mem3/commit/8d88fe5700be4bbbb757976b51a03bcd7e8b2032))
* **kswv:** fold the freed-cell pair into a per-row active_frread on the u16 kernels ([#254](https://github.com/fg-labs/bwa-mem3/issues/254)) ([eaf649f](https://github.com/fg-labs/bwa-mem3/commit/eaf649fcc5943d91f9d70bd4e215a6d489871bc5))
* **kswv:** hoist the freed-cell target into a per-row active_frread vector ([#252](https://github.com/fg-labs/bwa-mem3/issues/252)) ([fc0eb91](https://github.com/fg-labs/bwa-mem3/commit/fc0eb9132179ccac9d79ffe6c480951160582e60))
* **kswv:** hoist the u8 boundary test per-row on NEON and AVX-512 non-meth ([#253](https://github.com/fg-labs/bwa-mem3/issues/253)) ([a25e3b7](https://github.com/fg-labs/bwa-mem3/commit/a25e3b7bf7b3444cc0eadccce6243cd23279e076))
* **kswv:** skip the ref-boundary vbsl on all-real rows in 8-bit mate rescue (+7.5%) ([#244](https://github.com/fg-labs/bwa-mem3/issues/244)) ([78420a5](https://github.com/fg-labs/bwa-mem3/commit/78420a5553b85a7a3fbb98f61017354c8bf7e20a))
* **mem:** resolve SA lookups across reads, not per read ([#236](https://github.com/fg-labs/bwa-mem3/issues/236)) ([79a2425](https://github.com/fg-labs/bwa-mem3/commit/79a24254168c4cbef086dcfeb073ecbd5632e438))
* **meth:** score mate rescue with one read#-derived matrix, not both ([#224](https://github.com/fg-labs/bwa-mem3/issues/224)) ([1baa39b](https://github.com/fg-labs/bwa-mem3/commit/1baa39b17c69584263694fda5ddb666bcc65ad71))
* **meth:** vectorize both-hypothesis mate rescue on the batched kswv path ([#221](https://github.com/fg-labs/bwa-mem3/issues/221)) ([46aeaae](https://github.com/fg-labs/bwa-mem3/commit/46aeaaeb818ae8e5272f29ba8807e76db59f7d5b)), closes [#219](https://github.com/fg-labs/bwa-mem3/issues/219)
* **seed:** prefetch the correct cp_occ lines in the bwtSeed third round ([#242](https://github.com/fg-labs/bwa-mem3/issues/242)) ([c5e3827](https://github.com/fg-labs/bwa-mem3/commit/c5e3827df83b5dd85a998423d03cb93bec99263f))


### Documentation

* **best-practices:** add anti-patterns for marginally-mappable input ([#232](https://github.com/fg-labs/bwa-mem3/issues/232)) ([1344ab8](https://github.com/fg-labs/bwa-mem3/commit/1344ab86826cf6b3ba4a563bb4b482addb506428))
* **ext:** record the measured evidence behind the ungapped fast-path bound ([#233](https://github.com/fg-labs/bwa-mem3/issues/233)) ([878bda0](https://github.com/fg-labs/bwa-mem3/commit/878bda0972509f44aa2073bafd0fb51997002b33))
* **mem:** scope long-read claims to what actually works ([#240](https://github.com/fg-labs/bwa-mem3/issues/240)) ([ac090bc](https://github.com/fg-labs/bwa-mem3/commit/ac090bc13a3b2e37217b0989e204fce15f9fd07e))
* **meth:** correct --meth override claims and document -T 40 split filtering ([#226](https://github.com/fg-labs/bwa-mem3/issues/226)) ([50a8d3f](https://github.com/fg-labs/bwa-mem3/commit/50a8d3f7fb56a631ec030eef4edfd1042619710f))
* **profiles:** recommend --skip-contained-ext for short-read pipelines ([#241](https://github.com/fg-labs/bwa-mem3/issues/241)) ([37fa409](https://github.com/fg-labs/bwa-mem3/commit/37fa4098e5180a8a0b99e878b65095562ec3ec15))

## [0.6.0](https://github.com/fg-labs/bwa-mem3/compare/v0.5.0...v0.6.0) (2026-07-16)


### Features

* **version:** report whether mimalloc is the active allocator ([#217](https://github.com/fg-labs/bwa-mem3/issues/217)) ([2cd9fe9](https://github.com/fg-labs/bwa-mem3/commit/2cd9fe9c27185af98d2f6a0b59c04431a50eea0a))


### Performance

* **bam:** cut per-record work in the --bam writer (mem_aln_to_bam) ([#212](https://github.com/fg-labs/bwa-mem3/issues/212)) ([837f40e](https://github.com/fg-labs/bwa-mem3/commit/837f40e1e974217ee1c45f7a4ceea6b3b6c00821))
* **bsw:** drop dead per-row H1/H2 setup stores in the SW batch wrappers ([#211](https://github.com/fg-labs/bwa-mem3/issues/211)) ([039b09f](https://github.com/fg-labs/bwa-mem3/commit/039b09f8ecd11e724b098a1729d42703c2ce058d))
* **dedup:** drop provably-dead exact-duplicate passes in mem_sort_dedup_patch ([#205](https://github.com/fg-labs/bwa-mem3/issues/205)) ([c544258](https://github.com/fg-labs/bwa-mem3/commit/c5442582e44da347d8a81e9b58a70a18ed0467a4))
* **fmi:** arm64 lockstep for third-pass reseeding (bwtSeedStrategy) ([#215](https://github.com/fg-labs/bwa-mem3/issues/215)) ([c5f69ec](https://github.com/fg-labs/bwa-mem3/commit/c5f69ec0aaa8d2fae290228c7b24ee774f740d62))
* **kswv:** remove write-only Hmax scratch buffer from batched SW kernels ([#214](https://github.com/fg-labs/bwa-mem3/issues/214)) ([b14ef49](https://github.com/fg-labs/bwa-mem3/commit/b14ef4993ee9a60813b0412b8a5d5d40aa19b66b))
* **mem:** add unique-mapper fast paths to mem_chain_flt and mem_mark_primary_se ([#209](https://github.com/fg-labs/bwa-mem3/issues/209)) ([1700aab](https://github.com/fg-labs/bwa-mem3/commit/1700aabc263a8776dc12b66acbb566bda67a37ba))
* **mem:** drop redundant per-call scratch allocations in mem_gen_alt ([#216](https://github.com/fg-labs/bwa-mem3/issues/216)) ([3418525](https://github.com/fg-labs/bwa-mem3/commit/3418525750b15df76dd0d5c5d3634dab4c8f61d4))
* **mem:** pool per-read scratch in get_sa_entries_prefetch and mem_reg2aln ([#208](https://github.com/fg-labs/bwa-mem3/issues/208)) ([5747a13](https://github.com/fg-labs/bwa-mem3/commit/5747a131cde9c22490c5ace470b0903d395ad130))
* **mem:** remove dead/instrumentation work from the extension hot path ([#206](https://github.com/fg-labs/bwa-mem3/issues/206)) ([439be97](https://github.com/fg-labs/bwa-mem3/commit/439be9709844a0557ad01ec7cdf3b542e9acf66d))
* **seed:** replace sortSMEMs qsort with a counting sort by rid ([#207](https://github.com/fg-labs/bwa-mem3/issues/207)) ([4356b5c](https://github.com/fg-labs/bwa-mem3/commit/4356b5ce50348a2421b2b8e2517d12f642cfb8e8))
* **seed:** vectorize backwardExt occ-counting on arm64 (NEON) + hoist invariants ([#210](https://github.com/fg-labs/bwa-mem3/issues/210)) ([64a55a2](https://github.com/fg-labs/bwa-mem3/commit/64a55a25c18350a66a93438d64f3d93a77259a8a))


### Refactoring

* **mem:** static cleanup — remove dead code, silence int64 format warnings ([#213](https://github.com/fg-labs/bwa-mem3/issues/213)) ([8c57b7e](https://github.com/fg-labs/bwa-mem3/commit/8c57b7ef6b0ff45d3b6d1b17031c9f0c46879199))


### Documentation

* retitle LICENSE for BWA-MEM3 and add Fulcrum copyright ([#203](https://github.com/fg-labs/bwa-mem3/issues/203)) ([089a956](https://github.com/fg-labs/bwa-mem3/commit/089a956375b27372d812a757cc87663526126fc8))

## [0.5.0](https://github.com/fg-labs/bwa-mem3/compare/v0.4.0...v0.5.0) (2026-07-04)


### Features

* add opt-in --seed-order seed reordering (default off, byte-identical) ([#186](https://github.com/fg-labs/bwa-mem3/issues/186)) ([04749a1](https://github.com/fg-labs/bwa-mem3/commit/04749a16fb2e2a4f0bd8a318fc6bdd6439e2689f))
* add opt-in --smem-dedup (dedup identical SMEMs before chaining) ([#187](https://github.com/fg-labs/bwa-mem3/issues/187)) ([1384972](https://github.com/fg-labs/bwa-mem3/commit/1384972417677943b68b8292181017223ad27876))
* **mem:** add --adaptive-band (chain-geometry adaptive banding) for long reads ([#194](https://github.com/fg-labs/bwa-mem3/issues/194)) ([4fe92a6](https://github.com/fg-labs/bwa-mem3/commit/4fe92a6ea55ebe8617cabed79902bceeabecbd2c))
* **mem:** add --extend-mate-concordant; fix --fast --meth placement regression ([#195](https://github.com/fg-labs/bwa-mem3/issues/195)) ([c9ffef1](https://github.com/fg-labs/bwa-mem3/commit/c9ffef1551d13ef22b104a75c5be785857e01efa))
* **mem:** add --fast speed preset ([#189](https://github.com/fg-labs/bwa-mem3/issues/189)) ([a946af8](https://github.com/fg-labs/bwa-mem3/commit/a946af8fcd0309faed952662123f88d152b3b585))
* **mem:** add --max-extend-chains and bundle it into --fast ([#193](https://github.com/fg-labs/bwa-mem3/issues/193)) ([e39b3d4](https://github.com/fg-labs/bwa-mem3/commit/e39b3d497e3e10535b2d9eabb9bb7fa48d1c1379))
* **mem:** add --skip-contained-ext and enable it under --fast ([#192](https://github.com/fg-labs/bwa-mem3/issues/192)) ([2d2b2b4](https://github.com/fg-labs/bwa-mem3/commit/2d2b2b437b70cdc5c445ea83da7cb158af654cd1))


### Bug Fixes

* **bandedSWA:** 8-bit SW drops query-end gscore/gtle on zero-score-row exit ([#198](https://github.com/fg-labs/bwa-mem3/issues/198)) ([611e21b](https://github.com/fg-labs/bwa-mem3/commit/611e21b924d09ccbd7e2e3d18092a90dc798e4b2))
* **bandedSWA:** getScores{8,16} must not scribble padding past numPairs ([#199](https://github.com/fg-labs/bwa-mem3/issues/199)) ([9aae808](https://github.com/fg-labs/bwa-mem3/commit/9aae808c0a721bf88dfc4c486c8bf8b785cdd508))


### Performance

* **bandedSWA:** gate the getScores overshoot guard to sub-slice callers ([#201](https://github.com/fg-labs/bwa-mem3/issues/201)) ([162e909](https://github.com/fg-labs/bwa-mem3/commit/162e90923488fea4634c00a990156794940b9797))


### Refactoring

* **kswv:** drop duplicate F warm-up prefetch in kswv512_16 ([#191](https://github.com/fg-labs/bwa-mem3/issues/191)) ([6e4cf2b](https://github.com/fg-labs/bwa-mem3/commit/6e4cf2b34a527309e8dc675e941a9fd548feadf1))


### Documentation

* **changelog:** backfill the 0.4.0 breaking-change notice ([#183](https://github.com/fg-labs/bwa-mem3/issues/183)) ([e1c381a](https://github.com/fg-labs/bwa-mem3/commit/e1c381ad4b27e7e3bcf26dbbfeea130cff6e2af0))
* **changelog:** render the live changelog, not the frozen NEWS.md ([#184](https://github.com/fg-labs/bwa-mem3/issues/184)) ([12a46d8](https://github.com/fg-labs/bwa-mem3/commit/12a46d8317fd3dabb031f835348ebdf126e76601))
* **contributing:** document breaking-change commit footers ([#181](https://github.com/fg-labs/bwa-mem3/issues/181)) ([4ebd122](https://github.com/fg-labs/bwa-mem3/commit/4ebd1220ffda0a0da544bfe0a9bab94840d73a8c))
* **release:** describe the release-please flow, not manual tagging ([#182](https://github.com/fg-labs/bwa-mem3/issues/182)) ([fc0c4b6](https://github.com/fg-labs/bwa-mem3/commit/fc0c4b65f356e6571c9f33f563a13d9fc6bd3f3e))

## [0.4.0](https://github.com/fg-labs/bwa-mem3/compare/v0.3.0...v0.4.0) (2026-06-27)


### ⚠ BREAKING CHANGES

* **index:** `bwa-mem3 index` no longer writes the unpacked `.0123` reference file by default ([#177](https://github.com/fg-labs/bwa-mem3/pull/177)). `bwa-mem3 mem` now reconstructs reference bases from the packed `.pac` on demand ("pac-fetch") and ignores any `.0123` present, so the file is redundant for bwa-mem3 itself. External tools that read `.0123` directly — most notably sharing a single index with **bwa-mem2** — will break, since the expected file is now absent. To restore the old on-disk layout, re-run indexing with the new opt-in flag `index --emit-unpacked-ref`. Alignment output is byte-for-byte identical; the change is purely to the index artifact set.


### Features

* **mem:** --min-ext-len opt-in filter to skip extension of short seeds ([#169](https://github.com/fg-labs/bwa-mem3/issues/169)) ([13db252](https://github.com/fg-labs/bwa-mem3/commit/13db252ad4c7c39eff18bd3674430aac51570346))
* **meth:** native bisulfite (BS-seq) alignment via --meth (D3) ([#174](https://github.com/fg-labs/bwa-mem3/issues/174)) ([a0296b1](https://github.com/fg-labs/bwa-mem3/commit/a0296b1766f17aab85b720fef1cf7f3984735105))


### Performance

* **mem:** pac-fetch the reference from .pac instead of loading/building .0123 ([#177](https://github.com/fg-labs/bwa-mem3/issues/177)) ([9c4bbf2](https://github.com/fg-labs/bwa-mem3/commit/9c4bbf217d3cbf02143a2fbd9fb75329126d9010))
* **meth:** batched (SIMD) asymmetric mate rescue (closes [#173](https://github.com/fg-labs/bwa-mem3/issues/173)) ([#175](https://github.com/fg-labs/bwa-mem3/issues/175)) ([f146a18](https://github.com/fg-labs/bwa-mem3/commit/f146a18a32232a6ea05b040b10917a464a2d264c))


### Documentation

* **book:** document the libdeflate build prerequisite (incl. AL2023) ([#172](https://github.com/fg-labs/bwa-mem3/issues/172)) ([c2b6ec7](https://github.com/fg-labs/bwa-mem3/commit/c2b6ec78bc8fadb1aa63f99aee4f90671f840e7d))
* **book:** recommend -y 0 (drop 3rd-round seeding) as an opt-in speed knob ([#171](https://github.com/fg-labs/bwa-mem3/issues/171)) ([ca9ac1f](https://github.com/fg-labs/bwa-mem3/commit/ca9ac1f2289c200708193c3f6ebbe094f230d79d))
* collapse the mdbook sidebar into nested, foldable sections ([#180](https://github.com/fg-labs/bwa-mem3/issues/180)) ([c6ac47b](https://github.com/fg-labs/bwa-mem3/commit/c6ac47b9ebc3146144377f4a56e689b3fdf11946))
* deep mdbook cleanup — dedup, consolidate, and tighten ([#179](https://github.com/fg-labs/bwa-mem3/issues/179)) ([54d6d11](https://github.com/fg-labs/bwa-mem3/commit/54d6d1161c374d7eb7fcef0eaac7c1156fe0bfb4))
* **meth:** disclose collapsed-mode placement drift vs bwameth.py ([#178](https://github.com/fg-labs/bwa-mem3/issues/178)) ([96f29e2](https://github.com/fg-labs/bwa-mem3/commit/96f29e257e9acd9072caa246b1589c8943babc26))
* **settings-profiles:** note repeat-aggregating downstream caveat for -m 10 ([#168](https://github.com/fg-labs/bwa-mem3/issues/168)) ([38fd1ec](https://github.com/fg-labs/bwa-mem3/commit/38fd1ec9717e29cace9eac33bd643a7da001da4e))

## [0.3.0](https://github.com/fg-labs/bwa-mem3/compare/v0.2.2...v0.3.0) (2026-06-21)


### Features

* **bsw:** make the 8-bit h0-prefix seed unsigned [0,255] ([#151](https://github.com/fg-labs/bwa-mem3/issues/151)) ([9f51c5f](https://github.com/fg-labs/bwa-mem3/commit/9f51c5f521612a7c3e1c0dae9d71db38be0121b8))
* **bsw:** recover the 8-bit banded Smith–Waterman path for reads ≥128 bp ([#140](https://github.com/fg-labs/bwa-mem3/issues/140)) ([155a916](https://github.com/fg-labs/bwa-mem3/commit/155a91632de602663377197ff1168d200a24f344))
* **kswv:** AVX2 16-bit mate-rescue kernel (kswv256_16) ([#162](https://github.com/fg-labs/bwa-mem3/issues/162)) ([9107b82](https://github.com/fg-labs/bwa-mem3/commit/9107b825189da87822bccac152e4a98daf00aac1))
* **meth:** carry original-reference `@SQ` `M5`/`UR` and `@CO`/`@PG` into `--meth` headers ([#139](https://github.com/fg-labs/bwa-mem3/issues/139)) ([e94ad8b](https://github.com/fg-labs/bwa-mem3/commit/e94ad8b847b6637b95aea6dff3e07f560e22a532))
* **prof:** off-by-default --profile stage-timing instrumentation ([#152](https://github.com/fg-labs/bwa-mem3/issues/152)) ([83cf7ab](https://github.com/fg-labs/bwa-mem3/commit/83cf7ab9d8e4589354e16654726fbdfaa1b4b63c))
* **reader:** content-detecting FASTQ reader fast path (libdeflate BGZF) ([#128](https://github.com/fg-labs/bwa-mem3/issues/128)) ([cdd71bf](https://github.com/fg-labs/bwa-mem3/commit/cdd71bf4590cf2e1a5d908d6e852f780c93c3a8a))


### Bug Fixes

* **bsw:** bound getScores8/16 prefetch reads to the padding contract ([#150](https://github.com/fg-labs/bwa-mem3/issues/150)) ([87ed5d4](https://github.com/fg-labs/bwa-mem3/commit/87ed5d4a78b6b78789bca45be5649373576ee668))
* **fmi:** widen mem_lim to int64 and guard SA-entry allocations ([#156](https://github.com/fg-labs/bwa-mem3/issues/156)) ([2d18c1e](https://github.com/fg-labs/bwa-mem3/commit/2d18c1ed00de5e4a60fa31d15fec04c7b74877a0))
* **kthread:** drive kt_for with a persistent worker pool ([#154](https://github.com/fg-labs/bwa-mem3/issues/154)) ([26b24e7](https://github.com/fg-labs/bwa-mem3/commit/26b24e7bac9fd7f73921b902ba98a628776087b6))
* **meth:** emit `-R` read group as `@RG` header in `--meth` mode ([#137](https://github.com/fg-labs/bwa-mem3/issues/137)) ([ccd1fc5](https://github.com/fg-labs/bwa-mem3/commit/ccd1fc56d26bfd54ab1f9604845332353afd75a6))
* **seeding:** widen SMEM read positions from int16_t to int32_t ([#142](https://github.com/fg-labs/bwa-mem3/issues/142)) ([037c418](https://github.com/fg-labs/bwa-mem3/commit/037c418b3b3c49deeb41bd57e37eacc7895e6fca))
* **test:** make meth layer-2 FAIL diagnostics reachable under set -e ([#133](https://github.com/fg-labs/bwa-mem3/issues/133)) ([d2d6688](https://github.com/fg-labs/bwa-mem3/commit/d2d66880bce736900b81e7610cbae1d1556ba7ea))


### Performance

* **bsw:** AVX2 SIMD tuning for the Smith-Waterman kernels ([#161](https://github.com/fg-labs/bwa-mem3/issues/161)) ([458b216](https://github.com/fg-labs/bwa-mem3/commit/458b21653cb6f84808032360a85a78f291064a4e))
* **bsw:** NEON SIMD tuning for the Smith-Waterman kernels ([#160](https://github.com/fg-labs/bwa-mem3/issues/160)) ([d971ff0](https://github.com/fg-labs/bwa-mem3/commit/d971ff0694d8a90cf836dfff873534c9d3a160a4))
* **bsw:** prefetch next batch's ref/query in the AVX2 8-bit wrapper ([#163](https://github.com/fg-labs/bwa-mem3/issues/163)) ([e6082a0](https://github.com/fg-labs/bwa-mem3/commit/e6082a05fc6af07da26dae90210ea651b601dd91))
* **bsw:** short-circuit the inert per-row re-baseline scan ([#147](https://github.com/fg-labs/bwa-mem3/issues/147)) ([8e284e0](https://github.com/fg-labs/bwa-mem3/commit/8e284e0cf274e8c77c78457e372c1ab92de69f3f))
* **bsw:** vectorize the per-row epilogue side-channel loop ([#149](https://github.com/fg-labs/bwa-mem3/issues/149)) ([403aeb7](https://github.com/fg-labs/bwa-mem3/commit/403aeb76ad8ec2664736998c018a7e033af0c070))
* **fmi:** size SA-entry staging buffers to the exact write count ([#157](https://github.com/fg-labs/bwa-mem3/issues/157)) ([aa0fe33](https://github.com/fg-labs/bwa-mem3/commit/aa0fe3335c58badf97893fae32f67ff02f69217a))
* **read:** vendored zlib-ng inflate + chunk cap + 3rd pipeline worker ([#153](https://github.com/fg-labs/bwa-mem3/issues/153)) ([5cf89e3](https://github.com/fg-labs/bwa-mem3/commit/5cf89e3c60e01bec089183e511efa27327414035))
* **sw:** reassociate affine-gap recurrences on NEON (kswv + bandedSWA) ([#166](https://github.com/fg-labs/bwa-mem3/issues/166)) ([a02fcb4](https://github.com/fg-labs/bwa-mem3/commit/a02fcb446574d5b5d03abdbf73c9b129deead2d4))


### Refactoring

* **bsw:** derive extension gaps from H (standard Gotoh), not M ([#141](https://github.com/fg-labs/bwa-mem3/issues/141)) ([f715fbd](https://github.com/fg-labs/bwa-mem3/commit/f715fbd92b4c4acabe7b7ab8dde35d5ad7fe211c))
* **bsw:** drop the dead qlen[] parameter from the 8-bit kernels ([#143](https://github.com/fg-labs/bwa-mem3/issues/143)) ([42321df](https://github.com/fg-labs/bwa-mem3/commit/42321df6100f04451b8a4d5ed7f214ea28522ec8))
* **bsw:** remove dead SW code paths (SORT_PAIRS, non-CORE macros, SSE2 polyfill) ([#148](https://github.com/fg-labs/bwa-mem3/issues/148)) ([ad8937e](https://github.com/fg-labs/bwa-mem3/commit/ad8937ef05fb48992ca61be70c21d4b6e245a7b3))


### Documentation

* add memory budgeting and data-type tuning guide ([#145](https://github.com/fg-labs/bwa-mem3/issues/145)) ([7127d80](https://github.com/fg-labs/bwa-mem3/commit/7127d80f7204c7951bf74d8191a883c4e0f00a82))
* add situational --supp-rep-hard-cap 20 note for SV-aware pipelines ([#134](https://github.com/fg-labs/bwa-mem3/issues/134)) ([e20bc0e](https://github.com/fg-labs/bwa-mem3/commit/e20bc0e28d3501a732e9b1eba88a9920ec877bae))
* **perf:** refresh reference-architecture table to v0.3.0 (a02fcb4) ([#167](https://github.com/fg-labs/bwa-mem3/issues/167)) ([2cd7bfa](https://github.com/fg-labs/bwa-mem3/commit/2cd7bfa1cfb5392856905df3b88cebeda2a1d659))
* **perf:** what drives the speedup, full perf-PR catalog, and fix the stale RTD build ([#155](https://github.com/fg-labs/bwa-mem3/issues/155)) ([0b01fb7](https://github.com/fg-labs/bwa-mem3/commit/0b01fb73a2e9a8b9e8899f30b2afd94c9e6a1418))
* recommend -s 0 for --meth Pass-2 re-seeding in settings profiles ([#132](https://github.com/fg-labs/bwa-mem3/issues/132)) ([45e02e0](https://github.com/fg-labs/bwa-mem3/commit/45e02e0095f862c99eaf99273b2912ff2229ec9b))
* recommend a recent compiler on aarch64, with measured NEON numbers ([#165](https://github.com/fg-labs/bwa-mem3/issues/165)) ([b591684](https://github.com/fg-labs/bwa-mem3/commit/b59168479ddb0fddba86cd0eb0e61ee0a1c8fcb0))
* settings profiles (bwa drop-in vs recommended) ([#131](https://github.com/fg-labs/bwa-mem3/issues/131)) ([4d845c9](https://github.com/fg-labs/bwa-mem3/commit/4d845c92154d29f944191ec080009d916ce4b72b))

## [0.2.2](https://github.com/fg-labs/bwa-mem3/compare/v0.2.1...v0.2.2) (2026-06-08)


### Bug Fixes

* **lto:** pass explicit -flto=N on GCC to bypass jobserver under sandboxes ([#122](https://github.com/fg-labs/bwa-mem3/issues/122)) ([c6240a7](https://github.com/fg-labs/bwa-mem3/commit/c6240a74959e35a8fcbaa29a6f3b78c6a67ec7c3))
* **smem:** free lockstep SMEM caches at thread exit (closes [#116](https://github.com/fg-labs/bwa-mem3/issues/116)) ([#117](https://github.com/fg-labs/bwa-mem3/issues/117)) ([9454f10](https://github.com/fg-labs/bwa-mem3/commit/9454f106cba06ad3b139a01e5fc2b73a24356ce7))


### Performance

* **sort:** stabilize alnreg tie-breaks + drop in pdqsort at dedup-patch sort sites ([#123](https://github.com/fg-labs/bwa-mem3/issues/123)) ([85f8542](https://github.com/fg-labs/bwa-mem3/commit/85f8542080863c2fda9c198cb8743d18803e83e8))


### Documentation

* **bench:** inject generated divergence catalog + per-release concordance table ([#126](https://github.com/fg-labs/bwa-mem3/issues/126)) ([fea1c94](https://github.com/fg-labs/bwa-mem3/commit/fea1c9407a71b4e6c0efbd12aaa6a3a98b85861a))
* correct concordance claims and document supplementary-alignment divergence ([#125](https://github.com/fg-labs/bwa-mem3/issues/125)) ([8b2dc69](https://github.com/fg-labs/bwa-mem3/commit/8b2dc697c926c779b03be6369e8a08e95283b284))
* document bwa-mem3&lt;-&gt;bwa-mem2 non-bit-identity + auditable PR list ([#124](https://github.com/fg-labs/bwa-mem3/issues/124)) ([bffae5a](https://github.com/fg-labs/bwa-mem3/commit/bffae5a09267877fe514c458d4956b717bcefb8f))

## [0.2.1](https://github.com/fg-labs/bwa-mem3/compare/v0.2.0...v0.2.1) (2026-05-17)


### Bug Fixes

* **changelog:** strip preamble so release-please owns the file ([#112](https://github.com/fg-labs/bwa-mem3/issues/112)) ([56e580c](https://github.com/fg-labs/bwa-mem3/commit/56e580cf4f2515280440b2882a32c3a6a7b6d15c))
* **mapq:** propagate SMEM SA-count to seed n_hits so --supp-rep-hard-cap works ([#101](https://github.com/fg-labs/bwa-mem3/issues/101)) ([cca9d4f](https://github.com/fg-labs/bwa-mem3/commit/cca9d4f41023e501636d2b1ebd2d1e825f95e8e3))
* **smem:** track enc_qdb byte capacity separately from wsize_mem ([#100](https://github.com/fg-labs/bwa-mem3/issues/100)) ([ab922b6](https://github.com/fg-labs/bwa-mem3/commit/ab922b6af531ed4521b6cd9bd3c43a661d51a421))


### Documentation

* **readme:** add bioconda badges and install instructions ([#106](https://github.com/fg-labs/bwa-mem3/issues/106)) ([830276c](https://github.com/fg-labs/bwa-mem3/commit/830276ce01774805c20bfff69c63cfebab239166))

#ifndef BWAMEM3_KSW_GLOBAL2_WAVE_H
#define BWAMEM3_KSW_GLOBAL2_WAVE_H
/* Anti-diagonal (wavefront) SIMD ksw_global2 kernels.
 * Included by ksw.cpp AFTER MINUS_INF, eh_t, and push_cigar are defined.
 * Each SIMD tier (selected by target macros) compiles an int32 kernel
 * (ksw_g2_wave) and, on tiers that support it, a narrower int16 kernel
 * (ksw_g2_wave16) with twice the lanes. Both are byte-identical to the scalar
 * ksw_global2 (see the LEMMA 1-10 proof of byte-identity below): the int16
 * recurrence is the same, only the element width differs, and it is entered
 * only where ksw_g2_wave16_safe proves the whole H/E/F range fits int16. The
 * dispatcher in ksw.cpp uses a SIMD path only for m==5, a CIGAR request, and
 * w >= the tier's width crossover (perf gate; every path is byte-identical so
 * the choice never changes output — it only picks the fastest safe kernel). */
/*=============================================================================
 * PROOF OF BYTE-IDENTICAL OUTPUT: ksw_g2_wave  ==  ksw_global2_scalar
 *=============================================================================
 * Claim: for every feasible input with qlen>=1 and w >= |tlen-qlen|+3 (the
 * band-width floor guaranteed by every production caller, see bwa.cpp), the
 * anti-diagonal SIMD kernel ksw_g2_wave (NEON/AVX2/AVX-512 -- same recurrence
 * and boundary logic, differing only in vector width and intrinsic spelling)
 * returns a score, n_cigar and cigar[] byte-identical to ksw_global2_scalar.
 * The proof is empirically gated by test/unit/test_ksw_global2_wave.cpp
 * (differential parity vs the scalar oracle on randomized + edge inputs).
 *
 * Notation. H/E/F/d are the usual banded-Gotoh cells; M(i,j)=H(i-1,j-1)+S(i,j)
 * is the diagonal (match/mismatch) candidate. Scalar's row band is
 * beg(i)=max(0,i-w), end(i)=min(qlen,i+w+1). The wavefront visits anti-
 * diagonals s=i+j in increasing order; on diagonal s its in-band row range is
 * iS(s)=max(0,s-(qlen-1),ceil((s-w)/2)), iE(s)=min(tlen-1,s,floor((s+w)/2)),
 * coded as (s-w+1)>>1 and (s+w)>>1. Rolling buffers Hb/Eb/Fb (ring size 3/2/2
 * over s, indexed by row i with a fixed store offset i+1) hold H/E/F; z[]/
 * zoff[]/ziS[] hold direction bytes in diagonal-major layout.
 *
 * LEMMA 1 (recurrence locality). H,E,F,d at (i,j) are a pure function of
 * exactly three predecessors: H(i-1,j-1) (diagonal), E(i,j) (produced by row
 * i-1, same column, via the E-threading store), and F(i,j) (produced by
 * column j-1, same row, via the F-threading store). Each predecessor has
 * strictly smaller i+j. Anti-diagonal order (wavefront) and row-major order
 * (scalar) are both linear extensions of this dependency DAG, so by induction
 * on s=i+j both compute the identical mathematical H/E/F/d at every cell,
 * independent of evaluation order.
 *
 * LEMMA 2 (score lookup fidelity). The SIMD table lookup (2-level vpshufb /
 * vqtbl2 / vpermb over the flattened `mat` bytes, indexed by target[i]*m+
 * query[j], sign-extended via shift-left-24/shift-right-24) returns exactly
 * mat[target[i]*m+query[j]], the same int8_t scalar's query-profile q[j]
 * yields, sign-extended the same way by C's usual arithmetic conversions.
 * AVX2's reversed-query buffer and NEON's rev4 only change how the byte is
 * *loaded* (for contiguous SIMD access), never its value.
 *
 * LEMMA 3 (direction-byte identity). Given identical inputs (Lemma 1), both
 * kernels run the identical 4-comparison chain with identical tie-breaks:
 * origin = (M>=E)?0:1, overridden to 2 if !(max(M,E)>=F); E-extend flag set
 * iff (E-e_del) > (M-oe_del); F-extend flag set iff (F-e_ins) > (M-oe_ins).
 * The wavefront's mm/e/h_me/h/t1/em/Ecur/t2/fm/Fcur variables are the exact
 * same int32 add/sub/max ops as scalar's m/e/h/f/t; hence d, H, E, F are
 * bit-identical cell-for-cell.
 *
 * LEMMA 4 (boundary seeding equivalence). Scalar's eh[] init (eh[0].h=0;
 * eh[j].h=-(o_ins+e_ins*j) for 1<=j<=w, else MINUS_INF; eh[j].e=MINUS_INF
 * always) and its implicit column h1=-(o_del+e_del*(i+1)) at beg==0 are
 * reproduced exactly by the wavefront's i==0 and j==0 lane overrides (Hi0,
 * Hj0, applied j0m-then-i0m so cell (0,0) resolves to 0), by the freset mask
 * (F reset to MINUS_INF at j==beg(i), matching scalar's per-row f=MINUS_INF),
 * and by evalid's i>=1 guard (E undefined at row 0, matching eh[j].e init).
 *
 * LEMMA 5 (band-region equivalence). {(i,j): 0<=i<tlen, 0<=j<qlen, beg(i)<=j
 * <end(i)} equals {(i,s-i): iS(s)<=i<=iE(s)} exactly: j>=i-w <=> i<=floor((s+w)/2)
 * and j<=i+w <=> i>=ceil((s-w)/2), combined with 0<=i<tlen and 0<=j<qlen --
 * algebraically identical to the iS/iE formulas. No cell is computed by one
 * kernel and skipped by the other.
 *
 * LEMMA 6 (band monotonicity / column persistence). beg(i),end(i) are
 * nondecreasing in i, so once column j exits the band it never re-enters.
 * Hence at any row i where column j is in-band, eh[j].e read at that row is
 * either the real E value row i-1 produced (if (i-1,j) was in-band) or the
 * untouched MINUS_INF init (iff row i is column j's first entry). The
 * wavefront's evalid mask ((i-1,j) in scalar band, derived the same way as
 * Lemma 5) selects precisely between these two cases.
 *
 * LEMMA 7 (ring-buffer indexing correctness). The fixed store offset (+1
 * relative to the producing row) makes Hd2[i]=H(i-1,j-1), Ep1[i]=E(i,j) (per
 * Lemma 6), Fp1[i+1]=F(i,j) exactly the predecessors Lemma 1 requires. Since
 * an in-band cell's diagonal/E/F predecessors are themselves in-band or an
 * explicitly-overridden boundary lane (Lemmas 4-5), stale values left in
 * unread out-of-band ring slots are never read by an in-band cell.
 *
 * LEMMA 8 (interior fast-path equivalence). The interior chunk-skip branch
 * fires only when i0>iS(s) and i0+lanes-1<iE(s); this implies every lane in
 * the chunk has i>=1, j=s-i>=1, and is fully in-band, so i0m, j0m, freset are
 * always false and evalid/inband always true -- exactly the conditions under
 * which the general branch reduces to the interior branch's unconditional
 * loads. The split is a pure throughput optimization with no semantic effect.
 *
 * LEMMA 9 (final score, last-write-wins). Column j=qlen-1 is visited at
 * diagonal s=i+(qlen-1), strictly increasing in i in both orderings; the last
 * write in both is therefore at i=tlen-1 (feasible by the caller's w floor),
 * so both capture H(tlen-1,qlen-1) as the final score. Precondition qlen>=1
 * (the sole production call shape) makes this capture reachable; the tlen==0
 * corner reduces identically in both to the closed-form -(o_ins+e_ins*qlen).
 *
 * LEMMA 10 (backtrack trajectory identity). Both backtrack loops start at the
 * textually identical cell (tlen-1, min(tlen-1+w,qlen-1)) and apply the
 * textually identical transition which = d>>(which<<1)&3 against the *same*
 * d-byte (Lemmas 3+7), pushing ops via the same shared push_cigar. By
 * induction on backtrack steps the full (i,k,op) trajectory is identical,
 * so the pushed op/len sequence, the identical trailing appends, and the
 * identical reversal loop yield byte-identical n_cigar and cigar[].
 *
 * THEOREM. Lemmas 1-10 compose: for qlen>=1 and w>=|tlen-qlen|+3, ksw_g2_wave
 * and ksw_global2_scalar produce identical (score, n_cigar, cigar[]) on every
 * feasible input, for all three arch variants.
 *
 * OUT OF SCOPE. Infeasible bands (w < |tlen-qlen|+3, where scalar may return
 * MINUS_INF or an incomplete band) are unreachable -- every production caller
 * enforces the w floor -- and are explicitly excluded; the kernels are only
 * asserted to differ, if at all, in that unreachable regime. qlen==0 is
 * likewise excluded (no production call site passes an empty query); the
 * wavefront's qlen==0 score default is a formality, not a proven-equal case.
 *===========================================================================*/
/* This is an implementation fragment, not a freestanding header: it must be
 * included from ksw.cpp AFTER MINUS_INF, eh_t, and push_cigar are defined.
 * Fail loudly at a wrong include site instead of emitting an undefined-
 * identifier cascade. (push_cigar is a function, so only MINUS_INF — defined
 * just above the include — can be checked at preprocess time; it stands in
 * for the whole contract.) */
#ifndef MINUS_INF
#  error "ksw_global2_wave.h must be included from ksw.cpp after MINUS_INF/eh_t/push_cigar are defined"
#endif
#include <cstring>
#include <climits>
#include <vector>
#if defined(__AVX512BW__) || defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__aarch64__)
#  include <arm_neon.h>
#endif

/* Reusable per-thread scratch (grown, never shrunk; no per-call zeroing —
 * every read is guarded, proven in the boundary proof). */
/* Int16 "minus infinity" sentinel used by the int16 kernels (all SIMD tiers).
 * Distinct from the int32 MINUS_INF (which does not fit int16). It is small enough that
 * its own decrement (VMINF16 - (o+e)) cannot wrap int16, and large-negative
 * enough that a real feasible score never reaches it — the ksw_g2_wave16_safe
 * gate proves both separations hold before the int16 kernel is ever entered. */
#define KSW_VMINF16 (-30000)

/* Single source of truth for the wavefront direction-byte store capacity.
 * Per diagonal the in-band cell count is at most min(qlen,2w+1); summed over
 * the tlen+qlen-1 diagonals that bounds the total, and +64 is slack for the
 * trailing SIMD over-store. KswWaveScratch::ensure() sizes zr to this and
 * ksw_g2_wave_area_ok() checks the running offset fits int — the two MUST
 * agree, so both derive the size from here rather than re-spelling the
 * formula. Arch-independent (pure integer). */
static inline long ksw_wave_zneed(int qlen,int tlen,int w){
    long band = qlen < 2L*w+1 ? qlen : 2L*w+1;
    return band*((long)tlen+qlen) + 64;
}

/* Retention policy for the direction-byte store zr (applied in ensure()). Unlike
 * the maxlen-keyed buffers — linear in read length and bounded by the run's
 * stationary read-length distribution, so grow-only is correct for them — zr
 * scales with the band*(tlen+qlen) product driven by the per-call band width w.
 * A single wide-band long-read/indel-rich pair (the caller's
 * min_w = |rlen - l_query| + 3 floor can push w into the thousands) would
 * otherwise pin that high-water footprint in this thread_local for the whole
 * process lifetime, multiplied by the thread count. So zr alone gets a windowed
 * high-water decay: track the largest zneed over a fixed window of wave calls
 * and, at window end, release zr down to that window max (floored) whenever the
 * retained capacity exceeds it by the hysteresis factor. This preserves the
 * zero-allocation steady state for both short-read WGS (zr stays tiny, never
 * shrinks) and sustained wide-band workloads (the window max stays large, never
 * shrinks), while bounding a one-off spike to at most one window of retention.
 * Sizing policy cannot change output — every zr byte the backtrack reads was
 * written earlier in the same call (LEMMA 10) — so byte-identity is untouched. */
static constexpr long KSW_ZR_SHRINK_FLOOR = 4L << 20;   // never shrink zr below 4 MiB
static constexpr int  KSW_ZR_DECAY_WINDOW = 64;         // wave calls per decay check

/* Anonymous namespace (internal linkage) is load-bearing, not cosmetic. ksw.cpp
 * is compiled once per SIMD tier (sse41/sse42/avx/avx2/avx512bw on x86), and the
 * int16 ring buffers Hb16/Eb16/Fb16 below are #if-guarded so KswWaveScratch has
 * 12 vector members on the int16-capable tiers (avx2/avx512bw/NEON) but only 9
 * on the scalar tiers. With external linkage that divergence is a silent ODR
 * violation: the implicit ctor/dtor become linkonce_odr (COMDAT) and the linker
 * folds every TU's copy to ONE — and if it keeps a 9-member destructor, the
 * int16 vectors allocated by ensure() on an int16 tier are never freed at
 * worker-thread exit, so LeakSanitizer flags 3 live allocations per worker
 * thread (x86 only; arm64 has a single tier, hence no divergent copy to fold
 * against). Internal linkage gives every tier TU its own complete ctor/dtor, so
 * the int16 vectors are freed by the same tier that allocated them. Same
 * anonymous-namespace idiom as PacFetchScratch in bntseq.cpp, but not for the
 * same reason: there it is plain internal-linkage hygiene (bntseq.cpp is a
 * single TU with no #if-divergent layout, so nothing folds — its leak is
 * handled by the thread-exit destructor itself); here the internal linkage is
 * load-bearing, preventing the per-tier destructor fold described above. */
namespace {
struct KswWaveScratch {
    int maxlen=0, pad=64, stride=0;
    std::vector<int32_t> Hb,Eb,Fb; std::vector<uint8_t> tpad,qpad,qrev,zr;
    std::vector<int> zoff,ziS;
    long zwin_max=0;   // largest zneed seen in the current decay window (see below)
    int  zwin_n=0;     // wave calls counted in the current decay window
#if defined(__AVX512BW__) || defined(__AVX2__) || defined(__aarch64__)
    std::vector<int16_t> Hb16,Eb16,Fb16;   // int16 ring buffers for the int16 wave kernel (all SIMD tiers)
#endif
    void ensure(int qlen,int tlen,int w){
        int L=qlen>tlen?qlen:tlen;
        if(L>maxlen||stride==0){                       // grow the maxlen-keyed buffers (never shrink)
            maxlen=L>maxlen?L:maxlen;
            stride=maxlen+1+32;
            Hb.assign(3L*stride,MINUS_INF); Eb.assign(2L*stride,MINUS_INF); Fb.assign(2L*stride,MINUS_INF);
            tpad.assign(maxlen+2*pad,0); qpad.assign(maxlen+2*pad,0); qrev.assign(maxlen+2*pad,0);
            zoff.assign(2*maxlen+2,0); ziS.assign(2*maxlen+2,0);
#if defined(__AVX512BW__) || defined(__AVX2__) || defined(__aarch64__)
            Hb16.assign(3L*stride,(int16_t)KSW_VMINF16); Eb16.assign(2L*stride,(int16_t)KSW_VMINF16); Fb16.assign(2L*stride,(int16_t)KSW_VMINF16);
#endif
        }
        // Direction-byte store: sized to the in-band cell count, NOT maxlen^2.
        // Per row the band holds <= min(qlen,2w+1) cells, so summed over tlen
        // rows the total is <= min(qlen,2w+1)*(tlen+qlen); +64 is slack for the
        // last SIMD store. Grows with the widest band seen and decays under the
        // windowed high-water policy below once wide bands stop arriving. (w is
        // a per-call band width, so it is not folded into maxlen-keyed growth.)
        long zneed=ksw_wave_zneed(qlen,tlen,w);
        // Windowed high-water decay (see KSW_ZR_* above). Fold this call's need
        // into the window max FIRST, then at window end release zr toward that
        // max when retained capacity runs more than the hysteresis factor (2x)
        // above it. Running before the growth below guarantees the shrink target
        // (keep >= zwin_max >= zneed) is never smaller than the current need, so
        // the post-condition zr.size() >= zneed always holds. The swap idiom is
        // required because assign()/resize() never release capacity.
        if(zneed>zwin_max) zwin_max=zneed;
        if(++zwin_n>=KSW_ZR_DECAY_WINDOW){
            long keep=zwin_max>KSW_ZR_SHRINK_FLOOR?zwin_max:KSW_ZR_SHRINK_FLOOR;
            if((long)zr.size()>2*keep) std::vector<uint8_t>(keep,0).swap(zr);
            zwin_max=0; zwin_n=0;
        }
        if(zneed>(long)zr.size()) zr.assign(zneed,0);
    }
};
}  // anonymous namespace

/* Arch-independent traceback: walk the diagonal-major direction bytes from
 * (tlen-1, qlen-1) back to the origin and emit the CIGAR. No SIMD — shared
 * verbatim by all three tier kernels (was duplicated per tier). Reads the
 * same z / zoff / ziS the kernel wrote, and the scalar's identical z bytes
 * (byte-identity proof, LEMMA 10). */
static inline void ksw_wave_backtrack(const uint8_t*z,const int*zoff,const int*ziS,int tlen,int qlen,int w,int*n_cigar_,uint32_t**cigar_){
	int n_cigar=0,m_cigar=0,which=0; uint32_t*cigar=0,tmp;
	int i=tlen-1,k=(tlen-1+w+1<qlen?tlen-1+w+1:qlen)-1;
	while(i>=0&&k>=0){ int s=i+k; which=z[zoff[s]+(i-ziS[s])]>>(which<<1)&3;
		if(which==0)cigar=push_cigar(&n_cigar,&m_cigar,cigar,0,1),--i,--k;
		else if(which==1)cigar=push_cigar(&n_cigar,&m_cigar,cigar,2,1),--i;
		else cigar=push_cigar(&n_cigar,&m_cigar,cigar,1,1),--k; }
	if(i>=0)cigar=push_cigar(&n_cigar,&m_cigar,cigar,2,i+1);
	if(k>=0)cigar=push_cigar(&n_cigar,&m_cigar,cigar,1,k+1);
	for(i=0;i<n_cigar>>1;++i)tmp=cigar[i],cigar[i]=cigar[n_cigar-1-i],cigar[n_cigar-1-i]=tmp;
	*n_cigar_=n_cigar,*cigar_=cigar;
}

/* Overflow-safety gate for the int16 kernels (all SIMD tiers). Returns true iff every H/E/F
 * value the int16 kernel could produce for this (qlen,tlen,w,gaps,mat) provably
 * stays inside a safe int16 window, so int16 output is byte-identical to the
 * int32 scalar and no vsubq_s16 on the sentinel can wrap. Derivation:
 *   A = best per-step score, B = worst per-step penalty (as a positive number).
 *   Most-positive reachable H <= qlen*A + A  (at most min(i,j)+1<=qlen diagonal
 *     steps of <=A each; gap steps are <=0; +A covers the trailing mm add).
 *   Most-negative reachable finite H/E/F >= -(L*B) - (o_max + e_max*(2w+2)) - B
 *     (L=min(qlen,tlen) diagonal penalty steps, plus an affine decay tail bounded
 *     by the 2w+1 in-band span of a row/column, plus one mm step of slack).
 * Two separations then guarantee byte-identity: (1) real values stay clear of
 * the sentinel so it always loses max() exactly as int32 MINUS_INF does, and
 * (2) the sentinel's own decrement never underflows int16. A position guard
 * (qlen,tlen<32000) keeps the int16 index math (i0/iv/jv) from overflowing on
 * long reads. Conservative by design; a false return costs only a tier of speed.
 * Arch-independent (pure integer) so it is safe to reference from any TU. */
static inline bool ksw_g2_wave16_safe(int qlen,int tlen,int w,int o_del,int e_del,int o_ins,int e_ins,int m,const int8_t*mat){
	long A=0,B=0;
	for(int k=0;k<m*m;++k){ long v=mat[k]; if(v>A)A=v; if(-v>B)B=-v; }
	long o_max=o_del>o_ins?o_del:o_ins, e_max=e_del>e_ins?e_del:e_ins;
	long L=qlen<tlen?qlen:tlen, W=w, SLACK=64;
	long H_upper=(long)qlen*A + A;
	long H_lower=-(L*B) - (o_max + e_max*(2*W+2)) - B;
	return qlen<32000 && tlen<32000
		&& (long)tlen + W <= 32000   // keeps iv+w+1 (band-edge index math) inside int16
		&& H_upper <= 32767 - SLACK
		&& H_lower >= (long)KSW_VMINF16 + o_max + e_max + SLACK
		&& (long)KSW_VMINF16 - (o_max + e_max) > -32768 + SLACK;
}

/* The wavefront direction-byte store is addressed by a per-diagonal `int` offset
 * (zoff/zbase_s) that accumulates the in-band cell count across diagonals. Return
 * true iff that running total provably fits `int`, so the offset can never wrap.
 * zneed = min(qlen,2w+1)*(tlen+qlen)+64 mirrors KswWaveScratch::ensure()'s sizing.
 * For any realistic read this is orders of magnitude below INT_MAX (the int16
 * gate's tlen+w<=32000 bound already keeps the int16 path well clear); only a
 * pathological multi-megabase single alignment region could exceed it, and such a
 * call falls back to the (long-indexed, wrap-safe) scalar path. Arch-independent. */
static inline bool ksw_g2_wave_area_ok(int qlen,int tlen,int w){
	return ksw_wave_zneed(qlen,tlen,w) <= 0x7fffffffL;
}

#if defined(__AVX512BW__)
#define KSW_WAVE_WMIN 16
static int ksw_g2_wave(int qlen,const uint8_t*query,int tlen,const uint8_t*target,int m,const int8_t*mat,int o_del,int e_del,int o_ins,int e_ins,int w,int*n_cigar_,uint32_t**cigar_,KswWaveScratch&S){
	int oe_del=o_del+e_del,oe_ins=o_ins+e_ins;
	assert(m==5);             // AVX2 folds the score index as tv*5; NEON/non-VBMI tables hold <=32 entries — any m!=5 is silently wrong
	assert(n_cigar_&&cigar_); // score is captured only on the z!=0 path, so a score-only call would return MINUS_INF for qlen>w
	if(n_cigar_)*n_cigar_=0; bool want=(n_cigar_&&cigar_);
	uint8_t*z=want?S.zr.data():0; const int stride=S.stride,pad=S.pad;
	int*zoff=S.zoff.data(),*ziS=S.ziS.data(); long roff=0;
	int32_t*Hb=S.Hb.data(),*Eb=S.Eb.data(),*Fb=S.Fb.data();
	auto Hs=[&](int d){return &Hb[(long)(((d%3)+3)%3)*stride];};
	auto Es=[&](int d){return &Eb[(long)(((d%2)+2)%2)*stride];};
	auto Fs=[&](int d){return &Fb[(long)(((d%2)+2)%2)*stride];};
	uint8_t*tpad=S.tpad.data(),*qpad=S.qpad.data();
	for(int i=0;i<tlen;++i)tpad[pad+i]=target[i]; for(int j=0;j<qlen;++j)qpad[pad+j]=query[j];
#if defined(__AVX512VBMI__)
	int8_t tab64[64]; memset(tab64,0,64); memcpy(tab64,mat,m*m<=64?m*m:64);
	__m512i TAB=_mm512_loadu_si512((const void*)tab64);
#else
	int8_t tlo[16],thi[16]; memset(tlo,0,16); memset(thi,0,16);
	for(int i=0;i<m*m&&i<16;++i)tlo[i]=mat[i]; for(int i=16;i<m*m&&i<32;++i)thi[i-16]=mat[i];
	const __m512i TABLO=_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)tlo));
	const __m512i TABHI=_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)thi));
#endif
	const __m512i REVIDX=_mm512_set_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
	const __m512i LANEID=_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
	const __m512i VMINF=_mm512_set1_epi32(MINUS_INF);
	int32_t score; if(qlen==0)score=0; else if(qlen<=w)score=-(o_ins+e_ins*qlen); else score=MINUS_INF;
	const int Sdim=(tlen-1)+(qlen-1)+1;
	for(int s=0;s<Sdim;++s){
		int32_t*Hc=Hs(s),*Ec=Es(s),*Fc=Fs(s); int32_t*Hd2=Hs(s-2),*Ep1=Es(s-1),*Fp1=Fs(s-1);
		// tight in-band i-range on diagonal s (band cells are a contiguous window):
		//   iS = max(0, s-(qlen-1), ceil((s-w)/2)),  iE = min(tlen-1, s, floor((s+w)/2))
		int iS=s-(qlen-1); { int a=(s-w+1)>>1; if(a>iS)iS=a; } if(iS<0)iS=0;
		int iE=s;          { int a=(s+w)>>1;   if(a<iE)iE=a; } if(iE>tlen-1)iE=tlen-1;
		int i_hi=iE;
		int zbase_s=(int)roff;
		if(want){ zoff[s]=zbase_s; ziS[s]=iS; xassert((long)zbase_s+(iE>=iS?iE-iS+1:0)<=(long)S.zr.size(),"ksw wave: direction-store overrun"); if(iE>=iS) roff += (iE-iS+1); }
		for(int i0=iS;i0<=iE;i0+=16){
			bool interior=(i0>iS && i0+15<iE);
			__m512i Hdiag,e,f; __mmask16 inband; __m512i iv,jv,beg;
			if (interior) {
				// interior chunk: no i==0/j==0/e-invalid/f-reset lanes; all in band
				Hdiag=_mm512_loadu_si512(&Hd2[i0]);
				e=_mm512_loadu_si512(&Ep1[i0]);
				f=_mm512_loadu_si512(&Fp1[i0+1]);
				inband=0xFFFF;
			} else {
				iv=_mm512_add_epi32(_mm512_set1_epi32(i0),LANEID);
				jv=_mm512_sub_epi32(_mm512_set1_epi32(s),iv);
				beg=_mm512_max_epi32(_mm512_setzero_si512(),_mm512_sub_epi32(iv,_mm512_set1_epi32(w)));
				__m512i endc=_mm512_min_epi32(_mm512_set1_epi32(qlen),_mm512_add_epi32(iv,_mm512_set1_epi32(w+1)));
				inband=_mm512_cmpge_epi32_mask(iv,_mm512_setzero_si512())&_mm512_cmplt_epi32_mask(iv,_mm512_set1_epi32(tlen))
					&_mm512_cmpge_epi32_mask(jv,beg)&_mm512_cmplt_epi32_mask(jv,endc);
				Hdiag=_mm512_loadu_si512(&Hd2[i0]);
				__mmask16 i0m=_mm512_cmpeq_epi32_mask(iv,_mm512_setzero_si512());
				__mmask16 j0m=_mm512_cmpeq_epi32_mask(jv,_mm512_setzero_si512());
				__m512i Hi0=_mm512_mask_blend_epi32(j0m, _mm512_sub_epi32(_mm512_set1_epi32(-o_ins),_mm512_mullo_epi32(_mm512_set1_epi32(e_ins),jv)), _mm512_setzero_si512());
				__m512i Hj0=_mm512_sub_epi32(_mm512_set1_epi32(-o_del),_mm512_mullo_epi32(_mm512_set1_epi32(e_del),iv));
				Hdiag=_mm512_mask_blend_epi32(j0m,Hdiag,Hj0);
				Hdiag=_mm512_mask_blend_epi32(i0m,Hdiag,Hi0);
				__m512i begm1=_mm512_max_epi32(_mm512_setzero_si512(),_mm512_sub_epi32(iv,_mm512_set1_epi32(w+1)));
				__m512i endm1=_mm512_min_epi32(_mm512_set1_epi32(qlen),_mm512_add_epi32(iv,_mm512_set1_epi32(w)));
				__mmask16 evalid=_mm512_cmpge_epi32_mask(iv,_mm512_set1_epi32(1))&_mm512_cmpge_epi32_mask(jv,begm1)&_mm512_cmplt_epi32_mask(jv,endm1);
				e=_mm512_mask_blend_epi32(evalid,VMINF,_mm512_loadu_si512(&Ep1[i0]));
				__mmask16 freset=_mm512_cmpeq_epi32_mask(jv,beg);
				f=_mm512_mask_blend_epi32(freset,_mm512_loadu_si512(&Fp1[i0+1]),VMINF);
			}
			// score
			__m128i tb=_mm_loadu_si128((const __m128i*)&tpad[pad+i0]);
			__m128i qb=_mm_loadu_si128((const __m128i*)&qpad[pad+(s-i0-15)]);
			__m512i tv=_mm512_cvtepu8_epi32(tb);
			__m512i qv=_mm512_permutexvar_epi32(REVIDX,_mm512_cvtepu8_epi32(qb));
			__m512i idx=_mm512_add_epi32(_mm512_mullo_epi32(tv,_mm512_set1_epi32(m)),qv);
#if defined(__AVX512VBMI__)
			__m512i res=_mm512_permutexvar_epi8(idx,TAB);                     // single vpermb
#else
			__m512i slo=_mm512_shuffle_epi8(TABLO,idx);                       // VBMI-free 2-level vpshufb
			__m512i shi=_mm512_shuffle_epi8(TABHI,_mm512_sub_epi8(idx,_mm512_set1_epi8(16)));
			__m512i res=_mm512_mask_blend_epi32(_mm512_cmpgt_epi32_mask(idx,_mm512_set1_epi32(15)),slo,shi);
#endif
			__m512i Sv=_mm512_srai_epi32(_mm512_slli_epi32(res,24),24);
			__m512i mm=_mm512_add_epi32(Hdiag,Sv);
			__m512i h_me=_mm512_max_epi32(mm,e); __m512i h=_mm512_max_epi32(h_me,f);   // h/E'/F' via max (byte-identical on ties)
			__m512i t1=_mm512_sub_epi32(mm,_mm512_set1_epi32(oe_del)); __m512i em=_mm512_sub_epi32(e,_mm512_set1_epi32(e_del));
			__m512i Ecur=_mm512_max_epi32(em,t1);
			__m512i t2=_mm512_sub_epi32(mm,_mm512_set1_epi32(oe_ins)); __m512i fm=_mm512_sub_epi32(f,_mm512_set1_epi32(e_ins));
			__m512i Fcur=_mm512_max_epi32(fm,t2);
			_mm512_storeu_si512(&Hc[i0+1],h); _mm512_storeu_si512(&Ec[i0+1],Ecur); _mm512_storeu_si512(&Fc[i0+1],Fcur);
			if(z){
				// direction byte d — only assembled when producing a CIGAR
				__mmask16 mge=_mm512_cmpge_epi32_mask(mm,e); __m512i d=_mm512_mask_blend_epi32(mge,_mm512_set1_epi32(1),_mm512_setzero_si512());
				__mmask16 hgf=_mm512_cmpge_epi32_mask(h_me,f); d=_mm512_mask_blend_epi32(hgf,_mm512_set1_epi32(2),d);
				__mmask16 emgt=_mm512_cmpgt_epi32_mask(em,t1); d=_mm512_mask_or_epi32(d,emgt,d,_mm512_set1_epi32(1<<2));
				__mmask16 fmgt=_mm512_cmpgt_epi32_mask(fm,t2); d=_mm512_mask_or_epi32(d,fmgt,d,_mm512_set1_epi32(2<<4));
				if(interior){ int base=zbase_s+(i0-iS); _mm_storeu_si128((__m128i*)&z[base], _mm512_cvtepi32_epi8(d)); }
				else {
					int32_t dv[16],hv[16]; _mm512_storeu_si512(dv,d); _mm512_storeu_si512(hv,h);
					int lim=i_hi-i0; if(lim>15)lim=15;
					for(int k=0;k<=lim;++k){ if(!((inband>>k)&1))continue; int i=i0+k,j=s-i;
						z[zbase_s+(i-iS)]=(uint8_t)dv[k]; if(j==qlen-1)score=hv[k]; }
				}
			}
		}
	}
	if(z) ksw_wave_backtrack(z,zoff,ziS,tlen,qlen,w,n_cigar_,cigar_);
	return score;
}

/* --- AVX-512 int16 kernel: 32 lanes (vs the int32 kernel's 16), for the narrow-
 * to-moderate bands where 16-lane int32 has not yet reached its crossover. Entered
 * only when ksw_g2_wave16_safe proves every H/E/F stays in int16 range, so it is
 * byte-identical to the int32 scalar; otherwise the dispatcher falls back to the
 * int32 wave or scalar (both already byte-identical). Structure mirrors the int32
 * AVX-512 kernel exactly at half the element width. jv is derived from the scalar
 * (s-i0), not from s, because s=i+j can exceed int16 range on long reads even when
 * i and j individually do not. Crossover: 32-lane int16 beats scalar from w~8 and
 * overtakes the 16-lane int32 kernel at w>=16 (measured on Sapphire Rapids,
 * clang-19, 150bp/b=4). WMIN=10 keeps a comfortable margin on the weaker non-VBMI
 * table-lookup path (VBMI is faster at the low end). */
#define KSW_WAVE16_WMIN 10
static int ksw_g2_wave16(int qlen,const uint8_t*query,int tlen,const uint8_t*target,int m,const int8_t*mat,int o_del,int e_del,int o_ins,int e_ins,int w,int*n_cigar_,uint32_t**cigar_,KswWaveScratch&S){
	int oe_del=o_del+e_del,oe_ins=o_ins+e_ins;
	assert(m==5);             // AVX2 folds the score index as tv*5; NEON/non-VBMI tables hold <=32 entries — any m!=5 is silently wrong
	assert(n_cigar_&&cigar_); // score is captured only on the z!=0 path, so a score-only call would return MINUS_INF for qlen>w
	if(n_cigar_)*n_cigar_=0; bool want=(n_cigar_&&cigar_);
	uint8_t*z=want?S.zr.data():0; const int stride=S.stride,pad=S.pad;
	int*zoff=S.zoff.data(),*ziS=S.ziS.data(); long roff=0;
	int16_t*Hb=S.Hb16.data(),*Eb=S.Eb16.data(),*Fb=S.Fb16.data();
	auto Hs=[&](int d){return &Hb[(long)(((d%3)+3)%3)*stride];};
	auto Es=[&](int d){return &Eb[(long)(((d%2)+2)%2)*stride];};
	auto Fs=[&](int d){return &Fb[(long)(((d%2)+2)%2)*stride];};
	uint8_t*tpad=S.tpad.data(),*qpad=S.qpad.data();
	for(int i=0;i<tlen;++i)tpad[pad+i]=target[i]; for(int j=0;j<qlen;++j)qpad[pad+j]=query[j];
#if defined(__AVX512VBMI__)
	int8_t tab64[64]; memset(tab64,0,64); memcpy(tab64,mat,m*m<=64?m*m:64);
	__m512i TAB=_mm512_loadu_si512((const void*)tab64);
#else
	int8_t tlo[16],thi[16]; memset(tlo,0,16); memset(thi,0,16);
	for(int i=0;i<m*m&&i<16;++i)tlo[i]=mat[i]; for(int i=16;i<m*m&&i<32;++i)thi[i-16]=mat[i];
	const __m512i TABLO=_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)tlo));
	const __m512i TABHI=_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)thi));
#endif
	// REVIDX lane k = 31-k (reverses the forward query load); LANEID lane k = k.
	const __m512i REVIDX=_mm512_set_epi16(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31);
	const __m512i LANEID=_mm512_set_epi16(31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
	const __m512i VMINF=_mm512_set1_epi16(KSW_VMINF16);
	int score; if(qlen==0)score=0; else if(qlen<=w)score=-(o_ins+e_ins*qlen); else score=MINUS_INF;
	const int Sdim=(tlen-1)+(qlen-1)+1;
	for(int s=0;s<Sdim;++s){
		int16_t*Hc=Hs(s),*Ec=Es(s),*Fc=Fs(s); int16_t*Hd2=Hs(s-2),*Ep1=Es(s-1),*Fp1=Fs(s-1);
		int iS=s-(qlen-1); { int a=(s-w+1)>>1; if(a>iS)iS=a; } if(iS<0)iS=0;
		int iE=s;          { int a=(s+w)>>1;   if(a<iE)iE=a; } if(iE>tlen-1)iE=tlen-1;
		int i_hi=iE;
		int zbase_s=(int)roff;
		if(want){ zoff[s]=zbase_s; ziS[s]=iS; xassert((long)zbase_s+(iE>=iS?iE-iS+1:0)<=(long)S.zr.size(),"ksw wave: direction-store overrun"); if(iE>=iS) roff += (iE-iS+1); }
		for(int i0=iS;i0<=iE;i0+=32){
			bool interior=(i0>iS && i0+31<iE);
			__m512i Hdiag,e,f; __mmask32 inband; __m512i iv,jv,beg;
			if (interior) {
				Hdiag=_mm512_loadu_si512(&Hd2[i0]);
				e=_mm512_loadu_si512(&Ep1[i0]);
				f=_mm512_loadu_si512(&Fp1[i0+1]);
				inband=0xFFFFFFFFu;
			} else {
				iv=_mm512_add_epi16(_mm512_set1_epi16((int16_t)i0),LANEID);
				jv=_mm512_sub_epi16(_mm512_set1_epi16((int16_t)(s-i0)),LANEID);
				beg=_mm512_max_epi16(_mm512_setzero_si512(),_mm512_sub_epi16(iv,_mm512_set1_epi16((int16_t)w)));
				__m512i endc=_mm512_min_epi16(_mm512_set1_epi16((int16_t)qlen),_mm512_add_epi16(iv,_mm512_set1_epi16((int16_t)(w+1))));
				inband=_mm512_cmpge_epi16_mask(iv,_mm512_setzero_si512())&_mm512_cmplt_epi16_mask(iv,_mm512_set1_epi16((int16_t)tlen))
					&_mm512_cmpge_epi16_mask(jv,beg)&_mm512_cmplt_epi16_mask(jv,endc);
				Hdiag=_mm512_loadu_si512(&Hd2[i0]);
				__mmask32 i0m=_mm512_cmpeq_epi16_mask(iv,_mm512_setzero_si512());
				__mmask32 j0m=_mm512_cmpeq_epi16_mask(jv,_mm512_setzero_si512());
				__m512i Hi0=_mm512_mask_blend_epi16(j0m, _mm512_sub_epi16(_mm512_set1_epi16((int16_t)(-o_ins)),_mm512_mullo_epi16(_mm512_set1_epi16((int16_t)e_ins),jv)), _mm512_setzero_si512());
				__m512i Hj0=_mm512_sub_epi16(_mm512_set1_epi16((int16_t)(-o_del)),_mm512_mullo_epi16(_mm512_set1_epi16((int16_t)e_del),iv));
				Hdiag=_mm512_mask_blend_epi16(j0m,Hdiag,Hj0);
				Hdiag=_mm512_mask_blend_epi16(i0m,Hdiag,Hi0);
				__m512i begm1=_mm512_max_epi16(_mm512_setzero_si512(),_mm512_sub_epi16(iv,_mm512_set1_epi16((int16_t)(w+1))));
				__m512i endm1=_mm512_min_epi16(_mm512_set1_epi16((int16_t)qlen),_mm512_add_epi16(iv,_mm512_set1_epi16((int16_t)w)));
				__mmask32 evalid=_mm512_cmpge_epi16_mask(iv,_mm512_set1_epi16(1))&_mm512_cmpge_epi16_mask(jv,begm1)&_mm512_cmplt_epi16_mask(jv,endm1);
				e=_mm512_mask_blend_epi16(evalid,VMINF,_mm512_loadu_si512(&Ep1[i0]));
				__mmask32 freset=_mm512_cmpeq_epi16_mask(jv,beg);
				f=_mm512_mask_blend_epi16(freset,_mm512_loadu_si512(&Fp1[i0+1]),VMINF);
			}
			// score: idx = target[i]*m + query[s-i]; target contiguous, query forward-loaded then lane-reversed
			__m256i tb=_mm256_loadu_si256((const __m256i*)&tpad[pad+i0]);
			__m256i qb=_mm256_loadu_si256((const __m256i*)&qpad[pad+(s-i0-31)]);
			__m512i tv=_mm512_cvtepu8_epi16(tb);
			__m512i qv=_mm512_permutexvar_epi16(REVIDX,_mm512_cvtepu8_epi16(qb));
			__m512i idx=_mm512_add_epi16(_mm512_mullo_epi16(tv,_mm512_set1_epi16((int16_t)m)),qv);
#if defined(__AVX512VBMI__)
			__m512i res=_mm512_permutexvar_epi8(idx,TAB);                     // single vpermb (even byte = mat[idx])
#else
			__m512i slo=_mm512_shuffle_epi8(TABLO,idx);                       // VBMI-free 2-level vpshufb
			__m512i shi=_mm512_shuffle_epi8(TABHI,_mm512_sub_epi8(idx,_mm512_set1_epi8(16)));
			__m512i res=_mm512_mask_blend_epi16(_mm512_cmpgt_epi16_mask(idx,_mm512_set1_epi16(15)),slo,shi);
#endif
			__m512i Sv=_mm512_srai_epi16(_mm512_slli_epi16(res,8),8);         // extract+sign-extend low byte per lane
			__m512i mm=_mm512_add_epi16(Hdiag,Sv);
			__m512i h_me=_mm512_max_epi16(mm,e); __m512i h=_mm512_max_epi16(h_me,f);   // h/E'/F' via max (byte-identical on ties)
			__m512i t1=_mm512_sub_epi16(mm,_mm512_set1_epi16((int16_t)oe_del)); __m512i em=_mm512_sub_epi16(e,_mm512_set1_epi16((int16_t)e_del));
			__m512i Ecur=_mm512_max_epi16(em,t1);
			__m512i t2=_mm512_sub_epi16(mm,_mm512_set1_epi16((int16_t)oe_ins)); __m512i fm=_mm512_sub_epi16(f,_mm512_set1_epi16((int16_t)e_ins));
			__m512i Fcur=_mm512_max_epi16(fm,t2);
			_mm512_storeu_si512(&Hc[i0+1],h); _mm512_storeu_si512(&Ec[i0+1],Ecur); _mm512_storeu_si512(&Fc[i0+1],Fcur);
			if(z){
				// direction byte d — only assembled when producing a CIGAR
				__mmask32 mge=_mm512_cmpge_epi16_mask(mm,e); __m512i d=_mm512_mask_blend_epi16(mge,_mm512_set1_epi16(1),_mm512_setzero_si512());
				__mmask32 hgf=_mm512_cmpge_epi16_mask(h_me,f); d=_mm512_mask_blend_epi16(hgf,_mm512_set1_epi16(2),d);
				__mmask32 emgt=_mm512_cmpgt_epi16_mask(em,t1); d=_mm512_or_si512(d,_mm512_maskz_mov_epi16(emgt,_mm512_set1_epi16(1<<2)));
				__mmask32 fmgt=_mm512_cmpgt_epi16_mask(fm,t2); d=_mm512_or_si512(d,_mm512_maskz_mov_epi16(fmgt,_mm512_set1_epi16(2<<4)));
				if(interior){ int base=zbase_s+(i0-iS); _mm256_storeu_si256((__m256i*)&z[base], _mm512_cvtepi16_epi8(d)); }
				else {
					int16_t dv[32],hv[32]; _mm512_storeu_si512(dv,d); _mm512_storeu_si512(hv,h);
					int lim=i_hi-i0; if(lim>31)lim=31;
					for(int k=0;k<=lim;++k){ if(!((inband>>k)&1))continue; int i=i0+k,j=s-i;
						z[zbase_s+(i-iS)]=(uint8_t)dv[k]; if(j==qlen-1)score=hv[k]; }
				}
			}
		}
	}
	if(z) ksw_wave_backtrack(z,zoff,ziS,tlen,qlen,w,n_cigar_,cigar_);
	return score;
}

#elif defined(__AVX2__)
#define KSW_WAVE_WMIN 26
// >= mask for signed int32: !(a<b) = !(b>a). 2 ops (cmpgt+andnot) vs 3.
static inline __m256i cmge(__m256i a,__m256i b){ return _mm256_andnot_si256(_mm256_cmpgt_epi32(b,a),_mm256_set1_epi32(-1)); }
// select mask? X : Y  via bitwise (mask all-1s/0): AND/ANDNOT/OR — avoids port-5 blendv.
static inline __m256i sel(__m256i mask,__m256i X,__m256i Y){ return _mm256_or_si256(_mm256_and_si256(mask,X),_mm256_andnot_si256(mask,Y)); }

static int ksw_g2_wave(int qlen,const uint8_t*query,int tlen,const uint8_t*target,int m,const int8_t*mat,int o_del,int e_del,int o_ins,int e_ins,int w,int*n_cigar_,uint32_t**cigar_,KswWaveScratch&S){
	int oe_del=o_del+e_del,oe_ins=o_ins+e_ins;
	assert(m==5);             // AVX2 folds the score index as tv*5; NEON/non-VBMI tables hold <=32 entries — any m!=5 is silently wrong
	assert(n_cigar_&&cigar_); // score is captured only on the z!=0 path, so a score-only call would return MINUS_INF for qlen>w
	if(n_cigar_)*n_cigar_=0; bool want=(n_cigar_&&cigar_);
	uint8_t*z=want?S.zr.data():0; const int stride=S.stride,pad=S.pad;
	int*zoff=S.zoff.data(),*ziS=S.ziS.data(); long roff=0;
	int32_t*Hb=S.Hb.data(),*Eb=S.Eb.data(),*Fb=S.Fb.data();
	auto Hs=[&](int d){return &Hb[(long)(((d%3)+3)%3)*stride];}; auto Es=[&](int d){return &Eb[(long)(((d%2)+2)%2)*stride];}; auto Fs=[&](int d){return &Fb[(long)(((d%2)+2)%2)*stride];};
	uint8_t*tpad=S.tpad.data(),*qrev=S.qrev.data();
	for(int i=0;i<tlen;++i)tpad[pad+i]=target[i];
	for(int j=0;j<qlen;++j)qrev[pad+j]=query[qlen-1-j];   // reversed query: score load becomes contiguous (no per-chunk permute)
	// VBMI-free score table: two 16-byte halves broadcast to both 128-lanes for 2-level vpshufb
	int8_t tlo[16],thi[16]; memset(tlo,0,16); memset(thi,0,16);
	for(int i=0;i<m*m&&i<16;++i)tlo[i]=mat[i]; for(int i=16;i<m*m&&i<32;++i)thi[i-16]=mat[i];
	const __m256i TABLO=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tlo));
	const __m256i TABHI=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)thi));
	int32_t score; if(qlen==0)score=0; else if(qlen<=w)score=-(o_ins+e_ins*qlen); else score=MINUS_INF;
	const __m256i VMINF=_mm256_set1_epi32(MINUS_INF);
	const __m256i LANEID=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
	const int Sdim=(tlen-1)+(qlen-1)+1;
	for(int s=0;s<Sdim;++s){
		int32_t*Hc=Hs(s),*Ec=Es(s),*Fc=Fs(s); int32_t*Hd2=Hs(s-2),*Ep1=Es(s-1),*Fp1=Fs(s-1);
		int iS=s-(qlen-1); {int a=(s-w+1)>>1; if(a>iS)iS=a;} if(iS<0)iS=0;
		int iE=s; {int a=(s+w)>>1; if(a<iE)iE=a;} if(iE>tlen-1)iE=tlen-1;
		int i_hi=iE; int zbase_s=(int)roff; if(want){zoff[s]=zbase_s;ziS[s]=iS; xassert((long)zbase_s+(iE>=iS?iE-iS+1:0)<=(long)S.zr.size(),"ksw wave: direction-store overrun"); if(iE>=iS)roff+=(iE-iS+1);}
		for(int i0=iS;i0<=iE;i0+=8){
			bool interior=(i0>iS && i0+7<iE);
			__m256i Hdiag,e,f,inband,iv,jv,beg;
			if(interior){
				Hdiag=_mm256_loadu_si256((const __m256i*)&Hd2[i0]);
				e=_mm256_loadu_si256((const __m256i*)&Ep1[i0]);
				f=_mm256_loadu_si256((const __m256i*)&Fp1[i0+1]);
				inband=_mm256_set1_epi32(-1);
			} else {
				iv=_mm256_add_epi32(_mm256_set1_epi32(i0),LANEID);
				jv=_mm256_sub_epi32(_mm256_set1_epi32(s),iv);
				beg=_mm256_max_epi32(_mm256_setzero_si256(),_mm256_sub_epi32(iv,_mm256_set1_epi32(w)));
				__m256i endc=_mm256_min_epi32(_mm256_set1_epi32(qlen),_mm256_add_epi32(iv,_mm256_set1_epi32(w+1)));
				inband=_mm256_and_si256(_mm256_and_si256(cmge(iv,_mm256_setzero_si256()),_mm256_cmpgt_epi32(_mm256_set1_epi32(tlen),iv)),
					_mm256_and_si256(cmge(jv,beg),_mm256_cmpgt_epi32(endc,jv)));
				Hdiag=_mm256_loadu_si256((const __m256i*)&Hd2[i0]);
				__m256i i0m=_mm256_cmpeq_epi32(iv,_mm256_setzero_si256());
				__m256i j0m=_mm256_cmpeq_epi32(jv,_mm256_setzero_si256());
				__m256i Hi0=sel(j0m,_mm256_setzero_si256(),_mm256_sub_epi32(_mm256_set1_epi32(-o_ins),_mm256_mullo_epi32(_mm256_set1_epi32(e_ins),jv)));
				__m256i Hj0=_mm256_sub_epi32(_mm256_set1_epi32(-o_del),_mm256_mullo_epi32(_mm256_set1_epi32(e_del),iv));
				Hdiag=sel(j0m,Hj0,Hdiag); Hdiag=sel(i0m,Hi0,Hdiag);
				__m256i begm1=_mm256_max_epi32(_mm256_setzero_si256(),_mm256_sub_epi32(iv,_mm256_set1_epi32(w+1)));
				__m256i endm1=_mm256_min_epi32(_mm256_set1_epi32(qlen),_mm256_add_epi32(iv,_mm256_set1_epi32(w)));
				__m256i evalid=_mm256_and_si256(cmge(iv,_mm256_set1_epi32(1)),_mm256_and_si256(cmge(jv,begm1),_mm256_cmpgt_epi32(endm1,jv)));
				e=sel(evalid,_mm256_loadu_si256((const __m256i*)&Ep1[i0]),VMINF);
				__m256i freset=_mm256_cmpeq_epi32(jv,beg);
				f=sel(freset,VMINF,_mm256_loadu_si256((const __m256i*)&Fp1[i0+1]));
			}
			// score: idx = target[i]*5 + query[s-i]. target contiguous; query from the
			// pre-reversed buffer -> plain contiguous load (no per-chunk cross-lane permute).
			__m256i tv=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)&tpad[pad+i0]));
			__m256i qv=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)&qrev[pad+(qlen-1-s+i0)]));
			__m256i tv5=_mm256_add_epi32(_mm256_slli_epi32(tv,2),tv);   // tv*5 = (tv<<2)+tv (m==5), off the score dep chain
			__m256i idx=_mm256_add_epi32(tv5,qv);
			// 2-level vpshufb table lookup (idx 0..24): low table for idx<16, high for idx>=16
			__m256i slo=_mm256_shuffle_epi8(TABLO,idx);
			__m256i shi=_mm256_shuffle_epi8(TABHI,_mm256_sub_epi8(idx,_mm256_set1_epi8(16)));
			__m256i ge16=_mm256_cmpgt_epi32(idx,_mm256_set1_epi32(15));
			__m256i res=_mm256_or_si256(_mm256_and_si256(ge16,shi),_mm256_andnot_si256(ge16,slo));
			__m256i Sv=_mm256_srai_epi32(_mm256_slli_epi32(res,24),24);  // extract+sign-extend low byte per lane
			__m256i mm=_mm256_add_epi32(Hdiag,Sv);
			__m256i h_me=_mm256_max_epi32(mm,e); __m256i h=_mm256_max_epi32(h_me,f);
			__m256i t1=_mm256_sub_epi32(mm,_mm256_set1_epi32(oe_del)); __m256i em=_mm256_sub_epi32(e,_mm256_set1_epi32(e_del));
			__m256i Ecur=_mm256_max_epi32(em,t1);
			__m256i t2=_mm256_sub_epi32(mm,_mm256_set1_epi32(oe_ins)); __m256i fm=_mm256_sub_epi32(f,_mm256_set1_epi32(e_ins));
			__m256i Fcur=_mm256_max_epi32(fm,t2);
			_mm256_storeu_si256((__m256i*)&Hc[i0+1],h); _mm256_storeu_si256((__m256i*)&Ec[i0+1],Ecur); _mm256_storeu_si256((__m256i*)&Fc[i0+1],Fcur);
			if(z){
				__m256i mge=cmge(mm,e); __m256i d=sel(mge,_mm256_setzero_si256(),_mm256_set1_epi32(1));
				__m256i hgf=cmge(h_me,f); d=sel(hgf,d,_mm256_set1_epi32(2));
				__m256i emgt=_mm256_cmpgt_epi32(em,t1); d=_mm256_or_si256(d,_mm256_and_si256(emgt,_mm256_set1_epi32(1<<2)));
				__m256i fmgt=_mm256_cmpgt_epi32(fm,t2); d=_mm256_or_si256(d,_mm256_and_si256(fmgt,_mm256_set1_epi32(2<<4)));
				if(interior){ int base=zbase_s+(i0-iS);
					__m128i lo=_mm256_castsi256_si128(d),hi=_mm256_extracti128_si256(d,1);
					__m128i p16=_mm_packus_epi32(lo,hi); __m128i p8=_mm_packus_epi16(p16,p16);
					_mm_storel_epi64((__m128i*)&z[base],p8);
				} else {
					int32_t ib[8],dv[8],hv[8]; _mm256_storeu_si256((__m256i*)ib,inband); _mm256_storeu_si256((__m256i*)dv,d); _mm256_storeu_si256((__m256i*)hv,h);
					for(int k=0;k<8;++k){ int i=i0+k; if(i>i_hi)break; if(!ib[k])continue; int j=s-i;
						z[zbase_s+(i-iS)]=(uint8_t)dv[k]; if(j==qlen-1)score=hv[k]; }
				}
			}
		}
	}
	if(z) ksw_wave_backtrack(z,zoff,ziS,tlen,qlen,w,n_cigar_,cigar_);
	return score;
}

/* --- AVX2 int16 kernel: 16 lanes (vs the int32 kernel's 8), for the narrow-to-
 * moderate bands where 8-lane int32 has not yet reached its crossover. Entered
 * only when ksw_g2_wave16_safe proves every H/E/F stays in int16 range, so it is
 * byte-identical to the int32 scalar; otherwise the dispatcher falls back to the
 * int32 wave or scalar (both already byte-identical). Structure mirrors the int32
 * AVX2 kernel exactly at half the element width (pre-reversed query buffer, 2-level
 * vpshufb table, bitwise select). jv is derived from the scalar (s-i0), not from
 * s, because s=i+j can exceed int16 range on long reads even when i,j do not.
 * Crossover: 16-lane int16 only overtakes the 8-lane int32 kernel at wide bands
 * (w>=33 on Sapphire Rapids, clang-19, 150bp/b=4); in the w~20-32 band int32 is
 * ~1% faster, so WMIN=33 keeps int16 to the region where it is a clear net win. */
#define KSW_WAVE16_WMIN 33
// >= mask for signed int16: !(a<b) = !(b>a).
static inline __m256i cmge16(__m256i a,__m256i b){ return _mm256_andnot_si256(_mm256_cmpgt_epi16(b,a),_mm256_set1_epi32(-1)); }

static int ksw_g2_wave16(int qlen,const uint8_t*query,int tlen,const uint8_t*target,int m,const int8_t*mat,int o_del,int e_del,int o_ins,int e_ins,int w,int*n_cigar_,uint32_t**cigar_,KswWaveScratch&S){
	int oe_del=o_del+e_del,oe_ins=o_ins+e_ins;
	assert(m==5);             // AVX2 folds the score index as tv*5; NEON/non-VBMI tables hold <=32 entries — any m!=5 is silently wrong
	assert(n_cigar_&&cigar_); // score is captured only on the z!=0 path, so a score-only call would return MINUS_INF for qlen>w
	if(n_cigar_)*n_cigar_=0; bool want=(n_cigar_&&cigar_);
	uint8_t*z=want?S.zr.data():0; const int stride=S.stride,pad=S.pad;
	int*zoff=S.zoff.data(),*ziS=S.ziS.data(); long roff=0;
	int16_t*Hb=S.Hb16.data(),*Eb=S.Eb16.data(),*Fb=S.Fb16.data();
	auto Hs=[&](int d){return &Hb[(long)(((d%3)+3)%3)*stride];}; auto Es=[&](int d){return &Eb[(long)(((d%2)+2)%2)*stride];}; auto Fs=[&](int d){return &Fb[(long)(((d%2)+2)%2)*stride];};
	uint8_t*tpad=S.tpad.data(),*qrev=S.qrev.data();
	for(int i=0;i<tlen;++i)tpad[pad+i]=target[i];
	for(int j=0;j<qlen;++j)qrev[pad+j]=query[qlen-1-j];   // reversed query: score load becomes contiguous (no per-chunk permute)
	int8_t tlo[16],thi[16]; memset(tlo,0,16); memset(thi,0,16);
	for(int i=0;i<m*m&&i<16;++i)tlo[i]=mat[i]; for(int i=16;i<m*m&&i<32;++i)thi[i-16]=mat[i];
	const __m256i TABLO=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tlo));
	const __m256i TABHI=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)thi));
	int score; if(qlen==0)score=0; else if(qlen<=w)score=-(o_ins+e_ins*qlen); else score=MINUS_INF;
	const __m256i VMINF=_mm256_set1_epi16(KSW_VMINF16);
	const __m256i LANEID=_mm256_setr_epi16(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
	const int Sdim=(tlen-1)+(qlen-1)+1;
	for(int s=0;s<Sdim;++s){
		int16_t*Hc=Hs(s),*Ec=Es(s),*Fc=Fs(s); int16_t*Hd2=Hs(s-2),*Ep1=Es(s-1),*Fp1=Fs(s-1);
		int iS=s-(qlen-1); {int a=(s-w+1)>>1; if(a>iS)iS=a;} if(iS<0)iS=0;
		int iE=s; {int a=(s+w)>>1; if(a<iE)iE=a;} if(iE>tlen-1)iE=tlen-1;
		int i_hi=iE; int zbase_s=(int)roff; if(want){zoff[s]=zbase_s;ziS[s]=iS; xassert((long)zbase_s+(iE>=iS?iE-iS+1:0)<=(long)S.zr.size(),"ksw wave: direction-store overrun"); if(iE>=iS)roff+=(iE-iS+1);}
		for(int i0=iS;i0<=iE;i0+=16){
			bool interior=(i0>iS && i0+15<iE);
			__m256i Hdiag,e,f,inband,iv,jv,beg;
			if(interior){
				Hdiag=_mm256_loadu_si256((const __m256i*)&Hd2[i0]);
				e=_mm256_loadu_si256((const __m256i*)&Ep1[i0]);
				f=_mm256_loadu_si256((const __m256i*)&Fp1[i0+1]);
				inband=_mm256_set1_epi16(-1);
			} else {
				iv=_mm256_add_epi16(_mm256_set1_epi16((int16_t)i0),LANEID);
				jv=_mm256_sub_epi16(_mm256_set1_epi16((int16_t)(s-i0)),LANEID);
				beg=_mm256_max_epi16(_mm256_setzero_si256(),_mm256_sub_epi16(iv,_mm256_set1_epi16((int16_t)w)));
				__m256i endc=_mm256_min_epi16(_mm256_set1_epi16((int16_t)qlen),_mm256_add_epi16(iv,_mm256_set1_epi16((int16_t)(w+1))));
				inband=_mm256_and_si256(_mm256_and_si256(cmge16(iv,_mm256_setzero_si256()),_mm256_cmpgt_epi16(_mm256_set1_epi16((int16_t)tlen),iv)),
					_mm256_and_si256(cmge16(jv,beg),_mm256_cmpgt_epi16(endc,jv)));
				Hdiag=_mm256_loadu_si256((const __m256i*)&Hd2[i0]);
				__m256i i0m=_mm256_cmpeq_epi16(iv,_mm256_setzero_si256());
				__m256i j0m=_mm256_cmpeq_epi16(jv,_mm256_setzero_si256());
				__m256i Hi0=sel(j0m,_mm256_setzero_si256(),_mm256_sub_epi16(_mm256_set1_epi16((int16_t)(-o_ins)),_mm256_mullo_epi16(_mm256_set1_epi16((int16_t)e_ins),jv)));
				__m256i Hj0=_mm256_sub_epi16(_mm256_set1_epi16((int16_t)(-o_del)),_mm256_mullo_epi16(_mm256_set1_epi16((int16_t)e_del),iv));
				Hdiag=sel(j0m,Hj0,Hdiag); Hdiag=sel(i0m,Hi0,Hdiag);
				__m256i begm1=_mm256_max_epi16(_mm256_setzero_si256(),_mm256_sub_epi16(iv,_mm256_set1_epi16((int16_t)(w+1))));
				__m256i endm1=_mm256_min_epi16(_mm256_set1_epi16((int16_t)qlen),_mm256_add_epi16(iv,_mm256_set1_epi16((int16_t)w)));
				__m256i evalid=_mm256_and_si256(cmge16(iv,_mm256_set1_epi16(1)),_mm256_and_si256(cmge16(jv,begm1),_mm256_cmpgt_epi16(endm1,jv)));
				e=sel(evalid,_mm256_loadu_si256((const __m256i*)&Ep1[i0]),VMINF);
				__m256i freset=_mm256_cmpeq_epi16(jv,beg);
				f=sel(freset,VMINF,_mm256_loadu_si256((const __m256i*)&Fp1[i0+1]));
			}
			// score: target contiguous; query from the pre-reversed buffer -> contiguous load.
			__m256i tv=_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&tpad[pad+i0]));
			__m256i qv=_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&qrev[pad+(qlen-1-s+i0)]));
			__m256i tv5=_mm256_add_epi16(_mm256_slli_epi16(tv,2),tv);   // tv*5 = (tv<<2)+tv (m==5), off the score dep chain
			__m256i idx=_mm256_add_epi16(tv5,qv);
			__m256i slo=_mm256_shuffle_epi8(TABLO,idx);                 // 2-level vpshufb (idx 0..24)
			__m256i shi=_mm256_shuffle_epi8(TABHI,_mm256_sub_epi8(idx,_mm256_set1_epi8(16)));
			__m256i ge16=_mm256_cmpgt_epi16(idx,_mm256_set1_epi16(15));
			__m256i res=sel(ge16,shi,slo);
			__m256i Sv=_mm256_srai_epi16(_mm256_slli_epi16(res,8),8);   // extract+sign-extend low byte per lane
			__m256i mm=_mm256_add_epi16(Hdiag,Sv);
			__m256i h_me=_mm256_max_epi16(mm,e); __m256i h=_mm256_max_epi16(h_me,f);
			__m256i t1=_mm256_sub_epi16(mm,_mm256_set1_epi16((int16_t)oe_del)); __m256i em=_mm256_sub_epi16(e,_mm256_set1_epi16((int16_t)e_del));
			__m256i Ecur=_mm256_max_epi16(em,t1);
			__m256i t2=_mm256_sub_epi16(mm,_mm256_set1_epi16((int16_t)oe_ins)); __m256i fm=_mm256_sub_epi16(f,_mm256_set1_epi16((int16_t)e_ins));
			__m256i Fcur=_mm256_max_epi16(fm,t2);
			_mm256_storeu_si256((__m256i*)&Hc[i0+1],h); _mm256_storeu_si256((__m256i*)&Ec[i0+1],Ecur); _mm256_storeu_si256((__m256i*)&Fc[i0+1],Fcur);
			if(z){
				__m256i mge=cmge16(mm,e); __m256i d=sel(mge,_mm256_setzero_si256(),_mm256_set1_epi16(1));
				__m256i hgf=cmge16(h_me,f); d=sel(hgf,d,_mm256_set1_epi16(2));
				__m256i emgt=_mm256_cmpgt_epi16(em,t1); d=_mm256_or_si256(d,_mm256_and_si256(emgt,_mm256_set1_epi16(1<<2)));
				__m256i fmgt=_mm256_cmpgt_epi16(fm,t2); d=_mm256_or_si256(d,_mm256_and_si256(fmgt,_mm256_set1_epi16(2<<4)));
				if(interior){ int base=zbase_s+(i0-iS);
					__m128i lo=_mm256_castsi256_si128(d),hi=_mm256_extracti128_si256(d,1);
					_mm_storeu_si128((__m128i*)&z[base], _mm_packus_epi16(lo,hi));   // 16 contiguous direction bytes
				} else {
					int16_t ib[16],dv[16],hv[16]; _mm256_storeu_si256((__m256i*)ib,inband); _mm256_storeu_si256((__m256i*)dv,d); _mm256_storeu_si256((__m256i*)hv,h);
					for(int k=0;k<16;++k){ int i=i0+k; if(i>i_hi)break; if(!ib[k])continue; int j=s-i;
						z[zbase_s+(i-iS)]=(uint8_t)dv[k]; if(j==qlen-1)score=hv[k]; }
				}
			}
		}
	}
	if(z) ksw_wave_backtrack(z,zoff,ziS,tlen,qlen,w,n_cigar_,cigar_);
	return score;
}

#elif defined(__aarch64__)
#define KSW_WAVE_WMIN 52
static inline int32x4_t rev4(int32x4_t v){ int32x4_t r=vrev64q_s32(v); return vextq_s32(r,r,2); }

static int ksw_g2_wave(int qlen,const uint8_t*query,int tlen,const uint8_t*target,int m,const int8_t*mat,int o_del,int e_del,int o_ins,int e_ins,int w,int*n_cigar_,uint32_t**cigar_,KswWaveScratch&S){
	int oe_del=o_del+e_del,oe_ins=o_ins+e_ins;
	assert(m==5);             // AVX2 folds the score index as tv*5; NEON/non-VBMI tables hold <=32 entries — any m!=5 is silently wrong
	assert(n_cigar_&&cigar_); // score is captured only on the z!=0 path, so a score-only call would return MINUS_INF for qlen>w
	if(n_cigar_)*n_cigar_=0; bool want=(n_cigar_&&cigar_);
	uint8_t*z = want? S.zr.data():0;
	int*zoff=S.zoff.data(),*ziS=S.ziS.data(); long roff=0;
	const int stride=S.stride, pad=S.pad;
	int32_t*Hb=S.Hb.data(),*Eb=S.Eb.data(),*Fb=S.Fb.data();
#if defined(__APPLE__)
	// Apple-silicon fast path: carry the H/E/F ring slots as running loop state (advanced
	// in the for-increment below) instead of recomputing them per diagonal via the #else
	// path's double-modulo lambdas. Byte-identical -- it yields the same physical slot for
	// each (s,ring) as ((d%3+3)%3)/((d%2+2)%2); hc/hd2 walk the size-3 H ring, efc/efp the
	// size-2 E/F rings, initialized to the s=0 slots. Measured ~+1.5-2.7% on the isolated
	// kernel with Apple clang across two Apple-silicon machines, and a wash with a modern
	// Linux clang (which already optimizes the modulo form), so the win is gated here. The
	// gate is really macOS-on-aarch64: __APPLE__ inside this __aarch64__-only kernel means
	// Apple silicon (do NOT use APPLE_SILICON -- simd_compat.h defines it 1 for every
	// aarch64 build, so it is true on Graviton too). It stands in for Apple's system clang,
	// the toolchain macOS ships; a non-Apple clang on macOS still takes this path but gets
	// the same harmless wash (still byte-identical), not a regression.
	// KEEP IN SYNC with the #else lambdas below: both encode the SAME ring sizes (H=3,
	// E/F=2); a ring-size change must update both forms or the two builds silently diverge.
	int hc=0,hd2=1,efc=0,efp=1;
#else
	// Portable form (every non-Apple build): recompute the ring slot per access via modulo.
	// KEEP IN SYNC with the Apple incremental form above (same H=3, E/F=2 ring sizes).
	auto Hs=[&](int d){return &Hb[(long)(((d%3)+3)%3)*stride];};
	auto Es=[&](int d){return &Eb[(long)(((d%2)+2)%2)*stride];};
	auto Fs=[&](int d){return &Fb[(long)(((d%2)+2)%2)*stride];};
#endif
	uint8_t*tpad=S.tpad.data(),*qpad=S.qpad.data();
	for(int i=0;i<tlen;++i)tpad[pad+i]=target[i];
	for(int j=0;j<qlen;++j)qpad[pad+j]=query[j];
	int8_t tab32[32]; memset(tab32,0,32); memcpy(tab32,mat,m*m<=32?m*m:32);
	uint8x16x2_t TAB; TAB.val[0]=vld1q_u8((const uint8_t*)tab32); TAB.val[1]=vld1q_u8((const uint8_t*)tab32+16);
	int32_t score; if(qlen==0)score=0; else if(qlen<=w)score=-(o_ins+e_ins*qlen); else score=MINUS_INF;
	const int32x4_t VMINF=vdupq_n_s32(MINUS_INF); const int32x4_t LANEID={0,1,2,3};
	const int Sdim=(tlen-1)+(qlen-1)+1;
#if defined(__APPLE__)
	for(int s=0;s<Sdim;++s, hc=(hc==2?0:hc+1), hd2=(hd2==2?0:hd2+1), efc^=1, efp^=1){
		int32_t*Hc=&Hb[(long)hc*stride],*Ec=&Eb[(long)efc*stride],*Fc=&Fb[(long)efc*stride]; int32_t*Hd2=&Hb[(long)hd2*stride],*Ep1=&Eb[(long)efp*stride],*Fp1=&Fb[(long)efp*stride];
#else
	for(int s=0;s<Sdim;++s){
		int32_t*Hc=Hs(s),*Ec=Es(s),*Fc=Fs(s); int32_t*Hd2=Hs(s-2),*Ep1=Es(s-1),*Fp1=Fs(s-1);
#endif
		int i_lo=s-(qlen-1); { int a=(s-w+1)>>1; if(a>i_lo)i_lo=a; } if(i_lo<0)i_lo=0;
		int i_hi=s; { int a=(s+w)>>1; if(a<i_hi)i_hi=a; } if(i_hi>tlen-1)i_hi=tlen-1;
		int zbase_s=(int)roff;                 // diagonal-major offset for this diagonal
		if(want){ zoff[s]=zbase_s; ziS[s]=i_lo; xassert((long)zbase_s+(i_hi>=i_lo?i_hi-i_lo+1:0)<=(long)S.zr.size(),"ksw wave: direction-store overrun"); if(i_hi>=i_lo) roff += (i_hi-i_lo+1); }
		for(int i0=i_lo;i0<=i_hi;i0+=4){
			bool interior = (i0>i_lo && i0+3<i_hi);
			int32x4_t Hdiag,e,f; uint32x4_t inband; int32x4_t iv,jv,beg;
			if (interior) {
				// interior chunk: whole tight band -> no i==0/j==0/e-invalid/f-reset lanes
				Hdiag=vld1q_s32(&Hd2[i0]);
				e=vld1q_s32(&Ep1[i0]);
				f=vld1q_s32(&Fp1[i0+1]);
				inband=vdupq_n_u32(0xFFFFFFFFu);
			} else {
				iv=vaddq_s32(vdupq_n_s32(i0),LANEID);
				jv=vsubq_s32(vdupq_n_s32(s),iv);
				beg=vmaxq_s32(vdupq_n_s32(0),vsubq_s32(iv,vdupq_n_s32(w)));
				int32x4_t endc=vminq_s32(vdupq_n_s32(qlen),vaddq_s32(iv,vdupq_n_s32(w+1)));
				inband=vandq_u32(vandq_u32(vcgeq_s32(iv,vdupq_n_s32(0)),vcltq_s32(iv,vdupq_n_s32(tlen))),
					vandq_u32(vcgeq_s32(jv,beg),vcltq_s32(jv,endc)));
				Hdiag=vld1q_s32(&Hd2[i0]);
				uint32x4_t i0m=vceqq_s32(iv,vdupq_n_s32(0)); uint32x4_t j0m=vceqq_s32(jv,vdupq_n_s32(0));
				int32x4_t Hi0=vbslq_s32(j0m,vdupq_n_s32(0),vsubq_s32(vdupq_n_s32(-o_ins),vmulq_s32(vdupq_n_s32(e_ins),jv)));
				int32x4_t Hj0=vsubq_s32(vdupq_n_s32(-o_del),vmulq_s32(vdupq_n_s32(e_del),iv));
				Hdiag=vbslq_s32(j0m,Hj0,Hdiag); Hdiag=vbslq_s32(i0m,Hi0,Hdiag);
				int32x4_t begm1=vmaxq_s32(vdupq_n_s32(0),vsubq_s32(iv,vdupq_n_s32(w+1)));
				int32x4_t endm1=vminq_s32(vdupq_n_s32(qlen),vaddq_s32(iv,vdupq_n_s32(w)));
				uint32x4_t evalid=vandq_u32(vcgeq_s32(iv,vdupq_n_s32(1)),vandq_u32(vcgeq_s32(jv,begm1),vcltq_s32(jv,endm1)));
				e=vbslq_s32(evalid,vld1q_s32(&Ep1[i0]),VMINF);
				uint32x4_t freset=vceqq_s32(jv,beg);
				f=vbslq_s32(freset,VMINF,vld1q_s32(&Fp1[i0+1]));
			}
			uint8x8_t tb=vld1_u8(&tpad[pad+i0]); uint8x8_t qb=vld1_u8(&qpad[pad+(s-i0-3)]);
			int32x4_t tv=vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(tb))));
			int32x4_t qv=rev4(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(qb)))));
			int32x4_t idx=vaddq_s32(vmulq_s32(tv,vdupq_n_s32(m)),qv);
			int8x16_t sres=vreinterpretq_s8_u8(vqtbl2q_u8(TAB,vreinterpretq_u8_s32(idx)));
			int32x4_t Sv=vshrq_n_s32(vshlq_n_s32(vreinterpretq_s32_s8(sres),24),24);
			int32x4_t mm=vaddq_s32(Hdiag,Sv);
			int32x4_t h_me=vmaxq_s32(mm,e); int32x4_t h=vmaxq_s32(h_me,f);   // h/E'/F' via max: byte-identical on ties
			int32x4_t t1=vsubq_s32(mm,vdupq_n_s32(oe_del)); int32x4_t em=vsubq_s32(e,vdupq_n_s32(e_del));
			int32x4_t Ecur=vmaxq_s32(em,t1);
			int32x4_t t2=vsubq_s32(mm,vdupq_n_s32(oe_ins)); int32x4_t fm=vsubq_s32(f,vdupq_n_s32(e_ins));
			int32x4_t Fcur=vmaxq_s32(fm,t2);
			vst1q_s32(&Hc[i0+1],h); vst1q_s32(&Ec[i0+1],Ecur); vst1q_s32(&Fc[i0+1],Fcur);
			if(z){
				// direction byte d — only assembled when producing a CIGAR
				uint32x4_t mge=vcgeq_s32(mm,e); int32x4_t d=vbslq_s32(mge,vdupq_n_s32(0),vdupq_n_s32(1));
				uint32x4_t hgf=vcgeq_s32(h_me,f); d=vbslq_s32(hgf,d,vdupq_n_s32(2));
				uint32x4_t emgt=vcgtq_s32(em,t1); d=vorrq_s32(d,vbslq_s32(emgt,vdupq_n_s32(1<<2),vdupq_n_s32(0)));
				uint32x4_t fmgt=vcgtq_s32(fm,t2); d=vorrq_s32(d,vbslq_s32(fmgt,vdupq_n_s32(2<<4),vdupq_n_s32(0)));
				if(interior){ int base=zbase_s+(i0-i_lo);
					uint16x4_t d16=vmovn_u32(vreinterpretq_u32_s32(d));
					uint8x8_t d8=vmovn_u16(vcombine_u16(d16,d16));
					uint32_t four=vget_lane_u32(vreinterpret_u32_u8(d8),0);
					memcpy(&z[base],&four,4);
				} else {
					uint32_t ib[4]; vst1q_u32(ib,inband); int32_t dv[4]; vst1q_s32(dv,d); int32_t hv[4]; vst1q_s32(hv,h);
					for(int k=0;k<4;++k){ int i=i0+k; if(i>i_hi)break; if(!ib[k])continue; int j=s-i;
						z[zbase_s+(i-i_lo)]=(uint8_t)dv[k];
						if(j==qlen-1)score=hv[k]; }
				}
			}
		}
	}
	if(z) ksw_wave_backtrack(z,zoff,ziS,tlen,qlen,w,n_cigar_,cigar_);
	return score;
}

/* --- NEON int16 kernel: 8 lanes (vs the int32 kernel's 4), for the narrow-to-
 * moderate bands where 4-lane int32 could not beat scalar. Entered only when
 * ksw_g2_wave16_safe proves every H/E/F stays in int16 range, so it is byte-
 * identical to the int32 scalar; otherwise the dispatcher falls back to the
 * int32 wave or scalar (both already byte-identical). Structure mirrors the
 * int32 kernel exactly at half the element width; no 2x unroll (the int32
 * experiment showed the ceiling is the diagonal recurrence, not intra-diagonal
 * ILP). jv is derived from the scalar (s-i0), not from s, because s=i+j can
 * exceed int16 range on long reads even when i and j individually do not.
 * Crossover: the 8-lane int16 kernel becomes a net win over scalar from w~16
 * (measured on Graviton4, clang-19, 150bp/b=4); narrower bands stay scalar, so
 * WMIN=16. */
#define KSW_WAVE16_WMIN 16
static inline int16x8_t rev8(int16x8_t v){ int16x8_t r=vrev64q_s16(v); return vextq_s16(r,r,4); }

static int ksw_g2_wave16(int qlen,const uint8_t*query,int tlen,const uint8_t*target,int m,const int8_t*mat,int o_del,int e_del,int o_ins,int e_ins,int w,int*n_cigar_,uint32_t**cigar_,KswWaveScratch&S){
	int oe_del=o_del+e_del,oe_ins=o_ins+e_ins;
	assert(m==5);             // AVX2 folds the score index as tv*5; NEON/non-VBMI tables hold <=32 entries — any m!=5 is silently wrong
	assert(n_cigar_&&cigar_); // score is captured only on the z!=0 path, so a score-only call would return MINUS_INF for qlen>w
	if(n_cigar_)*n_cigar_=0; bool want=(n_cigar_&&cigar_);
	uint8_t*z = want? S.zr.data():0;
	int*zoff=S.zoff.data(),*ziS=S.ziS.data(); long roff=0;
	const int stride=S.stride, pad=S.pad;
	int16_t*Hb=S.Hb16.data(),*Eb=S.Eb16.data(),*Fb=S.Fb16.data();
#if defined(__APPLE__)
	// Apple-silicon fast path: incremental ring-slot tracking (full rationale, gating, and
	// the macOS-on-aarch64 / non-Apple-clang notes are in the int32 NEON kernel above).
	// Byte-identical; measured ~+1.5-2.7% on Apple clang, a wash on a modern Linux clang.
	// KEEP IN SYNC with the #else lambdas below (same H=3, E/F=2 ring sizes).
	int hc=0,hd2=1,efc=0,efp=1;
#else
	// Portable form; KEEP IN SYNC with the Apple incremental form above (same H=3, E/F=2).
	auto Hs=[&](int d){return &Hb[(long)(((d%3)+3)%3)*stride];};
	auto Es=[&](int d){return &Eb[(long)(((d%2)+2)%2)*stride];};
	auto Fs=[&](int d){return &Fb[(long)(((d%2)+2)%2)*stride];};
#endif
	uint8_t*tpad=S.tpad.data(),*qpad=S.qpad.data();
	for(int i=0;i<tlen;++i)tpad[pad+i]=target[i];
	for(int j=0;j<qlen;++j)qpad[pad+j]=query[j];
	int8_t tab32[32]; memset(tab32,0,32); memcpy(tab32,mat,m*m<=32?m*m:32);
	uint8x16x2_t TAB; TAB.val[0]=vld1q_u8((const uint8_t*)tab32); TAB.val[1]=vld1q_u8((const uint8_t*)tab32+16);
	int score; if(qlen==0)score=0; else if(qlen<=w)score=-(o_ins+e_ins*qlen); else score=MINUS_INF;
	const int16x8_t VMINF=vdupq_n_s16(KSW_VMINF16); const int16x8_t LANEID={0,1,2,3,4,5,6,7};
	const int Sdim=(tlen-1)+(qlen-1)+1;
#if defined(__APPLE__)
	for(int s=0;s<Sdim;++s, hc=(hc==2?0:hc+1), hd2=(hd2==2?0:hd2+1), efc^=1, efp^=1){
		int16_t*Hc=&Hb[(long)hc*stride],*Ec=&Eb[(long)efc*stride],*Fc=&Fb[(long)efc*stride]; int16_t*Hd2=&Hb[(long)hd2*stride],*Ep1=&Eb[(long)efp*stride],*Fp1=&Fb[(long)efp*stride];
#else
	for(int s=0;s<Sdim;++s){
		int16_t*Hc=Hs(s),*Ec=Es(s),*Fc=Fs(s); int16_t*Hd2=Hs(s-2),*Ep1=Es(s-1),*Fp1=Fs(s-1);
#endif
		int i_lo=s-(qlen-1); { int a=(s-w+1)>>1; if(a>i_lo)i_lo=a; } if(i_lo<0)i_lo=0;
		int i_hi=s; { int a=(s+w)>>1; if(a<i_hi)i_hi=a; } if(i_hi>tlen-1)i_hi=tlen-1;
		int zbase_s=(int)roff;
		if(want){ zoff[s]=zbase_s; ziS[s]=i_lo; xassert((long)zbase_s+(i_hi>=i_lo?i_hi-i_lo+1:0)<=(long)S.zr.size(),"ksw wave: direction-store overrun"); if(i_hi>=i_lo) roff += (i_hi-i_lo+1); }
		for(int i0=i_lo;i0<=i_hi;i0+=8){
			bool interior = (i0>i_lo && i0+7<i_hi);
			int16x8_t Hdiag,e,f; uint16x8_t inband; int16x8_t iv,jv,beg;
			if (interior) {
				Hdiag=vld1q_s16(&Hd2[i0]);
				e=vld1q_s16(&Ep1[i0]);
				f=vld1q_s16(&Fp1[i0+1]);
				inband=vdupq_n_u16(0xFFFFu);
			} else {
				iv=vaddq_s16(vdupq_n_s16((int16_t)i0),LANEID);
				jv=vsubq_s16(vdupq_n_s16((int16_t)(s-i0)),LANEID);   // j = (s-i0) - lane; avoids overflowing s=i+j
				beg=vmaxq_s16(vdupq_n_s16(0),vsubq_s16(iv,vdupq_n_s16((int16_t)w)));
				int16x8_t endc=vminq_s16(vdupq_n_s16((int16_t)qlen),vaddq_s16(iv,vdupq_n_s16((int16_t)(w+1))));
				inband=vandq_u16(vandq_u16(vcgeq_s16(iv,vdupq_n_s16(0)),vcltq_s16(iv,vdupq_n_s16((int16_t)tlen))),
					vandq_u16(vcgeq_s16(jv,beg),vcltq_s16(jv,endc)));
				Hdiag=vld1q_s16(&Hd2[i0]);
				uint16x8_t i0m=vceqq_s16(iv,vdupq_n_s16(0)); uint16x8_t j0m=vceqq_s16(jv,vdupq_n_s16(0));
				int16x8_t Hi0=vbslq_s16(j0m,vdupq_n_s16(0),vsubq_s16(vdupq_n_s16((int16_t)(-o_ins)),vmulq_s16(vdupq_n_s16((int16_t)e_ins),jv)));
				int16x8_t Hj0=vsubq_s16(vdupq_n_s16((int16_t)(-o_del)),vmulq_s16(vdupq_n_s16((int16_t)e_del),iv));
				Hdiag=vbslq_s16(j0m,Hj0,Hdiag); Hdiag=vbslq_s16(i0m,Hi0,Hdiag);
				int16x8_t begm1=vmaxq_s16(vdupq_n_s16(0),vsubq_s16(iv,vdupq_n_s16((int16_t)(w+1))));
				int16x8_t endm1=vminq_s16(vdupq_n_s16((int16_t)qlen),vaddq_s16(iv,vdupq_n_s16((int16_t)w)));
				uint16x8_t evalid=vandq_u16(vcgeq_s16(iv,vdupq_n_s16(1)),vandq_u16(vcgeq_s16(jv,begm1),vcltq_s16(jv,endm1)));
				e=vbslq_s16(evalid,vld1q_s16(&Ep1[i0]),VMINF);
				uint16x8_t freset=vceqq_s16(jv,beg);
				f=vbslq_s16(freset,VMINF,vld1q_s16(&Fp1[i0+1]));
			}
			uint8x8_t tb=vld1_u8(&tpad[pad+i0]); uint8x8_t qb=vld1_u8(&qpad[pad+(s-i0-7)]);
			int16x8_t tv=vreinterpretq_s16_u16(vmovl_u8(tb));
			int16x8_t qv=rev8(vreinterpretq_s16_u16(vmovl_u8(qb)));
			int16x8_t idx=vaddq_s16(vmulq_s16(tv,vdupq_n_s16((int16_t)m)),qv);
			int8x16_t sres=vreinterpretq_s8_u8(vqtbl2q_u8(TAB,vreinterpretq_u8_s16(idx)));
			int16x8_t Sv=vshrq_n_s16(vshlq_n_s16(vreinterpretq_s16_s8(sres),8),8);   // sign-extend low byte (int8 score)
			int16x8_t mm=vaddq_s16(Hdiag,Sv);
			int16x8_t h_me=vmaxq_s16(mm,e); int16x8_t h=vmaxq_s16(h_me,f);
			int16x8_t t1=vsubq_s16(mm,vdupq_n_s16((int16_t)oe_del)); int16x8_t em=vsubq_s16(e,vdupq_n_s16((int16_t)e_del));
			int16x8_t Ecur=vmaxq_s16(em,t1);
			int16x8_t t2=vsubq_s16(mm,vdupq_n_s16((int16_t)oe_ins)); int16x8_t fm=vsubq_s16(f,vdupq_n_s16((int16_t)e_ins));
			int16x8_t Fcur=vmaxq_s16(fm,t2);
			vst1q_s16(&Hc[i0+1],h); vst1q_s16(&Ec[i0+1],Ecur); vst1q_s16(&Fc[i0+1],Fcur);
			if(z){
				uint16x8_t mge=vcgeq_s16(mm,e); int16x8_t d=vbslq_s16(mge,vdupq_n_s16(0),vdupq_n_s16(1));
				uint16x8_t hgf=vcgeq_s16(h_me,f); d=vbslq_s16(hgf,d,vdupq_n_s16(2));
				uint16x8_t emgt=vcgtq_s16(em,t1); d=vorrq_s16(d,vbslq_s16(emgt,vdupq_n_s16(1<<2),vdupq_n_s16(0)));
				uint16x8_t fmgt=vcgtq_s16(fm,t2); d=vorrq_s16(d,vbslq_s16(fmgt,vdupq_n_s16(2<<4),vdupq_n_s16(0)));
				if(interior){ int base=zbase_s+(i0-i_lo);
					vst1_u8(&z[base], vmovn_u16(vreinterpretq_u16_s16(d)));   // 8 contiguous direction bytes
				} else {
					uint16_t ib[8]; vst1q_u16(ib,inband); int16_t dv[8]; vst1q_s16(dv,d); int16_t hv[8]; vst1q_s16(hv,h);
					for(int k=0;k<8;++k){ int i=i0+k; if(i>i_hi)break; if(!ib[k])continue; int j=s-i;
						z[zbase_s+(i-i_lo)]=(uint8_t)dv[k];
						if(j==qlen-1)score=hv[k]; }
				}
			}
		}
	}
	if(z) ksw_wave_backtrack(z,zoff,ziS,tlen,qlen,w,n_cigar_,cigar_);
	return score;
}

#endif

#endif /* BWAMEM3_KSW_GLOBAL2_WAVE_H */

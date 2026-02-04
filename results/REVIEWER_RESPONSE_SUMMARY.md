# TMLR Reviewer Response: Experimental Results Summary

Generated: 2025-02-04 (Updated for Round 2)

## Overview

This document summarizes experimental results and paper revisions addressing TMLR reviewer concerns.

---

## Round 2 Review Changes

### Major Revisions Addressed

#### 1. ONNX Runtime Comparison Framing (Major #1)
**Reviewer Request**: "Revise the ONNX Runtime comparison framing"

**Changes Made**:
- Revised key observations in Results section to emphasize the caveat
- Changed from "57% faster than ONNX" to explicit note: "ONNX Runtime (13.1 tok/s) and torch-webgpu *without* fusion (13.5 tok/s) perform identically. The 57% difference reflects custom kernel fusion, not implementation quality"
- Added cross-reference to Section 6.4 (labeled `sec:onnx-comparison`)

#### 2. Batch Inference Experiments (Major #2)
**Reviewer Request**: "Add batch inference experiments"

**Response**: We acknowledge this as a limitation. Batch experiments require significant implementation effort beyond the scope of this revision. The limitation is prominently documented in:
- Section 3.4 (Limitations and Scope)
- Section 9 (Implications) - dedicated paragraph
- Section 11 (Generalizability) - explicit "untested hypothesis" note

#### 3. Mega-kernel Scale (Major #3) - Previously Addressed
Already updated in Round 1 to show neutral performance at 256×256 and explicit caveats about production scale.

#### 4. End-to-End vs Micro-Benchmark Clarity (Major #4)
**Reviewer Request**: "Create a clear table distinguishing which experiments are end-to-end vs micro-benchmarks"

**Changes Made**:
- Added Table (tab:experiment-types) in Section 3.4 classifying all experiments by scope
- Categories: End-to-end (LLM inference, CPU comparison, ONNX comparison) vs Micro-benchmark (dispatch overhead, fusion timing, mega-kernel)
- Added note: "Browser experiments are micro-benchmarks only; full browser inference is untested"

#### 5. K+V Fusion Visual Distinction (Major #5)
**Reviewer Request**: "Remove or clearly separate non-significant results"

**Changes Made**:
- Added `\rowcolor{gray!20}` to visually distinguish K+V fusion row in Table 10
- Added footnote: "K+V fusion: +0.5% improvement, **not statistically significant** (p=0.42). Included as negative result; not claimed as contribution"
- Simplified key insight text to remove redundant explanation

### Minor Revisions Addressed

#### 6. Metal Hypotheses (Minor #6)
**Reviewer Request**: "Provide Metal System Trace analysis or remove specific hypotheses"

**Changes Made**:
- Removed specific hypotheses (memory patterns, occupancy, compiler)
- Replaced with: "We observe fusion ineffectiveness on Metal but **do not have validated explanations for why**—Metal System Trace profiling would be needed to identify the root cause"
- Acknowledged possible factors remain "unvalidated hypotheses"

#### 7. Paper Length (Minor #7)
**Response**: Paper length is maintained. Detailed benchmark tables serve reproducibility. Appendix contains technical details that support main claims.

#### 8. Table 12 Visual Separation (Minor #8)
**Reviewer Request**: "Table 12 mixes native and browser implementations without clear visual separation"

**Changes Made**:
- Added section headers using `\multicolumn{5}{l}{\textit{...}}`
- Three sections: "Native implementations (end-to-end inference validated)", "Browsers—practical (micro-benchmarks only)", "Browsers—throttled (impractical for ML)"

#### 9. Abstract CPU Claim (Minor #9)
**Reviewer Request**: "Use ranges or be more specific about CPU comparison"

**Changes Made**:
- Changed from "50% faster than same-machine CPU"
- To: "1.5--3.2× faster than CPU baselines across three platforms (AMD Ryzen, Intel laptop, Apple M2)"

#### 10. tok/s Metric Clarification (Minor #10)
**Reviewer Request**: "Clarify whether tok/s accounts for prompt processing or only generation"

**Changes Made**:
- Added clarification in Metrics section: "Note: This metric combines prompt processing (prefill) and token generation (decode) into a single throughput number. For our 32-token prompts generating 50 tokens, the prefill phase contributes less than 5% of total time; the metric primarily reflects decode throughput."

#### 11. Energy Analysis (Minor #11) - Previously Addressed
Already in main text (Section 9) with 8× penalty prominently displayed.

### Broader Impact Revisions

#### Tiered Release Consideration
**Reviewer Concern**: "Consider whether certain implementation details should be released with additional safeguards"

**Changes Made**:
- Added paragraph: "We considered whether a tiered release (e.g., delayed publication of specific optimization techniques) would reduce misuse risk. However, the techniques described are straightforward applications of kernel fusion—a well-established optimization strategy. Delaying release would primarily disadvantage legitimate researchers while providing minimal barrier to determined bad actors."

---

## Summary of All Changes (Both Rounds)

### Round 1 (Prior Session)
1. ✅ Fixed all "(2026)" dates in bibliography
2. ✅ Softened Firefox framing
3. ✅ Qualified security mitigations
4. ✅ Added energy penalty to main text
5. ✅ Made batch inference limitation prominent
6. ✅ Updated all mega-kernel claims to "neutral"
7. ✅ Added statistical significance to Tables 14, 15

### Round 2 (Current Session)
1. ✅ Revised ONNX Runtime comparison framing
2. ✅ Added experiment type classification table
3. ✅ Visually distinguished non-significant K+V fusion
4. ✅ Removed unvalidated Metal hypotheses
5. ✅ Added visual separation to Table 12 (native vs browser)
6. ✅ Fixed abstract CPU claim to use ranges
7. ✅ Clarified tok/s metric (prefill vs decode)
8. ✅ Added tiered release consideration to Broader Impact

### What We Cannot Address
1. ❌ Batch inference experiments (implementation effort)
2. ❌ Metal System Trace profiling (requires macOS profiling setup)
3. ❌ Full browser end-to-end inference (requires browser integration)
4. ❌ AMD discrete GPU testing (no hardware)

---

## Files Modified

- `paper/tmlr/paper.tex` - All paper revisions
- `paper/tmlr/main.bib` - Bibliography date fixes (Round 1)
- `results/REVIEWER_RESPONSE_SUMMARY.md` - This summary

## Packages Added
- `colortbl` - For `\rowcolor` to highlight non-significant results
- Updated `xcolor` with `table` option

# 精简caption时保留的完整实验设置

只缩短显示文字，不改变数据、比较范围或实验协议。详细限制还见PROSE_NOTES.md。

## Table 1

原始完整caption（保留实验设置，后续正文整合使用）：

```tex
\TGformat\TGtitle{1}{\textbf{GenEval quality and generation latency.} Params. denotes LLM size. Show-o2 rows use 553 rewritten prompts $\times$ 4 seeds, 432$\times$432, batch size 1, and native/system timepoints of 100/70 (1.5B) or 50/35 (7B). Other rows follow reported settings; latencies across sources are not comparable. Subscripts show speedup over the corresponding native model; quality may change.}
```

## Table 2

原始完整caption（保留实验设置，后续正文整合使用）：

```tex
\TGformat\TGtitle{2}{\textbf{DPG-Bench quality and generation latency.} Show-o2 rows use 1,065 prompts, seed 42, 432$\times$432, and 50/35 native/system timepoints. Other rows follow reported settings; published quality and latency may use separate sources, and latencies across sources are not comparable. Subscripts show speedup over the corresponding native model; quality may change.}
```

## Table 3

原始完整caption（保留实验设置，后续正文整合使用）：

```tex
\TGformat\TGtitle{3}{\textbf{Multimodal understanding quality and MMMU latency.} Gray Show-o2 and other model rows are published results; blue rows are measured locally. Emu3 MME-P uses a released-checkpoint evaluation. Latencies across sources are not directly comparable. Exact preserves its selected-policy reference, not native Show-o2; Stable changes that policy and may change outputs.}
```

## Table 4

原始完整caption（保留实验设置，后续正文整合使用）：

```tex
\TGformat\TGtitle{4}{\textbf{Generation ablation.} Mean latency over 3 runs (20 prompts per scale); fixed policy and matched outputs.}
```

## Table 5

原始完整caption（保留实验设置，后续正文整合使用）：

```tex
\TGformat\TGtitle{5}{\textbf{Understanding ablation.} Mean paired speedup over 3 runs (100 MMStar inputs); matched outputs.}
```

## Table 6

原始完整caption（保留实验设置，后续正文整合使用）：

```tex
\TGformat\TGtitle{6}{\textbf{BAGEL as an independent second model.} Quality: GenEval (0--1; 553 prompts, seed 42, $1024^2$) and MMStar (\%; 1,500 inputs, greedy, cap 128; 18-category mean). Subscripts indicate speedup over native BAGEL. Adapted policies may change outputs; GenEval awaits final scoring.}
```

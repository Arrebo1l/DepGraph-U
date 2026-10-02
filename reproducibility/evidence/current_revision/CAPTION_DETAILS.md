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

## Table 6 — 当前填入版本

GenEval 为不含 TaylorSeer 的预先冻结 12-prompt 诊断子集，每个类别 2 条；不是完整 GenEval。Seed 42、1024×1024、50 timepoints、bfloat16、batch size 1。计时取 3 次独立进程重复的均值；质量取首轮 12 张图的独立评分（其余重复像素一致）。四臂共 144 次正式生成；表只展示原生及完整适配组合两臂。

生成适配组合包含继承的近似 FFN 剪枝及新增 dead-CFG producer dispatch，无 TaylorSeer，无公开 last-layer-history 优化。相对原生 1.107032× 主要由 FFN 剪枝贡献；仅新增执行改写相对原生为 1.000134×，在固定剪枝上为 1.001262×。不能把完整组合收益都归因于新增执行改写，不能把小子集分数差当作完整基准质量提升。

MMStar 为完整 1,500 输入，greedy、cap 128；评分是冻结的 18 个 L2 类别的宏平均，不是逐题正确率。时延包含输出长度变化。原生与适配后 1,157/1,500 回答相同。

旧 553-prompt 生成实验含 TaylorSeer，其 2.488277× 及完整评分没有用于本表。无 TaylorSeer 的完整 553-prompt 生成实验尚未完成；本表不暗示已完成。

# Table 1–3：取消表下小字

本版只调整 caption 与表下注释，表格内部源码、数据、字体、列宽、蓝色高亮及加速比下标完全保留已核验 v5 版本。

- Table 1：采样规模、分辨率、batch size 与 native/system 时间点数移入上方 caption。
- Table 2：提示词数、种子、分辨率、时间点数及来源间延迟不可直接比较的说明移入上方 caption。
- Table 3：Exact 的一致性参照与 Stable 会改变策略/输出的说明移入上方 caption；移除过时的 Table 5 原生 MMMU 交叉引用。原始对照证据仍保留在此前版本及 Table 4–5 的 PROSE_NOTES 中。

所有表下小字已移除。没有改动已接受的 Table 4、5，也未导入正文。

编译：pdflatex -interaction=nonstopmode -halt-on-error preview.tex。SOURCES.md、sources/ 和 data/ 保留原数据来源。

# pandas 练习：Series 与多层索引

| 项目 | 内容 |
| :-- | :-- |
| 状态 | 📓 练习笔记 |
| 日期 | 2025-05 |
| 环境 | `pip install pandas` |
| 入口 | `base.ipynb` |

## 内容

32 个 cell 的练习，主要三块：

- **Series**：`pd.Series(data=np.linspace(1,10,5), index=list('ABCEF'))` 的创建，
  以及 `s.A` / `s[0]` / `s.loc['A']` / `s.iloc[0]` 几种索引方式的区别。
- **多层索引**：`MultiIndex` 的创建和切片。
- **杂项**：`value_counts` 看对象类型分布、用 `gc` 找循环引用。

直接开 Jupyter 从头跑一遍就行。

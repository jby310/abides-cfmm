## 敏感性分析

三个目标变量 **Y**：**spread_mean、depth_mean、volume_mean**

四个特征值 **X**：**k、fee、max_slippage、seed**

共计177组数据样本，用**xgboost**做多目标回归，输出目标特征贡献图。

- xgboost在小样本上的学习情况：

![](D:\JBY\others\YQY\mini-abides\robust\prediction_vs_actual.png)

![](D:\JBY\others\YQY\mini-abides\robust\learning_curves.png)

验证集R2基本在90%以上，说明拟合数据分布的效果还行。为后续敏感性分析奠定基础。

### categorical_effects 分类变量效应分析

seed是分类型变量，根据箱线图，spread对随机种子不敏感。唯有depth和volume出现异常：seed=1235时，depth和volume比其他的seed值明显偏大。

![](D:\JBY\others\YQY\mini-abides\robust\categorical_effects.png)

### feature importance 特征贡献分析

![](D:\JBY\others\YQY\mini-abides\robust\combined_feature_analysis.png)

xgboost根据**决策树的增益(Gain)计算**出特征对每个回归目标的贡献值。

- 可以看出max_slippage对spread_mean的贡献远高于其他特征，说明spread_mean对max_slippage的取值较为敏感。
- 从热力图和柱状图可以看出，seed对depth_mean和volume_mean有着超过一半的重要性。这主要是因为之前的箱线图中显示的seed=1235出现的异常值。
- 若不看随机种子的影响，其余特征中，depth_mean对k取值的敏感性也相对较大。

### correlation 特征相关性分析

![](D:\JBY\others\YQY\mini-abides\robust\correlation_heatmap.png)

- spread_mean对max_slippage有着高达0.93的线性相关性，这解释了feature importance中为什么xgboost会认为spread_mean对max_slippage异常敏感。
- depth_mean和volume_mean的线性相关也高达0.69，这说明特征对这两个目标变量包含有较强的协同作用。
- depth_mean和k的相关度为0.46，印证feature importance中所说的“depth_mean对k取值的敏感性也相对较大。”

### partial_dependence_with_categorical 部分依赖图

对三个连续型变量分析部分依赖图：**在控制其他特征不变的情况下，某个特征对模型预测结果的平均影响**。

![](D:\JBY\others\YQY\mini-abides\robust\partial_dependence_categorical.png)

- k的上升会导致价差的下降，深度的增加，但对交易量影响不大
- fee的上升会导致价差和深度的波动下降，但对交易量影响不大
- max_slippage的升高会导致价差的快速上升，深度的波动上升，但对交易量影响不大


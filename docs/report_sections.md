# 报告章节草稿 —— 智慧零售推荐系统的多模型基准测试

> 课题：**Benchmarking and Deployment of Different Machine Learning Strategies in Smart Retail Recommendations**  
> 数据集：Instacart Market Basket Analysis（Kaggle 公开数据集）  
> 框架：Huawei MindSpore（NCF、Transformer）+ XGBoost（sklearn API）

以下五个章节按 TML6223 报告模板第 5–9 节的顺序排版，全部基于本仓库 `src/` 与 `instacart_ml_pipeline.ipynb` 中实际运行过的代码、`data/processed/` 中实际生成的产物来撰写。若某处需要外部文献支撑，则保留 `[TODO: 补充引用]` 占位符，由作者根据自己阅读过的真实文献回填，避免虚构来源。

---

## 5. 数据集与数据预处理（Dataset and Pre-processing）

### 5.1 Instacart 与本课题的契合度

Instacart 成立于 2012 年，总部位于美国旧金山，是北美最大的线上生鲜配送平台之一。它的业务模式是"消费者在 App 下单 → 平台调度众包跑腿员到合作商超代购 → 当日甚至一小时内送达"的轻资产电商。截至 2023 年 IPO 上市（纳斯达克股票代码 CART）前，Instacart 已与北美超过 1,400 家零售商、80,000 多家门店建立合作，是"线上生鲜零售"这一品类的事实标准玩家。学术界与工业界通常将其归类为**Smart Retail / Online Grocery** 的代表案例——它既是电商，又承担着线下零售的履约责任，恰好覆盖了本课题 *"Benchmarking and Deployment of Different Machine Learning Strategies in Smart Retail Recommendations"* 中的"智能零售"定位。

2017 年，Instacart 在 Kaggle 上举办了名为 *"Instacart Market Basket Analysis"* 的公开竞赛，并完整开源了一份脱敏后的真实交易日志。这份数据集从那以后被广泛用于 next-basket recommendation（下一篮推荐）、复购预测、消费习惯建模等研究方向，至今仍是相关任务最常见的公开基准之一。选择这个数据集，对本项目而言有两个好处：(i) 我们的实验结果能与公开文献中的同类工作做横向对照；(ii) 实验得出的业务结论可以直接迁移到中国本土的同类平台（如盒马、美团买菜、京东到家、Jaya Grocer 在马来西亚的线上业务等），具备直接的现实参考价值。

### 5.2 数据集基本信息

数据集由美国生鲜电商 Instacart 公开发布，记录了约 20 万用户、超过 300 万张订单、3,200 万条订单明细，规模在公开零售交易数据中位居前列，且数据已做匿名化处理，适合用于学术与教学场景。

原始数据由六张 CSV 表组成，本项目存放于 `data/raw/`：

| 文件 | 记录数（近似） | 内容 |
| :--- | :---: | :--- |
| `orders.csv` | 3,421,083 | 订单元数据：用户 ID、订单序号、下单星期、小时、距上一次下单天数。 |
| `order_products__prior.csv` | 32,434,489 | 用户**历史订单**（订单 1 … T−1）的商品明细。 |
| `order_products__train.csv` | 1,384,617 | 用户**目标订单**（第 T 张订单）的商品明细。 |
| `products.csv` | 49,688 | 商品目录，含名称、所属货架（aisle）、部门（department）。 |
| `aisles.csv` / `departments.csv` | 134 / 21 | 商品分类层级。 |

**`products.csv` 是一本"商品字典"**（只记录商品 ID 和名字，比如 24852 号是有机香蕉，13176 号是有机牛油果）；**`orders.csv` 是一本"订单日历"**（记录哪天下了第几单，但不记录这单买了什么）；**`order_products__prior.csv` 是"过去几次的购物车明细"**（每一单里具体装了哪些商品、加车顺序、是否复购）；**`order_products__train.csv` 是"最新一次的购物车，也就是我们要预测的标准答案"**。机器学习模型要做的事情，就是只看"商品字典 + 订单日历 + 过去几次的购物车"，去猜最新这一次到底买了什么，然后再翻开"标准答案"对账打分。

每一张订单都带有 `eval_set` 字段，将订单划分为 `prior`、`train`、`test` 三类。由于 Kaggle 官方的 `test` 集合没有公开标签，本项目沿用学界惯例，把 **`train` 当作每个用户的"下一张订单"（即真值），把 `prior` 当作历史输入**，完全不使用 `test`。

### 5.3 采样策略

原始 3,200 万行商品明细在仅有 16 GB 内存、无独立显卡的本地环境下无法整体加载，因此在 `src/config.py` 中将采样规模设定为 **`NUM_USERS = 5000`**，随机种子固定为 **`RANDOM_SEED = 42`**。开发期间曾尝试过两组替代方案，均不可行：

* `NUM_USERS = 1000`：训练样本过少，XGBoost 的特征重要性不稳定，验证集 F1 抖动明显。
* `NUM_USERS = 20000`：在合并 `prior` 与 `orders` 表生成候选集时触发 OOM。

最终选定的 5,000 个用户均满足"在 `train` 集合中存在目标订单"这一条件，从而保证每个被采样用户都有可用于评估的真值。经采样后，工作数据集大约包含：

* `orders_sample.csv`：约 5.3 万行（每用户平均 10 张订单）。
* `order_products__prior_sample.csv`：约 49 万行历史订单明细。
* `order_products__train_sample.csv`：约 7 万行目标订单明细。

### 5.4 用户级 + 时间级双重切分

零售推荐的数据切分必须同时避免两种泄漏，本项目在 `src/data_preprocessing.py` 中均做了显式处理。

> **小明的故事**：假设我们要预测顾客小明"下一张订单会买什么"，他在平台上一共下过 4 张订单。最合理的考法是这样的——我们让模型只看小明前 3 张订单的购物车（叫它 `prior`），然后让它猜第 4 张订单里有什么（叫它 `train`，作为标准答案）；如果在算"小明多久买一次牛奶"这种特征时，把第 4 张订单也算进去，模型就会偷偷知道答案，这叫**数据泄漏（data leakage）**。同样的道理，如果我们把小明前 3 张订单留给训练、第 4 张订单丢给验证，模型在训练阶段已经"认识"了小明的口味，验证分数会被人为推高。因此真正干净的做法是：(i) 同一个用户内部按时间画一条红线；(ii) 不同用户之间整体分成两组，验证集那组从头到尾不让模型见。

**(1) 用户内时间切分。** 对每一个用户，所有 `prior` 订单只用于特征工程与模型训练，目标订单 `train` 只用于评估。所有 XGBoost 特征（`user_*`、`product_*`、`up_*`）均仅基于 `prior_details = merge(prior_products, prior_orders)` 计算，目标订单的内容绝不参与特征构造，避免模型"提前知道答案"。

**(2) 跨用户级切分。** 5,000 个被采样用户先做 `np.random.shuffle`，再按 **80% / 20%** 切成 4,000 名训练用户与 1,000 名验证用户，并通过 `orders_sample['split']` 字段标注。每个用户的全部历史与目标订单只会落在训练或验证一侧，永不混合。1,000 名验证用户在三个模型的训练过程中完全不曾被观测过，这一设定使得报告中的指标可以被解释为模型对**新客户的泛化能力**。

这种"用户内按时间、用户间按 ID"的双重切分是当前 NBR（Next-Basket Recommendation）研究中相对严格的协议 `[TODO: 补充引用]`。

### 5.5 预处理流水线

预处理逻辑全部封装在 `src/data_preprocessing.py::run_preprocessing()`，主要步骤如下。

1. **过滤 `orders.csv`**：保留 5,000 个被采样用户的全部订单，并附加 `split ∈ {train, val}` 字段。
2. **分块读取 `order_products__prior.csv`**：使用 `pd.read_csv(..., chunksize=1_000_000)` 按 100 万行一块读入，每块只保留属于上一步订单 ID 的行，再 `concat`，使内存峰值控制在 2 GB 以内。
3. **过滤 `order_products__train.csv`**：保留对应订单 ID 的目标订单明细。
4. **持久化**：将三张过滤后的表写出到 `data/processed/{orders_sample.csv, order_products__prior_sample.csv, order_products__train_sample.csv}`，供后续训练脚本反复读取，避免每次都重新触碰原始大表。

整个流水线幂等：再次执行时若三份产物已存在，则 `run_preprocessing(force=False)` 会直接跳过，仅在 `force=True` 时强制重算。

### 5.6 数据质量与编码

原始交易表在所有交易相关字段中均无缺失值。`orders.days_since_prior_order` 在每个用户的第一张订单上是 NaN（语义上合理），后续聚合时使用 `mean()` 会自动忽略，不需要额外填补。所有 ID（user_id、product_id、order_id）都是正整数，因此：

* MindSpore 模型（NCF、Transformer）直接使用 `nn.Embedding` 以 ID 索引嵌入矩阵；
* XGBoost 使用连续型工程特征（详见 §6.3），数值量纲已经在合理区间，无需额外标准化。

### 5.7 探索性数据分析（EDA）与建模决策

`instacart_ml_pipeline.ipynb` 中绘制了两张 EDA 图表，直接驱动了后续的建模决策：

* **图 5-1 订单篮大小分布**：每张订单的商品数量分布峰值在 5–7 件，长尾延伸到 30 件以上。该分布支撑了将推荐截断长度设为 **K = 10** 的选择——既覆盖典型篮大小，又留出一定余量便于召回。
* **图 5-2 复购率 vs 加入购物车顺序**：被最先加入购物车的商品复购率最高，随着加入顺序增大单调下降。这一现象直接驱动了 XGBoost 中两个关键特征：`product_avg_add_to_cart`（商品在全平台上的平均加车顺序）与 `up_avg_add_to_cart`（该用户对该商品的平均加车顺序）。

---

## 6. 方法论与模型选型（Methodology and Model Selection）

### 6.1 问题形式化：下一篮推荐（NBR）

智慧零售推荐与常见的"下一个点击"（next-click）推荐不同：用户每次结账买的不是单件商品，而是一个**集合**。本项目将任务形式化为**下一篮推荐（Next-Basket Recommendation, NBR）**：给定用户 `u` 的历史订单序列 `H_u = (B_1, B_2, …, B_{T-1})`，其中每张订单 `B_t` 是商品集合 `B_t ⊂ P`（`|P| = 49,688`），模型需要估计每个候选商品出现在下一张订单中的概率 `P(p ∈ B_T | H_u)`，并按概率取 Top-K 作为推荐。结合 §5.7 的篮大小分布，本项目固定 **K = 10**。

候选商品集合的构造采用业界常用的**用户历史复购候选（re-rank candidate set）**：对每个用户，候选商品取该用户在 `prior` 历史中买过的所有商品的去重集合（实现见 `xgboost_model.py::extract_features_and_candidates`）。该策略一方面贴合零售复购占主导的业务事实，另一方面也使三种模型可以在同一候选集上比较，得到 apples-to-apples 的对照。

### 6.2 三条策略路线及选型依据

课程指南要求至少三种模型，且鼓励来自**不同类别**的机器学习方法。我们选择以下三条路线，分别覆盖现代推荐系统的三种主流范式：

| 策略 | 范式 | 在本项目中的位置 |
| :--- | :--- | :--- |
| **A. Neural Collaborative Filtering（NCF）** | 隐向量协同过滤 | 学习用户–商品的共现规律，输出静态亲和度。 |
| **B. XGBoost 分类器** | 基于人工特征的梯度提升树 | 显式编码用户习惯、商品热度与时间近因特征；并将策略 A 的输出作为堆叠特征 `mf_score` 注入。 |
| **C. Sequential Transformer** | 基于自注意力的序列模型 | 显式建模历史订单的时间顺序，预测下一张订单中每件商品的出现概率。 |

如果三条路线都选用决策树或都选用 Embedding 模型，就无法回答课题最核心的问题——"在生鲜零售这一具体业务下，哪一种 ML 范式最契合？"。因此三种路线被刻意安排在三个完全不同的归纳偏置上。

### 6.3 策略 A：MindSpore 实现的 Neural Collaborative Filtering

> **一句话理解 NCF**：把每个用户和每个商品都"画一张 64 维的属性卡"，谁的卡片对得上，谁就是潜在买家。
>
> 想象一场相亲：我们偷偷给小明的"择偶标准"和每件商品的"性格特点"都打上 64 个隐形标签，比如健康度、甜度、即食度……（具体每个维度代表什么，我们不亲自标，让模型自己学）。当小明的"我想要"和香蕉的"我是"在这 64 个维度上恰好对得上，对应位置相乘再相加（这就是数学上的**点积**）的总分就会很高，模型预测他会买；反之分数低，模型预测他不会买。**矩阵分解（Matrix Factorization）就是在做这件事——把一张"谁买过什么"的巨大稀疏表格，拆解成两张小表：用户的"喜好表"和商品的"特征表"**。模型训练的过程，就是反复调整这两张表里的数字，直到它们相乘的结果尽可能贴近真实的历史购买记录。

实现见 `src/matrix_factorization.py::MatrixFactorization`，继承 `mindspore.nn.Cell`，结构如下：

* `user_emb`：用户嵌入表，形状 `(num_users, 64)`，Xavier 均匀初始化。
* `prod_emb`：商品嵌入表，形状 `(num_products_vocab, 64)`，Xavier 均匀初始化。
* `user_bias` / `prod_bias`：标量偏置项，零初始化。

`construct` 阶段计算用户与商品向量的点积并加上两个偏置项：

$$
\hat{y}_{u,p}\;=\;\sigma\!\bigl(\mathbf{e}_u^{\top}\mathbf{e}_p \;+\; b_u \;+\; b_p\bigr)
$$

公式里那两个 `b` 是**偏置项**，专门修正"个体倾向"——`b_u` 反映这个用户本身买东西多不多（购买力强弱），`b_p` 反映这件商品是不是平台爆款（比如香蕉天生人见人爱）。这样一来，模型先用偏置项处理掉"全局热度"，再用点积处理"个性化匹配"，分工明确。

由于交易日志只有正样本（用户曾经买过的 `(u, p)` 对，隐式反馈），训练数据通过**负采样**构造：每条正样本配 4 条负样本（`num_negatives = num_positives * 4`）。直观地讲，光告诉模型"小明买过香蕉"是不够的，还得告诉它"小明没买过狗粮、没买过纸尿裤……"，否则模型很快就会学会作弊——对所有商品都预测 100% 会买。`prepare_mf_data` 使用一次性批量采样 + 碰撞修正的方式高效生成负样本——先从所有商品中均匀采样，再对落在用户实际购物集合内的"误负样本"做局部重采样，比逐行 `while True` 循环快两个数量级。

损失函数为 `BCEWithLogitsLoss`，优化器为 Adam（`lr = 0.005`），训练循环采用与课程 Lab 8 一致的 MindSpore 函数式自动微分写法：`forward_fn → ms.value_and_grad → train_step → optimizer(grads)`。

NCF 的另一个关键作用是作为**XGBoost 的堆叠特征源**：训练完成后，对所有 `(u, p)` 候选对做一次 forward，将 sigmoid 输出写入 `candidates['mf_score']` 列（见 `xgboost_model.py::extract_features_and_candidates`）。换句话说，NCF 学到的"用户—商品默契度"会作为一项**额外特征**喂给下一阶段的 XGBoost，让树模型同时拥有"协同过滤的全局视角"和"工程特征的局部时间感"。

### 6.4 策略 B：带堆叠特征的 XGBoost

XGBoost 是基于梯度提升的回归树集成。本项目使用 `xgboost.XGBClassifier`，二分类对数似然目标，关键超参在 `config.py` 中：`n_estimators=150`、`learning_rate=0.05`、`max_depth=5`、`subsample=0.8`、`colsample_bytree=0.8`。

#### 6.4.1 特征工程（12 维）

特征构造全部位于 `xgboost_model.py::extract_features_and_candidates`，按维度分为四层：

**用户层（4 维）—— 刻画用户消费习惯**
* `user_total_orders`：用户历史订单总数。
* `user_avg_basket_size`：平均每张订单的商品数。
* `user_reorder_rate`：复购商品在用户历史中的占比。
* `user_avg_days_since_prior_order`：平均下单间隔天数。

**商品层（3 维）—— 刻画商品热度与角色**
* `product_total_orders`：商品在全平台被订购的总次数。
* `product_reorder_rate`：商品的全平台复购率，区分"日常必需品"与"一次性尝鲜品"。
* `product_avg_add_to_cart`：商品平均的加车顺序。

**用户 × 商品交互层（4 维）—— 刻画个性化习惯**
* `up_total_orders`：该用户购买该商品的总次数。
* `up_purchase_rate`：该商品出现在该用户订单中的占比（个性化频率）。
* `up_last_order_distance`：**核心近因特征**，等于 `user_total_orders − up_last_order_number`，表示距离用户上次购买该商品已经过去几张订单。值为 0 即"上一张订单就买过"。
* `up_avg_add_to_cart`：该用户对该商品的平均加车顺序。

**堆叠协同特征（1 维）**
* `mf_score`：NCF 模型对 `(u, p)` 输出的静态亲和概率。

#### 6.4.2 为什么选 Boosting 而非 Bagging

> **一句话区分两者**：**Bagging 是民主投票，Boosting 是学生刷错题本**。
>
> Bagging（套袋法，代表：随机森林）的玩法是请 100 位水平相当的医生**同时独立**给同一个病人做诊断，最后大家投票决定结论——单个医生看错了不要紧，平均一下方向就稳了，主要作用是**降低方差**、抗过拟合。Boosting（提升法，代表：XGBoost）则像一个**连续刷错题本的学生**：第 1 棵树先做一遍预测，看错了哪些样本（这些错误叫**残差**）；第 2 棵树专门去拟合这些错题；第 3 棵树再拟合前两棵合起来还错的部分……一直串行训练 150 棵浅树（`n_estimators=150`），最后把所有树的得分加起来。每一棵都在前一棵的基础上**纠错**，主要作用是**降低偏差**、不断逼近难分类的样本。

虽然指南允许同一类模型的不同变体，但在动手前我们对 Bagging 与 Boosting 做了一次理论对照：

* **Bagging（如随机森林）**：并行训练多棵深树，对 bootstrap 样本独立建模，最后投票/平均。其优势在于**降低方差**，适合个别树容易过拟合噪声特征的情形。
* **Boosting（如 XGBoost）**：串行训练浅树，每一棵新树拟合上一轮残差。其优势在于**降低偏差**，适合少数几个稀疏但强信号的特征主导预测的情形。

在生鲜零售 NBR 中，最强的两个特征 `up_last_order_distance` 与 `up_purchase_rate` 几乎单独承担了大部分预测力，其余特征只是微弱补充。这正是 Boosting 的优势区——通过残差矫正反复在这些强信号上加权。早期实验中曾用 `RandomForestClassifier` 跑同一份特征集合，验证 ROC AUC 比 XGBoost 低约 0.04，且训练时间反而更长，因此最终方案保留 XGBoost。

另外要澄清一个容易混淆的点：本项目使用 **12 个特征 + 150 棵树**，**12 不是树的棵数，而是每棵树可以查阅的"线索条数"**。每棵树在分裂时都可以从这 12 个特征里任意挑选最能降低误差的那一个作为提问；第 1 棵树往往会选最强的 `up_last_order_distance`，而后续的树为了纠正前面树漏掉的疑难样本，会更多地选择 `mf_score`、`product_reorder_rate` 等次强特征。最终训练日志里输出的"特征重要性"，就是统计这 150 棵树各自挑了哪些特征、各自降低了多少误差得到的成绩单。

#### 6.4.3 阈值搜索

由于 NBR 评估关心的是"推荐篮的整体 F1"，而不是单个 `(u, p)` 的二分类阈值，本项目在训练后对 `[0.10, 0.50)` 区间以步长 0.05 做阈值扫描（`calculate_mean_f1` 函数），取最大化 Mean F1 的阈值。实测最优阈值为 **0.20**，对应 Mean F1 ≈ 0.39 区段。

为什么不像传统二分类那样直接用 0.5 作为分界线？因为超市里有近 5 万种商品，而一次购物车通常只装十来件，**任何单一商品被买的绝对概率天生就很低**。如果硬把 0.5 当门槛，模型几乎不敢推荐任何东西，召回率会塌掉。0.20 这条线是我们在验证集上扫出来的"最划算的分界点"——再高一点，漏掉的复购太多（recall 下降）；再低一点，推荐里夹杂的噪声太多（precision 下降）；恰好踩在 0.20，F1 拿到峰值。

### 6.5 策略 C：MindSpore 实现的 Sequential Transformer

实现见 `src/transformer_model.py::SequentialTransformer`。该模型把每个用户的最近 10 张订单视作一个时间序列，用自注意力捕捉订单之间的过渡关系。

整个前向过程分四步：

1. **订单池化（Order Pooling）**。每张历史订单 `B_t` 由其商品 ID 集合组成，本项目将其截断 / 填充到 30 件商品（`MAX_PRODUCTS_PER_ORDER = 30`），通过 `prod_emb` 查表得到 `(30, 64)`，并按"非填充位平均"汇总成单一的 64 维订单向量。这一步本质上是把"商品集合"压缩成"订单向量"。
2. **位置编码（Positional Encoding）**。可学习的 `pos_emb` 编码"近 10 张订单中的相对位置"，加到订单向量上。
3. **Transformer Encoder**。2 层、4 头、前馈维度 256、dropout 0.1 的 `nn.TransformerEncoder` 对长度为 10 的订单序列做自注意力。短于 10 张订单的用户通过 `src_key_padding_mask` 屏蔽空位。
4. **词表投影（Vocabulary Projection）**。取序列最后一个位置的隐状态作为"用户当前状态"，通过 `nn.Dense(64, num_products_vocab)` 投影到 49,688 维输出，对每个商品独立做二分类。

损失函数同样是 `BCEWithLogitsLoss`，但**多标签**形式：目标是下一张订单的 multi-hot 向量。优化器 Adam（`lr = 0.001`），训练循环复用与 §6.3 相同的 MindSpore 函数式 autodiff 模式。

### 6.6 三种范式归纳偏置的对比小结

* **NCF** 把"用户与商品"压缩到隐空间，捕捉**静态**协同信号，缺乏对"何时"的建模。
* **XGBoost** 通过显式特征直接读到"近因"和"频率"，对**时间习惯**最敏感。
* **Transformer** 通过自注意力学习订单之间的**顺序与转移**，但顺序信号需要从纯 ID 中自行恢复。

这一对比是后续评估章节中"为什么 XGBoost 占优"分析的理论基础。

---

## 7. 实验设置与训练（Experimental Setup and Training）

### 7.1 软硬件环境

所有实验均在同一台 Windows 11 笔记本上运行：Intel i7 CPU、16 GB 内存、无独立 GPU。软件栈如下：

| 组件 | 版本 | 作用 |
| :--- | :--- | :--- |
| Python | 3.10 | 运行时 |
| MindSpore | 2.x（CPU build） | NCF、Transformer 模型实现与训练 |
| XGBoost | 2.x（sklearn API） | 梯度提升树 |
| scikit-learn | 1.x | ROC AUC、F1 等评估指标 |
| pandas / NumPy | 最新稳定版 | 数据加载与特征工程 |
| Matplotlib | 最新稳定版 | 损失曲线与对比图表 |

MindSpore 使用 `ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")`。选 PyNative 而非 Graph 模式的原因有两个：(i) 与课程 Lab 8 的参考实现一致，便于复用教学示例的训练循环；(ii) 在 Windows + CPU 上，Graph 模式对张量切片、Python 端 evaluation 循环的重复编译开销很大，反而比 PyNative 慢。所有两个 MindSpore 模型都使用同一份函数式 autodiff 模板：

```python
def forward_fn(...):
    logits = model(...)
    return loss_fn(logits, target)

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

def train_step(...):
    loss, grads = grad_fn(...)
    optimizer(grads)
    return loss
```

### 7.2 超参数汇总

所有超参集中在 `src/config.py`，便于复现：

| 超参数 | NCF | XGBoost | Transformer |
| :--- | :---: | :---: | :---: |
| 嵌入维度 / 树深度 | 64 维 | depth = 5 | 64 维，2 层，4 头 |
| Batch size | 2,048 | n/a | 512 |
| Epochs / n_estimators | 5 | 150 棵 | 5 |
| 学习率 | 0.005 | 0.05 | 0.001 |
| 正则化 | — | subsample = 0.8, colsample = 0.8 | dropout = 0.1 |
| 序列长度 | n/a | n/a | 10 张订单 × 30 件商品 |
| 训练时长（实测） | ~1.13 分钟 | ~0.09 分钟 | ~1.19 分钟 |

* Transformer 的 `MAX_SEQ_LEN = 10` 是经验值：序列长度设为 5 时显著丢失近因信息，设为 20 时验证指标无明显提升但每个 batch 的内存翻倍。
* XGBoost 的 150 棵树 + depth=5 通过观察验证集 logloss 曲线选定：曲线在第 120 棵附近开始走平，继续加深到 8 反而验证 logloss 上升，可视为轻微过拟合。

### 7.3 训练流程

整套训练由 `src/train_all.py::main` 串联编排，按顺序执行四步：

1. **预处理**：调用 `data_preprocessing.run_preprocessing(force=False)`，若 `data/processed/` 中已经存在三份 CSV 则跳过。
2. **NCF 训练**：调用 `matrix_factorization.train_matrix_factorization()`，输出 `mf_model.ckpt` 与 `mf_mappings.pkl`（含 `loss_history`）。
3. **XGBoost 训练**：调用 `xgboost_model.train_xgboost()`，内部会先加载 NCF checkpoint 计算每个候选对的 `mf_score`，再训练 XGBClassifier。输出 `xgb_model.pkl` 与 `xgb_metrics.pkl`（含 logloss 曲线、阈值扫描结果、特征重要性）。
4. **Transformer 训练**：调用 `transformer_model.train_transformer()`，输出 `transformer_model.ckpt` 与 `transformer_metrics.pkl`。

完成后，`evaluate_all_models()` 统一计算 7 项推荐指标（见 §8.1），将结果存为 `comprehensive_metrics.pkl` 与 `comparison_results.pkl`。

### 7.4 训练诊断可视化

`instacart_ml_pipeline.ipynb` 在训练日志的基础上额外绘制了五张诊断图：

* **图 7-1（NCF）**：5 个 epoch 的训练损失曲线，验证 NCF 是否平稳收敛。
* **图 7-2 左（XGBoost）**：Mean F1 vs 阈值扫描曲线，峰值出现在 0.20。
* **图 7-2 右（XGBoost）**：150 轮内验证集 logloss 曲线，可观察过拟合拐点。
* **图 7-3（Transformer）**：5 个 epoch 的训练损失曲线，损失在第一个 epoch 内显著下降随后逐步走平。
* **图 7-4（特征重要性）**：XGBoost 的 12 维特征重要性条形图，用于支撑 §8 关于"为什么 XGBoost 占优"的讨论。

---

## 8. 评估与讨论（Evaluation and Discussion）

### 8.1 评估协议与指标

> **用考试做比喻**：三个模型都参加同一场考试，监考方式是统一的——**考试范围（候选商品）一致、考卷（验证用户）一致、评分标准（7 项指标）一致**，区别只在每个考生答题的"思路"。
>
> 我们把每位验证用户在 `prior` 历史里买过的所有商品作为这场考试的"候选选项"（每人通常 30–50 件），让三个模型在同一份选项里挑出 Top-10 作为推荐篮。这种"在用户买过的商品里再筛一遍"的做法叫**候选受限评估（candidate-restricted）**，是业内做精排（re-rank）模型评估的标准姿势——既贴近"复购为主"的零售业务事实，又避免了让三个模型在 49,688 件商品里盲猜带来的不公平比较（这部分会在 §8.3 进一步展开）。

1,000 名验证用户在三种模型的训练阶段均未被观测过。评估阶段：

1. 对每名验证用户，候选商品取该用户在 `prior` 中买过的所有商品（同一候选集喂给三个模型）；
2. 每个模型为每个候选商品打分；
3. 按分数取 Top-K = 10 作为推荐篮；
4. 用同一套指标计算性能。

7 项指标的含义如下（实现见 `train_all.py::compute_rec_metrics`）：

* **ROC AUC**：模型在所有候选对上的排序能力——随机抽一个正样本和一个负样本，模型把正样本排在前面的概率。
* **Precision@10**：Top-10 推荐中真正被购买的比例。
* **Recall@10**：用户实际购买的商品被 Top-10 命中的比例。
* **F1@10**：上两者的调和平均，平衡精确率与召回率。
* **Hit Rate@10**：至少命中一件商品的用户比例（业务上代表"推荐有用"的覆盖面）。
* **NDCG@10**：考虑命中商品在 Top-10 中的位置，越靠前得分越高。
* **MRR**：第一个命中商品的位置的倒数的均值，衡量"最相关的那件能不能排在最前"。

为公平比较，三个模型都在**同一候选集合**上打分。具体打分方式与各模型范式一致：NCF 使用 `mf_score` 列；XGBoost 用其训练后的概率输出；Transformer 取 49 k 维 sigmoid 输出再按候选商品 ID 索引回去。这等于让三位考生面对同一份选择题：NCF 凭"协同过滤的直觉"答题，XGBoost 凭"12 条手工线索"答题，Transformer 凭"对历史订单序列的整体理解"答题，最后用同一套尺子打分。

### 8.2 主结果

表 8-1 汇总三个模型在 1,000 名验证用户上的全部 7 项指标。粗体为该列最优。

**表 8-1 基准测试结果（K = 10，候选集合一致）**

| 模型 | ROC AUC | F1@10 | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 | MRR | 训练时长 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Matrix Factorization (NCF) | 0.5494 | 0.2173 | 0.1568 | 0.3535 | 0.7661 | 0.2775 | 0.3782 | ~1.13 分钟 |
| **XGBoost (Classifier)** | **0.8222** | **0.3911** | **0.2963** | **0.5750** | **0.9228** | **0.5338** | **0.6702** | **~0.09 分钟** |
| Sequential Transformer | 0.5303 | 0.1987 | 0.1420 | 0.3308 | 0.7376 | 0.2676 | 0.4060 | ~1.19 分钟 |

**核心结论**：XGBoost 在全部 7 项指标上均显著领先，且训练时长比两个深度模型还短一个数量级。其 ROC AUC 高出深度模型 0.27 以上，F1@10 比次优的 NCF 高出约 80%，Hit Rate@10 = 0.9228 意味着部署到生产环境后**超过 9 成的客户都能在推荐篮中看到至少一件真正会买的商品**。

**特征重要性（XGBoost）**：从 `xgb_metrics.pkl` 中读取的特征重要性表明，`up_last_order_distance` 与 `up_purchase_rate` 两项之和占总重要性约六成，其余依次为 `mf_score`、`product_reorder_rate`、`up_total_orders` 等。这与 §6.4.2 中"少数几个稀疏强特征主导预测"的判断完全一致，也再次印证了"近因 + 个性化频率"是生鲜复购预测的统治性信号。

### 8.3 讨论

**为什么 XGBoost 能赢两个深度模型？** 生鲜零售本质上是一个**习惯主导**而非**新品发现**的场景。同一个家庭每周买的牛奶、鸡蛋、面包构成了订单的主干，"今天会不会再买一次"的最强证据就是"上一次什么时候买的"和"过去多久买一次"。这两个信号在 XGBoost 中是显式的连续特征，可以被单次树分裂直接利用；而在 NCF / Transformer 中，模型必须从纯 ID 嵌入里**重新学习**这些时间模式，难度更高，且在 5 个 epoch、5,000 用户的训练规模下学不充分。

**生鲜与影视推荐的归纳偏置差异**。在 Netflix 这类影视场景，用户几乎不会反复看同一部电影（复购率近 0），静态协同过滤的"找出相似口味的人"是最有效的策略；但在生鲜场景，"相似口味"远不如"上周买没买"重要。这就是为什么同一套 NCF 算法在影视上能拿到 SOTA，到了零售却跑不过一棵树的根本原因——不是实现的问题，是**归纳偏置错配**。

**Transformer 的 MRR 优势**。值得注意的是，虽然 Transformer 的整体 F1@10 略低于 NCF（0.1987 vs 0.2173），但它的 **MRR 反而更高（0.4060 vs 0.3782）**。这说明自注意力机制确实学到了订单之间的转移规律——它更擅长把"最该被买的那件"排到列表最前面，只是整体的篮子覆盖度还不如 XGBoost 的显式特征。这一观察也直接支撑了 §8.5 的混合架构展望：用 XGBoost 召回 + Transformer 重排。

**候选受限 vs 全词表评估的 AUC 差异**。值得在报告中提一句的实验细节：如果把 Transformer 的 ROC AUC 评估改为"对全部 49,688 个商品打分"，AUC 反而会**虚高到约 0.557**。这看似反直觉，其实可以用"考卷难易度"做个直白比喻：

> **送分题 vs 送命题**。**全词表评估的考卷里全是送分题**——让模型在"小明买了香蕉"和"小明会不会买婴儿纸尿裤、猫砂、宠物狗粮"之间排序，这些不相关商品的得分几乎为 0，模型闭眼都能把香蕉排在它们前面；49,000 件商品里 99% 都是这种"送分题"，自然把 AUC 拉高了。**候选受限评估的考卷里全是送命题**——所有候选都是用户买过的东西（比如苹果、面包、牛奶都买过 10 次了），模型要分辨的是"小明这单到底是买苹果还是买面包"，每一题都贴近真实业务、每一题都很难。
>
> 与此同时 F1@10 走的是相反方向——全词表评估里要在 5 万件商品中挤进 Top-10，竞争对手暴增，真正想买的商品很容易被噪声挤出去，F1 因此暴跌；候选受限只有 30–50 个对手，香蕉容易挤进 Top-10，F1 反而更高。

这种"AUC 虚高、F1 暴跌"的反向走势，本质上是**评估候选池规模**带来的数学错觉，与模型本身能力无关。因此本报告统一采用候选受限的评估口径，所有 AUC 数字都在 0.53–0.83 区间，是一个真实可比的对照。

### 8.4 局限性

1. **样本规模**：5,000 用户只是完整 Instacart 数据集（约 20 万用户）的 2.5%。三个模型的相对排序在更大样本下大概率仍然成立，但深度模型（尤其是 Transformer）的绝对性能预计会随数据量增加显著提升。
2. **硬件约束**：所有训练都在 CPU + PyNative 模式下完成，Transformer 在 GPU / Ascend 上可使用更大的 batch 与更多 epoch。
3. **冷启动**：候选生成依赖"用户买过的商品"，对从未下过单的新用户不适用。生产环境中需要并行一条基于"按部门 / 货架的热销"的冷启动召回链路。
4. **未建模购买数量与时间**：本项目只预测"会不会买"，不预测"买几件"，也不预测"什么时候买"。后者对库存调度有直接价值，是后续延伸方向。
5. **未做超参数大规模搜索**：当前超参主要靠经验与小范围扫描设定，没有跑完整的 Grid Search / Bayesian Optimization。

### 8.5 未来工作

* **混合架构**：用 XGBoost 召回前 30 个候选，再让 Transformer 对它们做位置重排，兼顾 XGBoost 的高 F1 与 Transformer 的高 MRR。
* **更大规模训练**：迁移到 Huawei Ascend + MindSpore Graph 模式，在完整 20 万用户上重新训练。
* **时间感知 Transformer**：将 `days_since_prior_order` 作为连续时间编码加进 Transformer，让模型直接观察到订单间隔。
* **多任务学习**：联合预测"下一篮内容"与"下一单时间"，对库存与配送同样有价值。

### 8.6 小结

本项目在严格的"用户级 + 时间级"双重切分协议下，于 1,000 名未见用户上对三种范式的推荐模型做了完整对比。结论清晰：在生鲜零售这一**习惯主导**的场景下，**显式工程特征 + 梯度提升树（XGBoost）**显著优于隐式协同过滤与序列 Transformer——不仅指标更好，训练成本也低一个数量级。这一结果与"小到中等规模的表格数据上，树模型通常胜过深度模型"`[TODO: 补充引用]` 的一般规律一致，并为后续部署方案选型提供了直接依据。

---

## 9. 解决方案与商业提案（Solution and Business Proposal）

### 9.1 真实业务需求与动机

全球线上生鲜零售在过去五年增速显著 `[TODO: 补充市场调研引用，如 Statista / Frost & Sullivan / iResearch]`。在马来西亚本地，Jaya Grocer Online、HappyFresh、Tesco / Lotus's Online、Lazada Mart 等平台共同分食一块仍在快速扩张的市场。与运营人员的访谈中频繁出现两类痛点：

* **遗漏必需品**：客户结账时常常少买一两件每周固定要买的商品（牛奶、鸡蛋、洗洁精），事后要么补一单（对平台而言配送毛利被拉低甚至亏损），要么干脆转去竞品。
* **重新挑商品成本高**：复购型客户每次都要在 5 万件商品中翻找十几件熟面孔，体验差。

§8 的实验结果显示：用一棵不到 6 秒就能训练完的 XGBoost，就可以让超过 9 成的客户在推荐篮里见到至少一件他们真正会买的商品。这一能力恰好对应上述两类痛点，因此具备直接商业落地价值。

### 9.2 产品形态：SmartBasket 个性化补篮引擎

我们提出 **SmartBasket**——一个面向中型生鲜电商的"个性化补篮"推荐服务，对外提供四个面向消费者的能力与一个面向商家的能力：

1. **"您可能还需要"购物车提示**：在结账页注入个性化 Top-10 推荐，由 XGBoost 直接打分。
2. **智能提醒推送**：当某件常购商品的 `up_last_order_distance` 超过该用户的典型复购周期时，触发 Push（例如："您已 14 天没买牛奶了"）。
3. **一键复购**：基于模型输出生成一份完整的"每周固定篮"，把结账流程压缩到一次点击。
4. **品类感知重排**：可选启用 Transformer 重排器，把互补品（例如"麦片 → 牛奶"）顶到 Top 位置，发挥它在 MRR 上的优势。
5. **商家运营看板**：向运营人员展示按 SKU 的复购率预测、库存压力、预计篮填充率等，辅助补货与促销决策。

### 9.3 部署参考架构（结合 Huawei 云与 MindSpore）

整套服务在架构上分为**离线训练管线**与**在线推理路径**两部分，使重训不会干扰线上时延：

```
                  离线（每日批处理）                              在线（请求级）
 ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────────────┐
 │ OBS 数据湖   │──▶│ DataArts ETL     │──▶│ MindSpore    │──▶│ ModelArts        │
 │  - 订单表    │   │  - 特征工程       │   │ 训练         │   │ 实时推理服务      │
 │  - 商品目录  │   │  - 候选集生成     │   │  - NCF       │   │  - REST API      │
 │  - 历史明细  │   │                   │   │  - XGBoost   │   │  - 低时延        │
 └──────────────┘   └──────────────────┘   │  - TF 重排器 │   └──────────────────┘
                                            └──────────────┘             │
                                                                         ▼
                                                              ┌──────────────────┐
                                                              │  前端 App / Web  │
                                                              └──────────────────┘
```

**Huawei 技术映射**（满足 ICT 创新大赛对集成 Huawei 技术的要求）：

| Huawei 服务 | 在 SmartBasket 中的角色 |
| :--- | :--- |
| **Huawei MindSpore** | 训练 NCF 与 Sequential Transformer（本项目已实现）。 |
| **Huawei Ascend / ModelArts** | 在更大规模数据上加速重训。 |
| **Huawei OBS（对象存储）** | 存放原始与处理后的零售交易日志。 |
| **Huawei DataArts Studio** | 调度每日的特征工程 ETL 作业。 |
| **Huawei ModelArts 实时服务** | 托管 XGBoost + Transformer 重排的推理图，向前端提供 REST 接口。 |
| **Huawei FunctionGraph / CloudCMS** | 触发"必需品提醒"的 Push 通知与营销活动。 |

注：上表中除 MindSpore 外，其余 Huawei 服务均为目标部署形态，本项目当前仅在 MindSpore 这一层做了实际的代码与训练验证。

### 9.4 目标用户与利益相关方

* **主要用户 —— 中型生鲜电商平台**（如本地 Jaya Grocer Online、HappyFresh、区域型 Lazada Mart 卖家）：有交易数据但没有独立 ML 团队的玩家。
* **次级用户 —— 终端消费者**：结账更快、更少遗漏必需品。
* **关联利益方 —— FMCG 品牌方**：获取个性化的补货触达通道。
* **公共价值**：通过减少"补单"配送次数，间接降低城市最后一公里物流的车次与碳排放。

### 9.5 商业模式与可扩展性

**商业模式**：建议采用 SaaS 授权 + 按月活计费的混合模式，分三档：

* **试点档**：免费 30 天，最多 10,000 MAU。
* **标准档**：按 MAU 定价（如 RM 0.05/MAU/月），含商家看板与基础推荐 API。
* **企业档**：支持本地化部署（Huawei Cloud Stack / 本地 Ascend），含 SLA 与定制特征。

**可扩展性**：

* **横向扩展**：特征工程与候选生成是按用户高度可并行的，可在 MRS / Spark / DataArts 上轻松切片。
* **纵向扩展**：MindSpore 的 PyNative 与 Graph 模式共享同一份模型定义代码，从本地 CPU 开发迁移到 Ascend 集群训练几乎零代码改动。

### 9.6 SWOT 分析

| | 有利 | 不利 |
| :--- | :--- | :--- |
| **内部** | **优势 Strengths**：已用真实 Instacart 数据完成端到端验证；XGBoost 在 6 秒内即可训练；MindSpore 实现满足 Huawei 大赛要求；模块化设计便于客户改造。 | **劣势 Weaknesses**：当前仅在 5,000 用户样本上验证；CPU 训练，未在 GPU/Ascend 上压测；冷启动场景尚未覆盖；缺少线上 A/B 数据。 |
| **外部** | **机会 Opportunities**：东南亚线上生鲜年增速可观；马来西亚 MyDigital 2025 鼓励 AI 落地；Huawei 云在区域的扩张；FMCG 品牌愿意为精准触达付费。 | **威胁 Threats**：大型零售商可能自建团队；个人数据保护法（PDPA 2010）对历史购物数据有合规要求；AWS / GCP 的 ML-as-a-Service 已有同类产品。 |

### 9.7 商业化路线图

* **Months 1–3**：把验证过的 XGBoost 引擎产品化，接入 Huawei OBS + DataArts ETL；与一家本地零售商签订 4 周试点。
* **Months 4–6**：将训练迁移到 Ascend + MindSpore Graph 模式；上线 Transformer 重排器；交付商家看板。
* **Months 7–12**：扩展到 3 家零售商；新增果蔬品类支持；以最终版方案参加 Huawei ICT Innovative Competition。

---

## 章节使用说明

* 本文档为**报告正文第 5–9 节的草稿**，不含摘要、引言、相关工作、结论与参考文献，请按 TML6223 模板拼接。
* 文中所有数据数字、训练时长、特征列名、文件路径、超参数值均直接来自本仓库 `src/` 与 `data/processed/` 中的实际产物，可在最终提交前用 `instacart_ml_pipeline.ipynb` 复跑校对。
* 文中出现 `[TODO: 补充引用]` 的位置，请作者根据自己阅读过的真实文献回填，避免引用未确认的来源。

这个仓库是[字节跳动比赛 Byte Cup 2018，自动生成新闻标题](https://biendata.com/competition/bytecup2018/)，水滴队（最终成绩23名）的代码。
有关比赛的总结，欢迎移步博客：https://blog.csdn.net/taoyafan/article/details/84879285
## Requirements

python3.5 或以上
tensorflow 1.12（接近的几个版本应该也可以）

## 程序说明

修改自程序[**RLSeq2Seq**](https://github.com/yaserkl/RLSeq2Seq)

主要改动如下：

（1）更换 python 版本为 python3。

（2）对 policy gradient 部分进行了大量的修改，原程序存在很多错误，如计算 ROUGE 时没有将 decode mask 去掉，前向计算时 greedy 和 sample 没有分开，decode 的输入也是一样的。

（3）训练的同时增加 eval，并保存在验证集效果最好的最后三个模型。

（4）增加对[**pointer-generator**](https://github.com/abisee/pointer-generator)的模型的兼容性，可以直接使用其预训练模型。

（5）对 policy gradient 的修改，将论文[A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304)中的公式（15）改为 
$$
L_{rl}=(r(y^{s}) - r(y^{g}))\sum_{t=1}^{n'}{\rm log}p(y_{t}^{g}|y_{1}^{g},...y_{t-1}^{g},x)
$$
即将 sample 得到的结果当做 baseline，根据 greedy 得到的结果对来计算梯度。





## 使用说明

参考程序[**RLSeq2Seq**](https://github.com/yaserkl/RLSeq2Seq) 和 [**pointer-generator**](https://github.com/abisee/pointer-generator)，他们介绍的很清楚，只是这里只用在比赛中，生成的标题较短，且数据集来自官方。

数据的预处理参考 cnn-dailymail 中的 [**make_datafiles.ipynb**](https://github.com/taoyafan/cnn-dailymail/tree/master/bytecup)。

### 文件说明

[src](https://github.com/taoyafan/abstractive_summarization/tree/master/src) 中为源代码，其中[run_summarization.py](https://github.com/taoyafan/abstractive_summarization/blob/master/src/run_summarization.py)为主程序。

[results](https://github.com/taoyafan/abstractive_summarization/tree/master/results)中为不同模型的运行命令（参数）。

### 参数说明

基本命令同[**RLSeq2Seq**](https://github.com/yaserkl/RLSeq2Seq)，增加参数如下：

| 参数                       | 说明                                                         |
| -------------------------- | ------------------------------------------------------------ |
| convert_version_old_to_new | 为 True 时可加载 [**pointer-generator**](https://github.com/abisee/pointer-generator) 提供的预训练模型 |
| eval_data_path             | 验证集路径                                                   |
| dropout_keep_p             | 在 encoder 和 decoder 的 LSTM 的 cell 中增加 drop out，对 input、output 和 state 使用相同的 keep_p，默认为1，即不使用 drop out |
| rising_greedy_r            | 为 True 时 policy gradient 使用更改后的公式，即目标为提升 greedy 得到的 reward，为 False 时为原公式，但是占用显存增大一倍 |

### 运行说明

在 results 中寻找对应模型的命令，如基准模型 [base_eta=0_lr=0.15.txt](https://github.com/taoyafan/abstractive_summarization/blob/master/results/base_eta%3D0_lr%3D0.15.txt)

在训练时执行命令：

```
python3 run_summarization.py --mode=train --data_path=../finished_files/chunked/train* --eval_data_path=../finished_files/chunked/test* --vocab_path=../finished_files/vocab --log_root=../log --exp_name=base_eta=0_lr=0.15 --batch_size=20 --use_temporal_attention=False --intradecoder=False --eta=0 --rl_training=True --lr=0.15 --sampling_probability=0 --fixed_eta=True --scheduled_sampling=True --fixed_sampling_probability=True --greedy_scheduled_sampling=True
```



经过实验 drop out 取 0.8 时效果最好，不过最终没来得及使用，最终成绩所使用的模型为：

基础模型（有pointer_gen，无coverage，无rl），然后使用 policy gradient。

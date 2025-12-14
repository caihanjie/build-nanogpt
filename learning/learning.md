# GPT DEV

## ipynb-Building a GPT
### 打开数据集

### 设置词表大小

### encode（tokenizor）和decode

### encode整个数据集

### 划分训练集和验证集

### getbatch得到x和y 
x的T是block_size（context），y是x向右偏移1位

tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])
when input is tensor([18]) the target: 47
when input is tensor([18, 47]) the target: 56
when input is tensor([18, 47, 56]) the target: 57
when input is tensor([18, 47, 56, 57]) the target: 58
when input is tensor([18, 47, 56, 57, 58]) the target: 1
when input is tensor([18, 47, 56, 57, 58,  1]) the target: 15
when input is tensor([18, 47, 56, 57, 58,  1, 15]) the target: 47
when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target: 58

### Bigram Language Model
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

embedding: 把token转为词嵌入（为了简易先让嵌入维度等于词表大小（也就是等于预测结果））

forward

输出logit和target计算cross_entropy（相当于根据每个token embedding得到的下一个token预测结果和真实的下一个token计算交叉熵）

generate

传入idx，取最后一行的logit（最后一个预测结果就是下一个token的预测），经过softmax后进行采样，然后与前面传入的idx进行拼接，最终得到全部生成的idx，进行decode得到文本

## v1
### estimate_loss()
获取多个trani和val的batch计算loss均值

### 设置训练loop和optimizer

## v2 step1
### position embedding
和原始的token序列的embedding相加

由于pos_embd只在blocksize范围内，generate函数中当context超过时要做截断

### 把token_embedding_table的嵌入维度改成n_embd(也就是d_model)

### 添加一个lm_head 把词嵌入映射到词表大小

## ipynb-The mathematical trick in self-attention
### 从最简单的做法开始，每个token的embedding是前面token的连加和
用for循环实现

### 每个token的embedding是前面token的连加和的均值
用for循环实现

### 进一步用矩阵实现加权聚合，创建一个全1的下三角矩阵t
t@x 就可以实现每个token的embedding是前面token的连加和

t = t / torch.sum(t, 1, keepdim=True)

再用t除以每一行的总和，得到的权重矩阵
t@x 就可以实现，每个token是对前面的全部token的均值

### 进一步引入softmax
wei = wei.masked_fill(tril == 0, float('-inf'))

把wei中下三角矩阵为0的地方用-inf填充

得到的wei进行softmax就得到和上一部的t一样的效果

### 实现selfattention
分别创建qkv三个线性层，输出维度为head_size

Q @ K.T得到 (T,T)的注意力矩阵wei

然后对wei进行mask，也就是用下三角进行mask，为0的地方填充为-inf，在进行softmax就可以屏蔽掉每个token对后面token的权重

最后wei@v得到output

### sclaed-attention
wei = q @ k.transpose(-2, -1) * head_size**-0.5

使得wei的方差也是1

### temperature
在softmax前乘一个数，可以让softmax变尖锐或者平缓

## v2 step2
### 加入 class Head()

### 实现 class Mutil-Head-Attention
把n_embd拆成几个head

把每个head得到的output再拼接回到e_embd

### 实现FeedForward
ffwd是在selfattention后的mlp

### 修改二元语言模型
在原先词嵌入+pos嵌入后直接进入lm_head变成经过attention实现token间的交互，再通过ffwd的计算再映射到词表上得到下一个token

### 完成大致实现

## v3（最终）
### 新增 class Block
把self-attention和ffwd合并成一个Block

### 在二元语言模型中实现多个Block的堆叠

### 在Block中加入残差链接 Residual net
- self-attention后
- ffwd后

### Block中加入LayerNorm
ln对每个样本进行归一化
- self-attention前
- ffwd前
- block堆叠的最后

### 在MultiHeadSelfattention最后增加一个proj层

### 加入dropout
- 在sa_head计算中的注意力矩阵后
- 在ffwd后
- 在multihead-selfattention的输出后

### 超参数
```
batch_size = 64
block_size = 256
max_iters = 6000
learning_rate = 3e-4
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2
```
## v4
### 多卡训练ddp





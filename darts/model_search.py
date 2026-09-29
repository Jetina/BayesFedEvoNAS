import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.genotypes import PRIMITIVES, Genotype
from darts.operations import OPS, FactorizedReduce, ReLUConvBN
from darts.utils import count_parameters_in_MB


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:#PRIMITIVES:grnotype里面定义的，8个操作
            op = OPS[primitive](C, stride, False)#ops里面存储各种操作函数
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))#给池化操作后面加一个归一化层
            self._ops.append(op)#把这些op都放在预先定义好的modulelist里

    def forward(self, x, weights):
        # w is the operation mixing weights. see equation 2 in the original paper.
        return sum(w * op(x) for w, op in zip(weights, self._ops))#op(x)就是对输入x做一个相应的操作 w1*op1(x)+w2*op2(x)+...+w8*op8(x)
                                                                #也就是对输入x做8个操作并乘以相应的权重，把结果加起来


class Cell(nn.Module):

    '''
    DARTS代码的模型搜索部分：
    参数：单元中节点的数量，重复单元的次数，C开头的三个：前两个节点的输入通道和当前节点的输入通道数。reduction表示该单元是否减少输入size
    reduction_prev:前一个单元是否减少输入size
    '''
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        '''
         Args:  
            n_nodes: # of intermediate n_nodes  中间节点个数
            C_prev_prev: C_out[k-2]    第 k-2 个cell的输出通道数, 与输入node 1 相连
            C_prev : C_out[k-1]    第 k-1 个cell的输出通道数, 与输入node 2 相连
            C   : C_in[k] (current)     当前是第 k 个cell, 输入通道数为 C
            reduction_p: flag for whether the previous cell is reduction cell or not  前一个cell是否缩小了尺寸
            reduction: flag for whether the current cell is reduction cell or not     当前cell是否缩小了尺寸
        '''
        #input nodes的结构固定不变，不参与搜索
        #决定第一个input nodes的结构，取决于前一个cell是否是reduction
        if reduction_prev:#如果前一个单元是缩减块
            # 如果第 k-1 个cell是reduction cell, 前面的输出尺寸是缩小的, 因此k-2和k的尺寸也要缩小
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)#将张量空间维度减少两倍，#第一个input_nodes是cell k-2的输出，cell k-2的输出通道数为C_prev_prev，所以这里操作的输入通道数为C_prev_prev
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)#核大小是2，步长1

        #第二个输入节点的结构
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)#核大小1，步长1
        self._steps = steps#每个cell中4个节点链接状态待确定
        self._multiplier = multiplier

        self._ops = nn.ModuleList()#构建operation的modulelist
        self._bns = nn.ModuleList()
        '''
        i:0,1,2,3
        共24
        根据是否满足if条件，判断步长是2还是1
       
        '''
        #遍历4个intermediate nodes构建混合操作
        for i in range(self._steps):#对于每个层有2+i个操作，i是当前层索引。遍历当前节点i的所有前驱节点
            '''
            # 对第 i 个node来说, 他有 j 个前驱node  
            # 每个node的input都由 前 2 个cell的输出和当前cell的前面的node组成 (0..i-1)
            # 例如 i = 1 时,  j = 0, 1, 2, 其中 2 是前面的node i = 0
            '''
            for j in range(2 + i):#对第i个节点来说，他有j个前驱节点（每个节点的input都由前两个cell的输出和当前cell的前面的节点组成）
                # 只有自身为 reduction cell, 且j<2, 才扩大步长, 缩小尺寸
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)#op是构建两个节点之间的混合
                self._ops.append(op)#所有边的混合操作添加到ops

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]#当前节点的前驱节点
        offset = 0
        #遍历每个intermediate nodes，得到每个节点的output
        for i in range(self._steps):
            # s为当前节点i的output，在ops找到i对应的操作，然后对i的所有前驱节点做相应的操作（调用了MixedOp的forward），然后把结果相加
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)#把当前节点i的output作为下一个节点的输入
            #例如：states中为[s0,s1,b1,b2,b3,b4] b1,b2,b3,b4分别是四个intermediate output的输出
        return torch.cat(states[-self._multiplier:], dim=1)#对intermediate的output进行concat作为当前cell的输出
                                                       #dim=1是指对通道这个维度concat，所以输出的通道数变成原来的4倍


class InnerCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, weights):
        super(InnerCell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        # len(self._ops)=2+3+4+5=14
        offset = 0
        keys = list(OPS.keys())
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                weight = weights.data[offset + j]
                choice = keys[weight.argmax()]
                op = OPS[choice](C, stride, False)
                if 'pool' in choice:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)
            offset += i + 2

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class ModelForModelSizeMeasure(nn.Module):
    """
    This class is used only for calculating the size of the generated model.
    The choices of opeartions are made using the current alpha value of the DARTS model.
    The main difference between this model and DARTS model are the following:
        1. The __init__ takes one more parameter "alphas_normal" and "alphas_reduce"
        2. The new Cell module is rewriten to contain the functionality of both Cell and MixedOp
        3. To be more specific, MixedOp is replaced with a fixed choice of operation based on
            the argmax(alpha_values)
        4. The new Cell class is redefined as an Inner Class. The name is the same, so please be
            very careful when you change the code later
        5.

    """

    def __init__(self, C, num_classes, layers, criterion, alphas_normal, alphas_reduce,
                 steps=4, multiplier=4, stem_multiplier=3):
        super(ModelForModelSizeMeasure, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C  # 3*16
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # for layers = 8, when layer_i = 2, 5, the cell is reduction cell.
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = InnerCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                                 alphas_reduce)
            else:
                reduction = False
                cell = InnerCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                                 alphas_normal)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input_data):
        s0 = s1 = self.stem(input_data)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1)
            else:
                s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, device,steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        # print(Network)
        self._C = C#stem的通道数
        self._num_classes = num_classes#类数
        # self._num_classes = 2  # 类数
        self._layers = layers#层数
        self._criterion = criterion#损失函数
        self._steps = steps#每个单元中中间结点的数量
        self._multiplier = multiplier#中间节点输出通道数量
        self._stem_multiplier = stem_multiplier#网络stem通道数的乘数

        self.device = device#设备（gpu or cpu）

        C_curr = stem_multiplier * C  # 3*16
        self.stem = nn.Sequential(#stem用Sequential块定义，包含一个二维卷积、一个归一化层.快速降低特征图的分辨率，减小计算量
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),#参数：输入通道、输出通道、核大小、padding、偏差
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C #通道数更新
        self.cells = nn.ModuleList()
        reduction_prev = False

        # for layers = 8, when layer_i = 2, 5, the cell is reduction cell.
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:#//向下取整。迭代数从layyers//3或2*layers//3，假设layer=8，则i=2或5是真
                C_curr *= 2#乘2，设置缩减块
                reduction = True
            else:
                reduction = False#不变，缩减块还是flase
            # 这部分是DARTS部分的代码，Cell部分
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]#在modulelist中增加一个cell
            C_prev_prev, C_prev = C_prev, multiplier * C_curr#更新当前通道数

        self.global_pooling = nn.AdaptiveAvgPool2d(1)#构建一个平均池化
        self.classifier = nn.Linear(C_prev, num_classes)#构建一个线性分类器
        #初始化a
        self._initialize_alphas()
    #新建network，复制 a参数
    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.device).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            # reduction cell和normal cell的共享参数a不同
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            #每个cell之间的连接，s0来自上上个cell输出，s1来自上一个cell的输出
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    #初始化参数
    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))#k表示所有Cells中所有可选操作的数量之和
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))#nn.Parameter 对象，以便它能够自动更新梯度
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))#参数被初始化为服从标准正态分布的随机数乘以 1e-3，以便在模型的初始阶段进行较小的随机初始化。
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def new_arch_parameters(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops)).to(self.device)
        alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops)).to(self.device)
        _arch_parameters = [
            alphas_normal,
            alphas_reduce,
        ]
        return _arch_parameters

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _isCNNStructure(k_best):
            return k_best >= 4

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            cnn_structure_count = 0
            for i in range(self._steps):
                #对每个中间节点
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k

                    if _isCNNStructure(k_best):
                        cnn_structure_count += 1
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene, cnn_structure_count

        with torch.no_grad():
            gene_normal, cnn_structure_count_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
            gene_reduce, cnn_structure_count_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            genotype = Genotype(
                normal=gene_normal, normal_concat=concat,
                reduce=gene_reduce, reduce_concat=concat
            )
        return genotype, cnn_structure_count_normal, cnn_structure_count_reduce

    def get_current_model_size(self):
        model = ModelForModelSizeMeasure(self._C, self._num_classes, self._layers, self._criterion,
                                         self.alphas_normal, self.alphas_reduce, self._steps,
                                         self._multiplier, self._stem_multiplier)
        size = count_parameters_in_MB(model)
        # This need to be further checked with cuda stuff
        del model
        return size

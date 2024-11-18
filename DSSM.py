import torch
from ...basic.layers import MLP, EmbeddingLayer
from torch.nn import Embedding

class DSSM(torch.nn.Module):
    """Deep Structured Semantic Model

    Args:
        user_features (list[Feature Class]): 用户特征.
        item_features (list[Feature Class]): 物品特征.
        sim_func (str): 相似度计算公式.
        temperature (float): 温度参数
        user_params (dict): 用户塔参数:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        item_params (dict): 物品塔参数:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
    """

    def __init__(self, user_features, item_features, user_params, item_params, sim_func="cosine", temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        # 计算两个塔结果embedding之间的相似度，也可以使用LSH等方法
        self.sim_func = sim_func
        # 温度系数
        self.temperature = temperature
        # 分别计算user和item的emb维度之和
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in item_features])
        # 构建embedding层，这里是input为特征列表，output对应特征的字典
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.mode = None

    def forward(self, x):
        # user塔
        user_embedding = self.user_tower(x)
        # item塔
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        # 计算相似度：cosine-> similarity
        if self.sim_func == "cosine":
            y = torch.cosine_similarity(user_embedding, item_embedding, dim=1)
        elif self.sim_func == "dot":
            y = torch.mul(user_embedding, item_embedding).sum(dim=1)
        else:
            raise ValueError("similarity function only support %s, but got %s" % (["cosine", "dot"], self.sim_func))
        y = y / self.temperature
        return torch.sigmoid(y)

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  # [batch_size, num_features*deep_dims]
        user_embedding = self.user_mlp(input_user)  # [batch_size, user_params["dims"][-1]]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        input_item = self.embedding(x, self.item_features, squeeze_dim=True)  # [batch_size, num_features*embed_dim]
        item_embedding = self.item_mlp(input_item)  # [batch_size, item_params["dims"][-1]]
        return item_embedding


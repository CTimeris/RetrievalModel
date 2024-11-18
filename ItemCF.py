import math
import pickle
from collections import defaultdict
from tqdm import tqdm


def itemcf_sim(user_item_dict):
    """
    物品与物品之间的相似性矩阵计算
    :param user_item_dict: 用户交互序列  {user1:{item1, item2, ...}, user2...}
    :return: 相似性矩阵

    base公式：已有i，计算与j的相似度 = 同时喜欢用户i和j的用户数 / 喜欢i的用户数
        问题：若j热门物品，则与任何物品相似度都高
        打压热门物品：除以一个j被喜欢的数量：同时喜欢用户i和j的用户数 / 根号（喜欢i的用户数 * 喜欢j的用户数）
        进一步打压活跃用户：上面的分子 相当于每个用户权重都为1，可以根据其交互数n设置权重 (1/log(1+n))
    """
    i2i_sim = {}
    item_cnt = defaultdict(int)  # 物品的总点击次数（喜欢物品i的用户总数）

    for user, item_list in tqdm(user_item_dict.items()):
        for i in item_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})  # setdefault查找指定键,存在则返回对应的值,不存在则在字典中添加该键，值设为默认值
            for j in item_list:
                if i == j:
                    continue
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += 1 / math.log(len(item_list) + 1)  # 打压活跃用户

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])  # 打压热门物品

    # 将得到的相似性矩阵保存到本地
    # pickle.dump(i2i_sim_, open('itemcf_i2i_sim.pkl', 'wb'))
    return i2i_sim_


def itemcf_rec(user_id, user_item_dict, i2i_sim, last_n, sim_item_topk, recall_item_num, item_topk_click):
    """
    根据历史交互的n个物品，每个物品找回最相似的k个物品，计算nk个物品的兴趣分数，取前num个作为召回结果
    :param user_id: 用户id
    :param user_item_dict: {user1: {item1, item2..}...}
    :param i2i_sim: 字典，物品相似性矩阵
    :param last_n: 最近交互的n个物品
    :param sim_item_topk: 与当前物品最相似的前k个物品
    :param recall_item_num: 最后的召回物品数量
    :param item_topk_click: 热门物品池，用于补充
    :return: 召回的物品列表 {item1: score1, item2: score2...}
    """
    # 获取用户历史交互的文章
    user_hist_items = list(user_item_dict[user_id])
    # 不够n个就用全部，超过就取n个
    if len(user_hist_items) > last_n:
        n_hist_items = user_hist_items[:last_n]
    else:
        n_hist_items = user_hist_items

    item_rank = {}
    for loc, i in enumerate(n_hist_items):
        # 每个历史交互物品都找最相似的k个
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items:
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij  # 找出来的物品的总分

    # 不足，用热门商品补充
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100  # 给个负数，让热门物品排在最后
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank


if __name__ == '__main__':
    # 用户-物品索引
    user_item = {
        1: {1, 3, 5, 7, 10},
        2: {1, 2, 3, 4, 6, 7, 8},
        3: {1, 4, 7, 10},
        4: {2, 3, 4, 5, 9},
        5: {4, 5, 6, 9, 10},
        6: {3, 7},
        7: {4, 6, 8, 10}
    }
    hot_items = {3, 100, 1000}  # 热门物品池

    sim_matrix = itemcf_sim(user_item)
    # 线上召回通常在计算相似度矩阵后维护一个索引{item: {item1, item2 ...}，快速找到top_k相似物品
    last_n = 5  # 最近的n个交互
    sim_k = 2  # 每个物品找k个相似的
    recall_num = 3  # 一共召回num个
    userid = 1
    recall_items = itemcf_rec(userid, user_item, sim_matrix, last_n, sim_k, recall_num, hot_items)
    print(recall_items)

# ItemCF只要两个物品重合比例高的用户就判定为相似，假如用户在同一个小圈子（同一个群，被分享）
# 那么可能物品本来并不会有关联的物品因为它们得到了关联（不是因为用户兴趣而关联的）
# 改进：Swing模型为了防止ItemCF重合的用户来自同一个圈子，将用户重合度加入相似度的计算公式
# 统计同时喜欢i和喜欢j的用户集合，统计其中每两个用户的重合度，作为相似度的惩罚。

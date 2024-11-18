import math
import pickle
from collections import defaultdict
from tqdm import tqdm


def usercf_sim(item_user_dict):
    """
    用户相似性矩阵计算，与itemcf几乎一样
    :param item_user_dict: {{item1: user1, user2...}, item2:...}
    :return: 用户相似度矩阵
    """
    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_list in tqdm(item_user_dict.items()):
        for u in user_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v in user_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                u2u_sim[u][v] += 1 / math.log(len(user_list) + 1)

    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # 将得到的相似性矩阵保存到本地
    # pickle.dump(u2u_sim_, open('usercf_u2u_sim.pkl', 'wb'))
    return u2u_sim_


def usercf_rec(user_id, user_item_dict, u2u_sim, last_n, sim_user_topk, recall_item_num, item_topk_click):
    """
    每个用户找到最相似的k个用户，找它们近期最感兴趣的n个物品，取分数最高的nk个物品作为召回结果
    :param user_id: 用户id
    :param user_item_dict: {user: {item1, item2,...}}
    :param u2u_sim: 用户相似度矩阵
    :param last_n: 近期n个交互
    :param sim_user_topk: 最相似的k个用户
    :param recall_item_num: 召回的物品数量
    :param item_topk_click: 热门池
    :return: 召回的物品列表 {item1: score1, item2: score2...}
    """
    item_rank = {}
    # 当前用户交互过的物品
    now_user_items = user_item_dict[user_id]
    # 遍历前k个相似用户
    for u, wij in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        # 每个用户找近期交互的n个物品
        u_hist_items = list(user_item_dict[u])
        # 不够n个就用全部，超过就取n个
        if len(u_hist_items) > last_n:
            u_hist_items = u_hist_items[:last_n]
        # 遍历每个物品，加分
        for i in u_hist_items:
            item_rank.setdefault(i, 0)
            item_rank[i] += wij

    # 不足用热门商品补全
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
    # 物品-用户索引
    item_user = {
        1: {1, 2, 3},
        2: {2, 4},
        3: {1, 2, 4, 6},
        4: {2, 3, 4, 5},
        5: {1, 4, 5},
        6: {2, 5, 7},
        7: {2, 3, 6},
        8: {2, 7},
        9: {4, 5},
        10: {1, 3, 5, 7}
    }
    sim_matrix = usercf_sim(item_user)
    last_n = 5  # 最近的n个交互
    sim_k = 2  # 每个物品找k个相似的
    recall_num = 3  # 一共召回num个
    userid = 1
    recall_items = usercf_rec(userid, user_item, sim_matrix, last_n, sim_k, recall_num, hot_items)
    print(recall_items)

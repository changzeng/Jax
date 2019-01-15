# -*- coding: utf-8 -*-
"""
# @Time    : 2018/5/26 下午5:03
# @Author  : zhanzecheng
# @File    : model.py
# @Software: PyCharm
"""
import utils
import math
import heapq


class Node(object):
    """
    建立字典树的节点
    """

    def __init__(self, char):
        self.char = char
        # 字符结束标志
        self.word_finish = False
        # 用来计数
        self.count = 0
        # 用来存放节点
        self.child = {}
        # 方便计算 左右熵
        # 判断是否是后缀（标识后缀用的，也就是记录 b->c->a 变换后的标记）
        self.isback = False
        self.back_count = 0


class TrieNode(object):
    """
    建立前缀树，并且包含统计词频，计算左右熵，计算互信息的方法
    """

    def __init__(self, node, data=None, PMI_limit=20):
        """
        初始函数，data为外部词频数据集
        :param node:
        :param data:
        """
        self.root = Node(node)
        self.PMI_limit = PMI_limit
        if not data:
            return
        node = self.root
        for key, values in data.items():
            new_node = Node(key)
            new_node.count = int(values)
            new_node.word_finish = True
            node.child[key] = new_node

    def has_child(self, node, char):
        return node.child.get(char, None) is not None

    def add(self, word):
        """
        添加节点，对于左熵计算时，这里采用了一个trick，用a->b<-c 来表示 cba
        具体实现是利用 self.isback 来进行判断
        :param word:
        :return:  相当于对 [a, b, c] a->b->c, [b, c, a] b->c->a
        """
        node = self.root
        # 正常加载
        for count, char in enumerate(word):
            found_in_child = False
            # 在节点中找字符
            if self.has_child(node, char):
                node = node.child[char]
                found_in_child = True

            # 顺序在节点后面添加节点。 a->b->c
            if not found_in_child:
                new_node = Node(char)
                node.child[char] = new_node
                node = new_node

            # 判断是否是最后一个节点，这个词每出现一次就+1
            if count == len(word) - 1:
                node.count += 1
                node.word_finish = True

        # 建立后缀表示
        length = len(word)
        node = self.root
        if length == 3:
            word = list(word)
            word[0], word[1], word[2] = word[1], word[2], word[0]

            for count, char in enumerate(word):
                found_in_child = False

                if self.has_child(node, char):
                    node = node.child[char]
                    found_in_child = True

                # 顺序在节点后面添加节点。 b->c->a
                if not found_in_child:
                    new_node = Node(char)
                    node.child[char] = new_node
                    node = new_node

                # 判断是否是最后一个节点，这个词每出现一次就+1
                if count == len(word) - 1:
                    node.back_count += 1
                    node.isback = True

    def search_one(self):
        """
        计算互信息: 寻找一阶共现，并返回词概率
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        # 计算 1 gram 总的出现次数
        total = 0
        for char, child in node.child.items():
            if child.word_finish is True:
                total += child.count

        # 计算 当前词 占整体的比例
        for char, child in node.child.items():
            if child.word_finish is True:
                result[child.char] = utils.get_div(child.count, total)
        return result, total

    def search_bi(self):
        """
        计算互信息: 寻找二阶共现，并返回 log2( P(X,Y) / (P(X) * P(Y)) 和词概率
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        total = 0
        # 1 grem 各词的占比，和 1 grem 的总次数
        one_dict, total_one = self.search_one()
        for char_p, parent in node.child.items():
            for char_s, son in parent.child.items():
                if son.word_finish is True:
                    total += son.count

        for char_p, parent in node.child.items():
            for char_s, son in parent.child.items():
                if son.word_finish is True:
                    # 互信息值越大，说明 a,b 两个词相关性越大
                    PMI = math.log(max(son.count, 1), 2) - math.log(total, 2) - math.log(one_dict[char_p], 2) - math.log(one_dict[char_s], 2)
                    # 这里做了PMI阈值约束
                    if PMI > self.PMI_limit:
                        # 例如: dict{ "a_b": (PMI, 出现概率), .. }
                        result[char_p + '_' + char_s] = (PMI, utils.get_div(son.count, total))
        return result

    def search_left(self):
        """
        寻找左频次
        统计左熵， 并返回左熵 (bc - a 这个算的是 abc|bc 所以是左熵)
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for char_p, parent in node.child.items():
            for char_s, son in parent.child.items():
                total = 0
                p = 0.0
                for _, ch in son.child.items():
                    if ch.isback:
                        total += ch.back_count
                for _, ch in son.child.items():
                    if ch.isback:
                        p += utils.get_div(ch.back_count, total) * math.log(utils.get_div(ch.back_count, total), 2)
                # 计算的是信息熵
                result[parent.char + "_" + son.char] = -p
        return result

    def search_right(self):
        """
        寻找右频次
        统计右熵，并返回右熵 (ab - c 这个算的是 abc|ab 所以是右熵)
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for char_p, parent in node.child.items():
            for char_s, son in parent.child.items():
                total = 0
                p = 0.0
                for _, ch in son.child.items():
                    if ch.word_finish is True:
                        total += ch.count
                for _, ch in son.child.items():
                    if ch.word_finish is True:
                        p += utils.get_div(ch.count, total) * math.log(utils.get_div(ch.count, total), 2)
                # 计算的是信息熵
                result[char_p + "_" + char_s] = -p
        return result

    def find_word(self, N):
        bi = self.search_bi()
        left = self.search_left()
        right = self.search_right()
        result = {}
        for key, values in bi.items():
            result[key] = (values[0] + min(left[key], right[key])) * values[1]

        result = heapq.nlargest(N, result.items(), key=lambda x: x[1])

        add_word = {}
        for d in result:
            new_word = "".join(d[0].split('_'))
            add_word[new_word] = d[1]

        return result, add_word

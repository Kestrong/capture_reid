#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
from collections import defaultdict
from functools import reduce

import faiss
import numpy as np

logger = logging.getLogger("tasks")


def check_components(components):
    check_set = set()
    for cc in components:
        for c in cc:
            if c in check_set:
                print(cc)
            check_set.add(c)


def filter_single_by_features(c, features, th, valid_percent=0.5):
    sub_feats = features[c]
    total = len(c)
    scores = np.dot(sub_feats, sub_feats.T)
    th_count = total * valid_percent

    image_indexes = np.array(c)
    valid_datas = np.sum(scores >= th, axis=1)
    valid_indexes = np.where(valid_datas >= th_count)[0]
    valid_image_indexes = image_indexes[valid_indexes].tolist()
    return valid_image_indexes


def filter_clusters_by_features(clusters, features, th, labels=None, valid_percent=0.5):
    new_clusters = []
    image_count = 0
    for i, c in enumerate(clusters):
        if i % 10000 == 0:
            logger.info("filter_clusters_by_features i {} image_count {} new_clusters {}".format(i, image_count, len(new_clusters)))
        sub_feats = features[c]
        total = len(c)
        image_count += total
        scores = np.dot(sub_feats, sub_feats.T)
        th_count = total * valid_percent

        image_indexes = np.array(c)
        valid_datas = np.sum(scores >= th, axis=1)
        valid_indexes = np.where(valid_datas >= th_count)[0]
        invalid_indexes = np.where(valid_datas < th_count)[0]
        valid_image_indexes = image_indexes[valid_indexes].tolist()
        # 这样会导致标签会粘连在一起
        labels_sets = set()
        for index in c:
            if labels[index] != -1:
                labels_sets.add(labels[index])
        if len(labels_sets) > 1:
            logger.warning("filter_clusters_by_features conains %s", labels_sets)
        # assert len(labels_sets) <= 1

        for invalid_index in invalid_indexes:
            image_index = image_indexes[invalid_index]
            if labels is None or labels[image_index] == -1:
                new_clusters.append([image_index])
            else:
                valid_image_indexes.append(image_index)
        new_clusters.append(valid_image_indexes)
    return new_clusters


def gen_graph(knns, k, th):
    n = len(knns)
    # dict
    G = defaultdict(dict)

    edges_count = 0
    min_score = 1000
    max_score = 0
    for i in range(n):
        if i % 100000 == 0:
            logger.info("gen_graph %s", i)
        for j in range(k):
            dist, ner = knns[i]
            score = dist[j]
            ner_index = int(ner[j])
            if score >= th and i != ner_index:
                if i < ner_index:
                    pair = (i, ner_index)
                else:
                    pair = (ner_index, i)
                if pair[1] in G[pair[0]]:
                    continue
                # dict
                G[pair[0]][pair[1]] = score
                G[pair[1]][pair[0]] = score

                edges_count += 1
                max_score = max(max_score, score)
                min_score = min(min_score, score)

    return G, edges_count, min_score, max_score


def gen_graph_by_clusters(clusters, G, edges_count):
    for label in clusters:
        if label == -1:
            continue
        c = clusters[label]
        # while len(c) > 1:
        #     i = c.pop()
        #     for j in c:
        #         # list
        #         G[i][j] = 1.0
        #         G[j][i] = 1.0
        #         edges_count += 1
        # 新版的只要顺序相连就好
        for index in range(len(c) - 1):
            i = c[index]
            j = c[index + 1]
            G[i][j] = 1.0
            G[j][i] = 1.0
            edges_count += 1

    return G, edges_count


def connected_components_constraint(nodes_list, max_sz, th=None, full_graph=None, labels=None, features=None, th_knn=0.5):
    '''
    only use edges whose scores are above `th`
    if a component is larger than `max_sz`, all the nodes in this component are added into `remain` and returned for next iteration.
    '''
    assert labels is not None
    result = []
    remain = set()
    nodes = set(nodes_list)
    while nodes:
        n = nodes.pop()
        group = {n}

        # 这个为了约束一簇只能包含最多一个标签
        group_labels = set()
        if labels[n] != -1:
            group_labels.add(labels[n])

        queue = [n]
        valid = True
        while queue:
            n = queue.pop(0)
            if th is not None:
                # dict
                neighbors = set()
                for l in full_graph[n]:
                    # 原有的不用判断是否在nodes,现在有可能同一批的相似数据，拆成两层处理
                    if full_graph[n][l] >= th and l in nodes:
                        if len(group_labels) == 0:
                            neighbors.add(l)
                            if labels[l] != -1:
                                group_labels.add(labels[l])
                        else:
                            if labels[l] == -1 or labels[l] in group_labels:
                                neighbors.add(l)

            else:
                # dict
                neighbors = set()
                # 原有的不用判断是否在nodes,现在有可能同一批的相似数据，拆成两层处理
                for l in full_graph[n].keys():
                    if l in nodes:
                        if len(group_labels) == 0:
                            neighbors.add(l)
                            if labels[l] != -1:
                                group_labels.add(labels[l])
                        else:
                            if labels[l] == -1 or labels[l] in group_labels:
                                neighbors.add(l)

            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)

        #在循环中会一个拆成两个
        if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
            # if this group is larger than `max_sz`, add the nodes into `remain`
            # print("----------> valid False", len(group) > max_sz, len(remain.intersection(neighbors)) > 0)
            valid = False
            remain.update(group)
            #break
        if valid:  # if this group is smaller than or equal to `max_sz`, finalize it.
            if features is not None and len(group) > 20:
                filter_th = th_knn
                origin_group = group
                filtered_list = filter_single_by_features(list(group), features, th=filter_th, valid_percent=0.5)
                group = set(filtered_list)
                invalid_set = origin_group - group
                # if 200682 in origin_group:
                #     print("1====>>", len(origin_group), len(group), len(invalid_set), th_knn)
                # if 200515 in origin_group:
                #     print("2====>>", len(origin_group), len(group), len(invalid_set), th_knn)

                if len(invalid_set) > 0.9 * len(group):
                    # 筛选混杂的
                    remain.update(origin_group)
                elif len(invalid_set) > 0.2 * len(group):
                    # 大于０.2极大概率不是一条
                    # remain.update(origin_group)
                    remain.update(invalid_set)
                    result.append(list(group))
                    # result.append(list(origin_group))
                else:
                    #容忍0.2以内的错误数据
                    # remain.update(invalid_set)
                    # result.append(list(group))
                    result.append(list(origin_group))

                # if len(invalid_set) > 0:
                #     remain.update(origin_group)
                # else:
                #     result.append(list(origin_group))

                # result.append(list(origin_group))
            else:
                # if 200682 in group:
                #     print("1xxxxxxxxxxxx", len(group))
                # if 200515 in group:
                #     print("2xxxxxxxxxxxx", len(group))
                result.append(list(group))
    return result, remain


def graph_clustering_dynamic_th_with_graph(G, max_sz, step, start_th, max_iter=100, labels=None, features=None, th_knn=0.5):
    # first iteration
    comps, remain = connected_components_constraint(G.keys(), max_sz, None, G, labels=labels, features=features, th_knn=th_knn)
    # iteration
    components = comps
    one_count = reduce(lambda x, y: x + (1 if len(y) == 1 else 0), components, 0)
    processed_count = reduce(lambda x, y: x + len(y), components, 0)
    logger.info("connected_components_constraint iter {}, th {} components_label {}  all_count {} processed_count {} remain_count {}  one_count {} ".
                format(0, start_th, len(components), processed_count + len(remain), processed_count, len(remain), one_count))
    Iter = 0
    th = start_th
    while remain:
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, th, G, labels=labels, features=features, th_knn=th_knn)
        components.extend(comps)
        # check_components(components)
        one_count = reduce(lambda x, y: x + (1 if len(y) == 1 else 0), components, 0)
        processed_count = reduce(lambda x, y: x + len(y), components, 0)
        Iter += 1
        logger.info("connected_components_constraint iter {}, th {} components_label {}  all_count {} processed_count {} remain_count {}  one_count {} ".
                    format(Iter, th, len(components), processed_count + len(remain), processed_count, len(remain), one_count))
        if Iter >= max_iter or th > 0.98:
            logger.info("\t Force stopping at: th {}, remain {}".format(th, len(remain)))
            components.append(list(remain))
            remain = {}

    return components


def sim_by_feature(feats, queries, ners):
    sims = np.zeros_like(ners, dtype=np.float32)
    for index, ner in enumerate(ners):
        sub_feats = feats[ner]
        sims[index] = np.dot(queries[index], sub_feats.T)
    return sims


def cluster_by_knns(knns, features, th_knn, max_size, labels, is_filter=True):
    '''
    与face-train不同，这里聚类的相似度没有经过1-转换
    :param features:
    :param th_knn:
    :param max_size:
    :return:
    '''
    k = 80
    # 0,1到-1,1
    th_step = 0.05

    size = len(knns)
    # graph
    G, edges_count, min_score, max_socre = gen_graph(knns, k, th_knn)
    del knns
    if labels is not None:
        label_clusters = defaultdict(list)
        for index, label in enumerate(labels):
            label_clusters[label].append(index)
        G, new_edges_count = gen_graph_by_clusters(label_clusters, G, edges_count)
        logger.info("edges_count %s new_edges_cout %s", edges_count, new_edges_count)
    # cdp聚类
    clusters = graph_clustering_dynamic_th_with_graph(G, max_size, th_step, min_score, labels=labels, features=features, th_knn=th_knn)
    del G
    count = reduce(lambda x, y: x + len(y), clusters, 0)
    one_count = reduce(lambda x, y: x + (1 if len(y) == 1 else 0), clusters, 0)
    logger.info("cdp cluster graph image count {} one_count {} total {}".format(count, one_count, size))

    if is_filter:
        logger.info("cdp before filter_clusters_by_features count %s", len(clusters))
        clusters = filter_clusters_by_features(clusters, features, th_knn, labels)
        logger.info("cdp after filter_clusters_by_features count %s", len(clusters))

    ret_labels = np.ones((len(features))) * -1
    check_set = set()
    for label, c in enumerate(clusters):
        if len(c) > 1:
            for index in c:
                assert index not in check_set
                ret_labels[index] = label
                check_set.add(index)

    # 对-1忽略的赋值不同标签
    # other_label = len(clusters)
    # for i in range(len(ret_labels)):
    #     if ret_labels[i] == -1:
    #         ret_labels[i] = other_label
    #         other_label += 1

    logger.info("cdp  image count %s ignore count %s label count %s", size, (ret_labels == -1).sum(), len(set(ret_labels)) - 1)
    return ret_labels


def cluster(features, th_knn, max_size=300, labels=None):
    '''
    与face-train不同，这里聚类的相似度没有经过1-转换
    :param features:
    :param th_knn:
    :param max_size:
    :return:
    '''
    k = 80
    nprobe = 8

    # knn
    size, dim = features.shape
    metric = faiss.METRIC_INNER_PRODUCT
    nlist = min(4096, 8 * round(math.sqrt(size)))
    if size < 4 * 10000:
        fac_str = "Flat"  # same
    elif size < 80 * 10000:
        fac_str = "IVF" + str(nlist) + ",Flat"  # same
    elif size < 200 * 10000:
        fac_str = "IVF16384,Flat"  # same
    else:
        fac_str = "IVF16384,PQ8"  # same
    logger.info("cdp cluster fac str %s", fac_str)
    index = faiss.index_factory(dim, fac_str, metric)
    index.train(features)
    index.nprobe = min(nprobe, nlist)
    assert index.is_trained
    logger.info('cdp cluster nlist: {}, nprobe: {}'.format(nlist, nprobe))
    index.add(features)

    sims, ners = index.search(features, k=k)
    if "Flat" not in fac_str:
        sims = sim_by_feature(features, features, ners)
    knns = np.concatenate([sims[:, np.newaxis].astype(np.float32), ners[:, np.newaxis].astype(np.float32)], axis=1)
    # del features

    return cluster_by_knns(knns, features, th_knn, max_size, labels)


if __name__ == '__main__':

    import os
    from sklearn.metrics.cluster import fowlkes_mallows_score


    def single_remove_dict(Y, pred):
        single_idcs = []
        for index, l in enumerate(pred):
            if l != -1:
                single_idcs.append(index)
        return Y[single_idcs], pred[single_idcs]


    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    feats = np.load(os.path.expanduser("~/datasets/lfw/lfw_retina_r100_1655.fea.npy"))
    labels = np.load(os.path.expanduser("~/datasets/lfw/lfw_retina_r100_1655.labels.npy"))
    # feats = np.load(os.path.expanduser("~/datasets/faces_umd/train_retinar50_r100.fea.npy"))
    # labels = np.load(os.path.expanduser("~/datasets/faces_umd/train_retinar50_r100.labels.npy"))
    print("feats len ", len(feats))
    pred_labels = cluster(feats, 0.5, 300, np.ones(len(labels)) * -1)
    print("label num {} ignore num {}".format(len(set(pred_labels)), np.sum(pred_labels == -1)))

    print("f1 ", fowlkes_mallows_score(labels, pred_labels))
    labels, pred_labels = single_remove_dict(labels, pred_labels)
    print("f2 ", fowlkes_mallows_score(labels, pred_labels))

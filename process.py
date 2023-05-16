import os
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import pickle
import json
import gzip
import random
import math
from collections import defaultdict




###IO Functions###########
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True


## Processing Helpers ######
def interactions(data_name, rating_cutoff):

    """
    Gets a list containing each interaction from the data that fits our rating cutoff criteria
    """

    datas = [] 
    data_file = 'data/raw_data/reviews_' + data_name + '.json.gz'

    for inter in parse(data_file):
        if float(inter['overall']) <= rating_cutoff: # remove
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas



def meta_features(data_name, data_maps):
    
    """
    Creates a dictionary containing features for every item that occurs in the data
    """
    datas = {}
    meta_file = 'data/raw_data/meta_' + data_name + '.json.gz'
    item_asins = list(data_maps['item2id'].keys())
    for info in parse(meta_file):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas




def seq_timesort_interactions(datas):
    """
    Sort the interactions chronilogically
    returns a dictionary mapping each user to their interactions history
    """
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1]) 
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

def id_map(user_items): # user_items dict
    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


def get_attribute_Amazon(meta_infos, datamaps, attribute_core):

    attributes = defaultdict(int)
    for iid, info in tqdm(meta_infos.items()):
        for cates in info['categories']:
            for cate in cates[1:]:
                attributes[cate] +=1
        try:
            attributes[info['brand']] += 1
        except:
            pass

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info['brand']] >= attribute_core:
                new_meta[iid].append(info['brand'])
        except:
            pass
        for cates in info['categories']:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
    
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

def item_sentence_context(item_infos, datamaps):
    """
    Create a sentence to represent each item from its (title, brand, price, categories), 
    t
    """
    new_feat = {}
    for iid, info in item_infos.items():
        item_id = datamaps["item2id"][iid]
        sentence = ""
        if "title" in info:
            sentence += info["title"] + ". "
        if "brand" in info:
            sentence += "Product is made by {}. ".format(info["brand"])
        if "price" in info:
            sentence += "Price in US Dollars is {}. ".format(info["price"])
        if "categories" in info:
            sentence += "Relevant categories are"
            for cates in info["categories"]:
                for cate in cates:
                    sentence += " " + cate
            
            sentence += "."


        new_feat[item_id] = sentence
    return new_feat


def process(dataset_name, processed_acronym=None):

    """
    Process a single datset {Beauty, Sports and Outdoors, Toys and Games}
    """
    if not processed_acronym:
        processed_acronym = dataset_name


    rating_cutoff = 0 # keeping all reviews
    # remove users/items with less than 5 occurences
    user_core = 5
    item_core = 5
    attribute_core = 0

    datas = interactions(dataset_name+"_5", rating_cutoff=rating_cutoff)
    print(f"After removing ratings less than {rating_cutoff} there are {len(datas)} interactions")
    user_hist = seq_timesort_interactions(datas)
    print(f"Mapping from {len(user_hist.items())} users to their item histories.")




    user_hist, user_num, item_num, data_maps = id_map(user_hist)
    user_count, item_count, _ = check_Kcore(user_hist, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    
    print("Processing meta information...")
    item_info = meta_features(dataset_name, data_maps)
    attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(item_info, data_maps, attribute_core)
    

    # print(datamaps["1"])
    # print(item2attributes["1"])
    
    # for item, info in item_info.items():
    #     del item_info[item]
    #     item_info[datamaps["item2id"][item]] = info
    item_feat = item_sentence_context(item_info, datamaps)
    print("Saving processed data...")

    data_file = 'data/{}/'.format(processed_acronym) + 'sequential_data.txt'
    item2attributes_file = 'data/{}/'.format(processed_acronym) + 'item2attributes.json'
    datamaps_file = 'data/{}/'.format(processed_acronym) + 'datamaps.json'
    item_feat_file = 'data/{}/'.format(processed_acronym) + 'item_feat.json'


    with open(data_file, 'w+') as out:
        for user, items in user_hist.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w+') as out:
        out.write(json_str)
        
    json_str = json.dumps(datamaps)
    with open(datamaps_file, 'w+') as out:
        out.write(json_str)
    
    json_str = json.dumps(item_feat)
    with open(item_feat_file, 'w+') as out:
        out.write(json_str)



if __name__ == "__main__":
    # formal name to new name
    datasets = {"Beauty": "beauty", "Toys_and_Games": "toys", "Sports_and_Outdoors": "sports"}

    for key, val in datasets.items():
        process(key, val) 
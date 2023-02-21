import json
import pickle
import os
print('pid: ', os.getpid())

cls_index = {
    "train": [30, 25, 36, 4, 14, 22, 28, 7, 20, 8, 9, 3, 15, 34, 24, 29, 1,
              6, 0, 16, 37, 5, 33, 35, 27],
    "val": [17, 13, 18, 2, 40, 39],
    "test": [32, 11, 23, 19, 10, 26, 12, 31, 21, 38]
}
cls_name = {
    "train": ["arts", "healthy living", "weddings", "politics", "tech", "style", "worldpost", "women", "parents",
              "comedy", "queer voices", "impact", "religion", "style & beauty", "taste", "fifty", "entertainment",
              "black voices", "crime", "science", "food & drink", "weird news", "home & living", "divorce",
              "good news"],
    "val": ["latino voices", "media", "education", "world news", "culture & arts", "environment"],
    "test": ["parenting", "business", "green", "college", "sports", "the worldpost", "travel", "wellness",
             "arts & culture", "money"]}
cls_to_idx = {"crime": 0, "entertainment": 1, "world news": 2, "impact": 3, "politics": 4, "weird news": 5,
              "black voices": 6, "women": 7, "comedy": 8, "queer voices": 9, "sports": 10, "business": 11, "travel": 12,
              "media": 13, "tech": 14, "religion": 15, "science": 16, "latino voices": 17, "education": 18,
              "college": 19, "parents": 20, "arts & culture": 21, "style": 22, "green": 23, "taste": 24,
              "healthy living": 25, "the worldpost": 26, "good news": 27, "worldpost": 28, "fifty": 29, "arts": 30,
              "wellness": 31, "parenting": 32, "home & living": 33, "style & beauty": 34, "divorce": 35, "weddings": 36,
              "food & drink": 37, "money": 38, "environment": 39, "culture & arts": 40}
print("loading pickle huffpost...")
all = pickle.load(open("/data/lrs/test/fid/data_fewshot/huffpost_retrieved_data_all_fast.pkl", 'rb'))
print("have loaded pickle huffpost...")

dict_data = {}
pkl_dict_data = open("/data/lrs/test/fid/data_fewshot/dict_data_sts_retrieved_100.pkl", "wb")
f_dict_data = open("/data/lrs/test/fid/data_fewshot/dict_data_sts_retrieved_100.json", 'w')


for key, value in cls_index.items():
    for i in value:
        dict_data[str(i)] = []

print("create dict_data")

for line in all:
    cat = line['category'].lower()
    cat_idx = cls_to_idx[cat]
    cat_idx = str(cat_idx)
    dict_data[cat_idx].append(line)

print("start dump pkl")

# s = []
# for key in dict_data:
#     s.append(len(dict_data[key]))
#
# s

pickle.dump(dict_data, pkl_dict_data)

print("close file")

pkl_dict_data.close()

# json.dump(dict_data, f_dict_data)
#
# print("close file")
#
# f_dict_data.close()
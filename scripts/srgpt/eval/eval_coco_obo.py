import json
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco = COCO("/PATH_TO_COCO_ANNOTATIONS/coco/annotations/instances_val2017.json")

cid2name = {}
for _, v in coco.cats.items():
    cid = v["id"]
    cname = v["name"]
    cid2name[cid] = cname

for k in list(cid2name.keys()):
    v = cid2name[k]
    v = v.replace("_", " ")
    cid2name[k] = v.lower()

name2cid = {v: k for k, v in cid2name.items()}

correct, total = 0, 0
data_list = []
with open(sys.argv[1]) as f:
    for line in f:
        data = json.loads(line)
        if data["text"] == data["gt_name"]:
            correct += 1
        total += 1
        data_list.append(data)

print("Acc: ", correct / total * 100)

result_list = []
except_list = []
for data in data_list:
    try:
        text = data["text"].split(",")[0].lower()
        cid = name2cid[text]
        bbox = data["bbox"][0]
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        image_id = data["image_id"]
        score = 1.0

        result = {"bbox": bbox, "image_id": image_id, "category_id": cid, "score": score}
        result_list.append(result)
    except:
        except_list.append(text)


with open("./result.json", "w") as f:
    json.dump(result_list, f)

cocoGt = coco
cocoDt = cocoGt.loadRes("./result.json")
leval = COCOeval(cocoGt, cocoDt, iouType="bbox")

leval.evaluate()
leval.accumulate()
leval.summarize()

print(len(except_list), len(result_list))

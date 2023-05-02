import json

"""
< COCO Dataset Format >
coco.json
  - info
    - description
    - url
    - version
    - year
    - contributor
    - date_created
  - licenses
  - images
  - annotations
  - categories
"""

if __name__ == "__main__":
    # with open("/home/bulgogi/bolero/dataset/coco_minitrain/instances_minitrain2017.json", 'r') as f:
    with open("/home/bulgogi/bolero/dataset/dsc_dataset/roboflow/Guide_Real_Real_2.v3i.coco-segmentation/train/_annotations.coco.json") as f:
        data = json.load(f)

    # print(data)
    categories = []
    print(data.keys())
    print(data['info'].keys())
    print(data['images'][0].keys())
    print(data['annotations'][0].keys())
    for i in range(len(data['categories'])):
        category_name = data['categories'][i]['name']
        categories.append(category_name)
        # print(data['categories'][i]['name'])

    print(categories)
    for i, c in enumerate(categories):
        print(f"  {i}: {c}")


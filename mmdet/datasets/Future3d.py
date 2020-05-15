<<<<<<< HEAD
#coding=utf-8
=======
# coding=utf-8
# ck修改
>>>>>>> shuang
import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset


<<<<<<< HEAD

@DATASETS.register_module
class Future3dDataset(CustomDataset):

=======
@DATASETS.register_module
class Future3dDataset(CustomDataset):
>>>>>>> shuang
    '''
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    '''
    # 我把fine-grained category name搬了过来
    CLASSES = (
        'Children Cabinet', 'Nightstand', 'Bookcase / jewelry Armoire', 'Wardrobe',
<<<<<<< HEAD
        'Tea Table', 'Corner/Side Table', 'Sideboard / Side Cabinet / Console', 
        'Wine Cooler', 'TV Stand', 'Drawer Chest / Corner cabinet', 'Shelf', 
        'Round End Table', 'Double Bed', 'Bunk Bed', 'Bed Frame', 'Single bed', 
        'Kids Bed', 'Dining Chair', 'Lounge Chair / Book-chair / Computer Chair', 
        'Dressing Chair', 'Classic Chinese Chair', 'Barstool', 'Dressing Table', 
        'Dining Table', 'Desk', 'Three-Seat / Multi-person sofa', 'armchair',
        'Two-seat Sofa', 'L-shaped Sofa', 'Lazy Sofa', 'Chaise Longue Sofa', 
=======
        'Tea Table', 'Corner/Side Table', 'Sideboard / Side Cabinet / Console',
        'Wine Cooler', 'TV Stand', 'Drawer Chest / Corner cabinet', 'Shelf',
        'Round End Table', 'Double Bed', 'Bunk Bed', 'Bed Frame', 'Single bed',
        'Kids Bed', 'Dining Chair', 'Lounge Chair / Book-chair / Computer Chair',
        'Dressing Chair', 'Classic Chinese Chair', 'Barstool', 'Dressing Table',
        'Dining Table', 'Desk', 'Three-Seat / Multi-person sofa', 'armchair',
        'Two-seat Sofa', 'L-shaped Sofa', 'Lazy Sofa', 'Chaise Longue Sofa',
>>>>>>> shuang
        'Footstool / Sofastool / Bed End Stool / Stool', 'Pendant Lamp', 'Ceiling Lamp'
    )

    # devkit/dataset_category.py，起到str和int的映射关系，网络输出int，注释是str，我觉得这几个list都可以改写成{'Modern':0, 'Chinoiserie':1, ...}的形式，更方便查找！
    _ATTR_STYLE = [
<<<<<<< HEAD
        {'id': 0, 'category': 'Modern',},
        {'id': 1, 'category': 'Chinoiserie',},
        {'id': 2, 'category': 'Kids',},
        {'id': 3, 'category': 'European',},
        {'id': 4, 'category': 'Japanese',},
        {'id': 5, 'category': 'Southeast Asia',},
        {'id': 6, 'category': 'Industrial',},
        {'id': 7, 'category': 'American Country',},
        {'id': 8, 'category': 'Vintage/Retro',},
        {'id': 9, 'category': 'Light Luxury',},
        {'id': 10, 'category': 'Mediterranean',},
        {'id': 11, 'category': 'Korean',},
        {'id': 12, 'category': 'New Chinese',},
        {'id': 13, 'category': 'Nordic',},
        {'id': 14, 'category': 'European Classic',},
        {'id': 15, 'category': 'Others',},
        {'id': 16, 'category': 'Ming Qing',},
        {'id': 17, 'category': 'Neoclassical',},
        {'id': 18, 'category': 'Minimalist',},
    ]

    _ATTR_MATERIAL = [
        {'id': 0, 'category': 'Composition',},
        {'id': 1, 'category': 'Cloth',},
        {'id': 2, 'category': 'Leather',},
        {'id': 3, 'category': 'Glass',},
        {'id': 4, 'category': 'Metal',},
        {'id': 5, 'category': 'Solid Wood',},
        {'id': 6, 'category': 'Stone',},
        {'id': 7, 'category': 'Plywood',},
        {'id': 8, 'category': 'Others',},
        {'id': 9, 'category': 'Suede',},
        {'id': 10, 'category': 'Bamboo Rattan',},
        {'id': 11, 'category': 'Rough Cloth',},
        {'id': 12, 'category': 'Wood',},
        {'id': 13, 'category': 'Composite Board',},
        {'id': 14, 'category': 'Marble',},
        {'id': 15, 'category': 'Smooth Leather',},
    ]

    _ATTR_THEME = [
        {'id': 0, 'category': 'Smooth Net',},
        {'id': 1, 'category': 'Lines',},
        {'id': 2, 'category': 'Wrought Iron',},
        {'id': 3, 'category': 'Cartoon',},
        {'id': 4, 'category': 'Granite Texture',},
        {'id': 5, 'category': 'Floral',},
        {'id': 6, 'category': 'Inlay Gold Carve',},
        {'id': 7, 'category': 'Texture Mark',},
        {'id': 8, 'category': 'Striped Grid',},
        {'id': 9, 'category': 'Chinese Pattern',},
        {'id': 10, 'category': 'Gold Foil',},
        {'id': 11, 'category': 'Rivet',},
        {'id': 12, 'category': 'Soft Case',},
        {'id': 13, 'category': 'Wooden Vertical Texture',},
        {'id': 14, 'category': 'Graffiti Ink Stain',},
        {'id': 15, 'category': 'Linen Texture',},
=======
        {'id': 0, 'category': 'Modern', },
        {'id': 1, 'category': 'Chinoiserie', },
        {'id': 2, 'category': 'Kids', },
        {'id': 3, 'category': 'European', },
        {'id': 4, 'category': 'Japanese', },
        {'id': 5, 'category': 'Southeast Asia', },
        {'id': 6, 'category': 'Industrial', },
        {'id': 7, 'category': 'American Country', },
        {'id': 8, 'category': 'Vintage/Retro', },
        {'id': 9, 'category': 'Light Luxury', },
        {'id': 10, 'category': 'Mediterranean', },
        {'id': 11, 'category': 'Korean', },
        {'id': 12, 'category': 'New Chinese', },
        {'id': 13, 'category': 'Nordic', },
        {'id': 14, 'category': 'European Classic', },
        {'id': 15, 'category': 'Others', },
        {'id': 16, 'category': 'Ming Qing', },
        {'id': 17, 'category': 'Neoclassical', },
        {'id': 18, 'category': 'Minimalist', },
    ]

    _ATTR_MATERIAL = [
        {'id': 0, 'category': 'Composition', },
        {'id': 1, 'category': 'Cloth', },
        {'id': 2, 'category': 'Leather', },
        {'id': 3, 'category': 'Glass', },
        {'id': 4, 'category': 'Metal', },
        {'id': 5, 'category': 'Solid Wood', },
        {'id': 6, 'category': 'Stone', },
        {'id': 7, 'category': 'Plywood', },
        {'id': 8, 'category': 'Others', },
        {'id': 9, 'category': 'Suede', },
        {'id': 10, 'category': 'Bamboo Rattan', },
        {'id': 11, 'category': 'Rough Cloth', },
        {'id': 12, 'category': 'Wood', },
        {'id': 13, 'category': 'Composite Board', },
        {'id': 14, 'category': 'Marble', },
        {'id': 15, 'category': 'Smooth Leather', },
    ]

    _ATTR_THEME = [
        {'id': 0, 'category': 'Smooth Net', },
        {'id': 1, 'category': 'Lines', },
        {'id': 2, 'category': 'Wrought Iron', },
        {'id': 3, 'category': 'Cartoon', },
        {'id': 4, 'category': 'Granite Texture', },
        {'id': 5, 'category': 'Floral', },
        {'id': 6, 'category': 'Inlay Gold Carve', },
        {'id': 7, 'category': 'Texture Mark', },
        {'id': 8, 'category': 'Striped Grid', },
        {'id': 9, 'category': 'Chinese Pattern', },
        {'id': 10, 'category': 'Gold Foil', },
        {'id': 11, 'category': 'Rivet', },
        {'id': 12, 'category': 'Soft Case', },
        {'id': 13, 'category': 'Wooden Vertical Texture', },
        {'id': 14, 'category': 'Graffiti Ink Stain', },
        {'id': 15, 'category': 'Linen Texture', },
>>>>>>> shuang
    ]

    _SUPER_CATEGORIES_3D = [
        {'id': 1, 'category': 'Cabinet/Shelf/Desk'},
        {'id': 2, 'category': 'Bed'},
        {'id': 3, 'category': 'Chair'},
        {'id': 4, 'category': 'Table'},
        {'id': 5, 'category': 'Sofa'},
        {'id': 6, 'category': 'Pier/Stool'},
        {'id': 7, 'category': 'Lighting'},
    ]

    _CATEGORIES_3D = [
        {'id': 1, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Children Cabinet'},
        {'id': 2, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Nightstand'},
        {'id': 3, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Bookcase / jewelry Armoire'},
        {'id': 4, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Wardrobe'},
        {'id': 5, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Tea Table'},
        {'id': 6, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Corner/Side Table'},
        {'id': 7, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Sideboard / Side Cabinet / Console'},
        {'id': 8, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Wine Cooler'},
        {'id': 9, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'TV Stand'},
        {'id': 10, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Drawer Chest / Corner cabinet'},
        {'id': 11, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Shelf'},
        {'id': 12, 'category': 'Cabinet/Shelf/Desk', 'fine-grained category name': 'Round End Table'},
        {'id': 13, 'category': 'Bed', 'fine-grained category name': 'Double Bed'},
        {'id': 14, 'category': 'Bed', 'fine-grained category name': 'Bunk Bed'},
        {'id': 15, 'category': 'Bed', 'fine-grained category name': 'Bed Frame'},
        {'id': 16, 'category': 'Bed', 'fine-grained category name': 'Single bed'},
        {'id': 17, 'category': 'Bed', 'fine-grained category name': 'Kids Bed'},
        {'id': 18, 'category': 'Chair', 'fine-grained category name': 'Dining Chair'},
        {'id': 19, 'category': 'Chair', 'fine-grained category name': 'Lounge Chair / Book-chair / Computer Chair'},
        {'id': 20, 'category': 'Chair', 'fine-grained category name': 'Dressing Chair'},
        {'id': 21, 'category': 'Chair', 'fine-grained category name': 'Classic Chinese Chair'},
        {'id': 22, 'category': 'Chair', 'fine-grained category name': 'Barstool'},
        {'id': 23, 'category': 'Table', 'fine-grained category name': 'Dressing Table'},
        {'id': 24, 'category': 'Table', 'fine-grained category name': 'Dining Table'},
        {'id': 25, 'category': 'Table', 'fine-grained category name': 'Desk'},
        {'id': 26, 'category': 'Sofa', 'fine-grained category name': 'Three-Seat / Multi-person sofa'},
        {'id': 27, 'category': 'Sofa', 'fine-grained category name': 'armchair'},
        {'id': 28, 'category': 'Sofa', 'fine-grained category name': 'Two-seat Sofa'},
        {'id': 29, 'category': 'Sofa', 'fine-grained category name': 'L-shaped Sofa'},
        {'id': 30, 'category': 'Sofa', 'fine-grained category name': 'Lazy Sofa'},
        {'id': 31, 'category': 'Sofa', 'fine-grained category name': 'Chaise Longue Sofa'},
<<<<<<< HEAD
        {'id': 32, 'category': 'Pier/Stool', 'fine-grained category name': 'Footstool / Sofastool / Bed End Stool / Stool'},
=======
        {'id': 32, 'category': 'Pier/Stool',
         'fine-grained category name': 'Footstool / Sofastool / Bed End Stool / Stool'},
>>>>>>> shuang
        {'id': 33, 'category': 'Lighting', 'fine-grained category name': 'Pendant Lamp'},
        {'id': 34, 'category': 'Lighting', 'fine-grained category name': 'Ceiling Lamp'}
    ]

<<<<<<< HEAD

=======
>>>>>>> shuang
    # original implementation of coco.py
    # self.cat_ids在test阶段会用到，我的理解是，self.cat_ids就是_CATEGORIES_3D里的'id'，对应1~34
    '''
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos
<<<<<<< HEAD

=======
>>>>>>> shuang
    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)
    '''

    # 来自Future3dDataset.py
    def init_img2ann_dict(self, json_data):
        im2ann_dict = {}
        for ann_item in json_data['annotations']:
            im_id = ann_item['image_id']
            if im_id not in im2ann_dict.keys():
<<<<<<< HEAD
                im2ann_dict[im_id] = [ann_item,]
=======
                im2ann_dict[im_id] = [ann_item, ]
>>>>>>> shuang
            else:
                im2ann_dict[im_id].append(ann_item)

        return im2ann_dict
<<<<<<< HEAD
    
    def load_annotations(self, ann_file):
        self._json_data = mmcv.load(ann_file) #导入train_set.json或val_set.json
        if 'train' in ann_file:
            self.istrain = True #flag
            self._im2ann_dict = self.init_img2ann_dict(self._json_data) #annotations下的image_id与annotations下的dict的映射
            categories_data = self._json_data['categories'] #categories这个list
            self._cate_dict = {_['id']: _ for _ in categories_data} #categories下的id与id所属的dict的键值对
=======

    def load_annotations(self, ann_file):
        self._json_data = mmcv.load(ann_file)  # 导入train_set.json或val_set.json
        if 'train' in ann_file:
            self.istrain = True  # flag
            self._im2ann_dict = self.init_img2ann_dict(self._json_data)  # annotations下的image_id与annotations下的dict的映射
            categories_data = self._json_data['categories']  # categories这个list
            self._cate_dict = {_['id']: _ for _ in categories_data}  # categories下的id与id所属的dict的键值对
>>>>>>> shuang
        else:
            self.istrain = False
            self._im2ann_dict = None
            self._cate_dict = None
<<<<<<< HEAD
        return self._json_data['images'] #v2.0的self.img_infos改名为self.data_infos，self.data_infos是images这个list
=======
        return self._json_data['images']  # v2.0的self.img_infos改名为self.data_infos，self.data_infos是images这个list
>>>>>>> shuang

    def get_ann_info(self, idx):
        if not self.istrain:
            return None
<<<<<<< HEAD
        im_id = self.data_infos[idx]['id'] #images下的某个id，对应annotations里的image_id！
        ann_list = self._im2ann_dict[im_id] #该id对应上述映射的某一项，是一个list，其中有1或多个dict，ann_list相当于原函数的ann_info了
=======
        im_id = self.data_infos[idx]['id']  # images下的某个id，对应annotations里的image_id！
        ann_list = self._im2ann_dict[im_id]  # 该id对应上述映射的某一项，是一个list，其中有1或多个dict，ann_list相当于原函数的ann_info了
>>>>>>> shuang
        cate_ids = []
        cate_names = []
        fine_grained_cate_names = []
        file_names = []
        segms = []
        areas = []
        bboxes = []
        model_ids = []
        texture_ids = []
        poses = []
        fovs = []
        styles = []
        themes = []
        materials = []
        for i, ann_item in enumerate(ann_list):
            cate_id = ann_item['category_id']
<<<<<<< HEAD
            cate_ids.append(cate_id) #原函数有个cat2label操作，而且v2.0的cat2label的映射变了，这里我使用了devkit提供的_im2ann_dict函数
            cate_names.append(self._cate_dict[cate_id]['category_name']) #默认是str，有需要的话可以根据上面新加的几个list转成int
            fine_grained_cate_names.append(self._cate_dict[cate_id]['fine-grained category name']) #默认是str，有需要的话可以根据上面新加的几个list转成int
            file_names.append(self.data_infos[idx]['file_name'] + '.jpg') #训练集中的image文件夹，file_name这项注释默认是不带后缀的
            segms.append(ann_item['segmentation'])
            areas.append(ann_item['area'])
            x1, y1, w, h = ann_item['bbox']
            bboxes.append([x1, y1, x1+w, y1+h]) #模仿了coco.py里_parse_ann_info函数的做法，注意！！新版不是'x1+w-1'了！
            # 从model_id开始，后面的项都有可能是null
            model_ids.append(ann_item['model_id']) #如"004062"，要用的话后缀加.obj
            texture_ids.append(ann_item['texture_id']) #如"004062"，要用的话后缀加.png
            poses.append(ann_item['pose']) #pose不是像比赛官网说的是一个list，而是一个dict，里面有translation(长度为3)和rotation(长度为3*3)两个list！
            fovs.append(ann_item['fov']) #fov应该是视场角的意思，虽然我不知道怎么用进去:(
            styles.append(ann_item['style']) #默认是str，有需要的话可以根据上面新加的几个list转成int
            themes.append(ann_item['theme']) #默认是str，有需要的话可以根据上面新加的几个list转成int
            materials.append(ann_item['material']) #默认是str，有需要的话可以根据上面新加的几个list转成int
=======
            cate_ids.append(cate_id)  # 原函数有个cat2label操作，而且v2.0的cat2label的映射变了，这里我使用了devkit提供的_im2ann_dict函数
            cate_names.append(self._cate_dict[cate_id]['category_name'])  # 默认是str，有需要的话可以根据上面新加的几个list转成int
            fine_grained_cate_names.append(
                self._cate_dict[cate_id]['fine-grained category name'])  # 默认是str，有需要的话可以根据上面新加的几个list转成int
            file_names.append(self.data_infos[idx]['file_name'] + '.jpg')  # 训练集中的image文件夹，file_name这项注释默认是不带后缀的
            segms.append(ann_item['segmentation'])
            areas.append(ann_item['area'])
            x1, y1, w, h = ann_item['bbox']
            bboxes.append([x1, y1, x1 + w, y1 + h])  # 模仿了coco.py里_parse_ann_info函数的做法，注意！！新版不是'x1+w-1'了！
            # 从model_id开始，后面的项都有可能是null
            model_ids.append(ann_item['model_id'])  # 如"004062"，要用的话后缀加.obj
            texture_ids.append(ann_item['texture_id'])  # 如"004062"，要用的话后缀加.png
            poses.append(ann_item['pose'])  # pose不是像比赛官网说的是一个list，而是一个dict，里面有translation(长度为3)和rotation(长度为3*3)两个list！
            fovs.append(ann_item['fov'])  # fov应该是视场角的意思，虽然我不知道怎么用进去:(
            styles.append(ann_item['style'])  # 默认是str，有需要的话可以根据上面新加的几个list转成int
            themes.append(ann_item['theme'])  # 默认是str，有需要的话可以根据上面新加的几个list转成int
            materials.append(ann_item['material'])  # 默认是str，有需要的话可以根据上面新加的几个list转成int
>>>>>>> shuang

            # 模仿_parse_ann_info，把bbox和label转化为np.array
            bboxes = np.array(bboxes, dtype=np.float32)
            cate_ids = np.array(cate_ids, dtype=np.int64)

<<<<<<< HEAD
            seg_map = self.data_infos[idx]['file_name'] + '.png' #训练集中的idmap文件夹
=======
            seg_map = self.data_infos[idx]['file_name'] + '.png'  # 训练集中的idmap文件夹
>>>>>>> shuang

            # 这里上面的某些list我没放进去，大家看看需不需要增删
            ann = dict(
                labels=cate_ids,
                masks=segms,
                areas=areas,
                bboxes=bboxes,
                model_ids=model_ids,
                texture_ids=texture_ids,
                poses=poses,
                fovs=fovs,
                styles=styles,
                themes=themes,
                materials=materials,
                seg_map=seg_map
            )

            return ann

<<<<<<< HEAD

=======
>>>>>>> shuang
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self._json_data['annotations'])
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and img_info['id'] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    # 如果不指定classes，这个函数似乎用不着？我暂时没改
    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.
<<<<<<< HEAD

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

=======
        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.
        Args:
            class_ids (list[int]): list of category ids
>>>>>>> shuang
        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.catToImgs[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    # 这个函数我整合到get_ann_info函数里了，用不着了
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
<<<<<<< HEAD

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

=======
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
>>>>>>> shuang
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    # 以下应该是test的任务？请注意self.cat_ids！

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0] + 1,
            _bbox[3] - _bbox[1] + 1,
        ]

    def _proposal2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a json file.
<<<<<<< HEAD

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

=======
        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.
>>>>>>> shuang
        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".
<<<<<<< HEAD

=======
>>>>>>> shuang
        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            result_files['segm'] = '{}.{}.json'.format(outfile_prefix, 'segm')
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'proposal')
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).
<<<<<<< HEAD

=======
>>>>>>> shuang
        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
<<<<<<< HEAD

=======
>>>>>>> shuang
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
<<<<<<< HEAD
            format(len(results), len(self)))
=======
                format(len(results), len(self)))
>>>>>>> shuang

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.
<<<<<<< HEAD

=======
>>>>>>> shuang
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.
<<<<<<< HEAD

=======
>>>>>>> shuang
        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = self.img_ids
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float('{:.3f}'.format(cocoEval.stats[i + 6]))
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    pass  # TODO
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = '{}_{}'.format(metric, metric_items[i])
                    val = float('{:.3f}'.format(cocoEval.stats[i]))
                    eval_results[key] = val
                eval_results['{}_mAP_copypaste'.format(metric)] = (
                    '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

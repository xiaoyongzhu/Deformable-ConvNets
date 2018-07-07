# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

import _init_paths
import math
import argparse
import os
import sys
import logging
import pprint
import cv2
import tqdm as tqdm
from config.config import config, update_config
from utils.image import resize, transform
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fpn/cfgs/resnet_v1_101_xview_trainval_fpn_dcn_end2end_ohem.yaml')
# update_config(cur_path + '/../experiments/fpn/cfgs/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


# set up class names
num_classes = 61
classes = ['Fixed-wing Aircraft','Small Aircraft','Cargo Plane','Helicopter','Passenger Vehicle','Small Car','Bus','Pickup Truck','Utility Truck','Truck','Cargo Truck','Truck w/Box','Truck Tractor','Trailer','Truck w/Flatbed','Truck w/Liquid','Crane Truck','Railway Vehicle','Passenger Car','Cargo Car','Flat Car','Tank car','Locomotive','Maritime Vessel','Motorboat','Sailboat','Tugboat','Barge','Fishing Vessel','Ferry','Yacht','Container Ship','Oil Tanker','Engineering Vehicle','Tower crane','Container Crane','Reach Stacker','Straddle Carrier','Mobile Crane','Dump Truck','Haul Truck','Scraper/Tractor','Front loader/Bulldozer','Excavator','Cement Mixer','Ground Grader','Hut/Tent','Shed','Building','Aircraft Hangar','Damaged Building','Facility','Construction Site','Vehicle Lot','Helipad','Storage Tank','Shipping container lot','Shipping Container','Pylon','Tower']


def generate_detections(data, data_names, predictor, config, nms, image_list, detection_num):
    global classes
    ret_boxes = []
    ret_scores = []
    ret_classes = []
    k = 0
    conversion_dict = {0:"11",1:"12",2:"13",3:"15",4:"17",5:"18",6:"19",7:"20",8:"21",9:"23",10:"24",11:"25",12:"26",13:"27",14:"28",15:"29",16:"32",17:"33",18:"34",19:"35",20:"36",21:"37",22:"38",23:"40",24:"41",25:"42",26:"44",27:"45",28:"47",29:"49",30:"50",31:"51",32:"52",33:"53",34:"54",35:"55",36:"56",37:"57",38:"59",39:"60",40:"61",41:"62",42:"63",43:"64",44:"65",45:"66",46:"71",47:"72",48:"73",49:"74",50:"76",51:"77",52:"79",53:"83",54:"84",55:"86",56:"89",57:"91",58:"93",59:"94"}

    for idx, im in enumerate(image_list):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                        provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                        provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > 0.01, :]
            
            dets_nms.append(cls_dets)

        # dets_nms format:
        # a length 60 array, with each element being a dict representing each class
        # the coordinates are in (xmin, ymin, xmax, ymax, confidence) format. The coordinates are not normalized
        # one sample is: [290.09448    439.60617    333.31235    461.8115       0.94750994]
        # below iterates class by class
        image_detection_num = 0
        for index_class in range(len(dets_nms)):
            # for each class
            single_class_nms = dets_nms[index_class]
            image_detection_num += len(single_class_nms)
            if len(single_class_nms) != 0:
                print("detecting", single_class_nms.size, "number of objects", len(single_class_nms))
                # print(single_class_nms)
                for index_single_class_nms in range(min(len(single_class_nms),detection_num)):
                    # print("index_class,index_single_class_nms", index_class,index_single_class_nms, )
                    ret_boxes.append(dets_nms[index_class][index_single_class_nms][:-1]) #get all the element other than the last one
                    ret_scores.append(dets_nms[index_class][index_single_class_nms][-1]) #last element 
                    ret_classes.append(conversion_dict[index_class])
        # pad zeros
        # print("1st: len(ret_boxes), image_detection_num", len(ret_boxes), image_detection_num)
        if image_detection_num <= detection_num:
            for index_element in range(int(detection_num - image_detection_num)):
                ret_boxes.append(np.zeros((4,),dtype=np.float32)) #get all the element other than the last one
                ret_scores.append(0) #last element 
                ret_classes.append(0)
        else:
            print("~~~~~ too many predictions ~~~~~~~~~~~~~~~")
        print("len(ret_boxes), image_detection_num", len(ret_boxes), image_detection_num)
                
        print('testing image {} {:.4f}s, detection number {}'.format(idx +1 , toc(),image_detection_num))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # visualize
        # don't show the final images.
        # show_boxes(im, dets_nms, classes, 1, show_image = False, img_save_name = str(idx) + ".png")
    ret_boxes =   np.squeeze(np.array(ret_boxes))
    ret_scores = np.squeeze(np.array(ret_scores))
    ret_classes = np.squeeze(np.array(ret_classes))

    return ret_boxes, ret_scores, ret_classes


def draw_bboxes(img,boxes,classes):
    """
    Draw bounding boxes on top of an image
    Args:
        img : Array of image to be modified
        boxes: An (N,4) array of boxes to draw, where N is the number of boxes.
        classes: An (N,1) array of classes corresponding to each bounding box.
    Outputs:
        An array of the same shape as 'img' with bounding boxes
            and classes drawn
    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])

    idx = 0

    for i in range(len(boxes)):
        xmin,ymin,xmax,ymax = boxes[i]
        c = classes[i]

        draw.text((xmin+15,ymin+15), str(c))

        for j in range(4):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    return source

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    # parser.add_argument('--fpn_only', help='whether use R-FCN only (w/o Deformable ConvNets)', default=False, action='store_true')
    parser.add_argument("--input", help="Path to test chip")
    parser.add_argument("-o","--output",default="predictions.txt",help="Filepath of desired output")
    parser.add_argument("--cpu_only",default=True,help="whether CPU only or GPU")
    parser.add_argument("--chip_size",default=480,type=int,help="chip size for the input images; we will chip based on this resolution with (chip_size, chip_size)")

    args = parser.parse_args()
    return args

args = parse_args()

def chip_image(img, chip_size=(300,300)):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    width, height, _ = img.shape
    wn, hn = chip_size
    print("image size is", width, height, "actual ratio is", width/wn, height/hn, "and round up to",int(width/wn), int(height/hn))
    images = np.zeros((int(width / wn) * int(height / hn), wn, hn, 3))
    k = 0
    for i in tqdm.tqdm(range(int(width / wn))):
        for j in range(int(height / hn)):
            
            chip = img[wn * i:wn * (i + 1), hn * j:hn * (j + 1), :3]
            images[k] = chip
            
            k = k + 1
    
    return images.astype(np.uint8)

def main():
    global classes
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_fpn_dcn_rcnn' 
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # load demo data
    data = []
    portion = args.chip_size
    assert os.path.exists(args.input), ('%s does not exist'.format(args.input))
    im = cv2.imread(args.input, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    arr = np.array(im)
    origin_width,origin_height,_ = arr.shape
    cwn,chn = (portion, portion)
    wn,hn = (int(origin_width / cwn), int(origin_height / chn))
    padding_y = int(math.ceil(float(origin_height)/chn) * chn - origin_height)
    padding_x = int(math.ceil(float(origin_width)/cwn) * cwn - origin_width)
    print("padding_y,padding_x, origin_height, origin_width",padding_y,padding_x, origin_height, origin_width)
    # top, bottom, left, right - border width in number of pixels in corresponding directions
    im = cv2.copyMakeBorder(im,0,padding_x,0,padding_y,cv2.BORDER_CONSTANT,value=[0,0,0])
    # the section below could be optimized. but basically the idea is to re-calculate all the values
    arr = np.array(im)
    width,height,_ = arr.shape
    cwn,chn = (portion, portion)
    wn,hn = (int(width / cwn), int(height / chn))
    
    image_list = chip_image(im,(portion,portion))
    for im in image_list:
        target_size = args.input_size
        max_size =  args.input_size
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})


    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + '/../model/' + ('fpn_dcn_xview_480_640_800_alltrain'), 11, process=True)

    # arg_params, aux_params = load_param(cur_path + '/../model/' + ('fpn_dcn_coco' if not args.fpn_only else 'fpn_coco'), 0, process=True)
    print("loading parameter done")
   
    if args.cpu_only:
        predictor = Predictor(sym, data_names, label_names,
                          context=[mx.cpu()], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
        nms = py_nms_wrapper(config.TEST.NMS)
    else:
        predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
        nms = gpu_nms_wrapper(config.TEST.NMS,0)        

    num_preds = int(4000 * math.ceil(float(portion)/400))
    # test
    boxes, scores, classes = generate_detections(data, data_names, predictor, config, nms, image_list, num_preds)
    #Process boxes to be full-sized
    
    print("boxes shape is", boxes.shape, "wn, hn", wn, hn, "width, height", width, height)
    bfull = boxes.reshape((wn, hn, num_preds, 4))

    for i in range(wn):
        for j in range(hn):
            bfull[i, j, :, 0] += j*cwn
            bfull[i, j, :, 2] += j*cwn
            
            bfull[i, j, :, 1] += i*chn
            bfull[i, j, :, 3] += i*chn
            
            # clip values
            bfull[i, j, :, 0] = np.clip(bfull[i, j, :, 0], 0,origin_height)
            bfull[i, j, :, 2] = np.clip(bfull[i, j, :, 2], 0,origin_height)
            
            bfull[i, j, :, 1] = np.clip(bfull[i, j, :, 1], 0,origin_width)
            bfull[i, j, :, 3] = np.clip(bfull[i, j, :, 3], 0,origin_width)

    bfull = bfull.reshape((hn * wn, num_preds, 4))
    scores = scores.reshape((hn * wn, num_preds))
    classes = classes.reshape((hn * wn, num_preds))


    #only display boxes with confidence > .5
    print("bfull.shape,scores.shape",bfull.shape,scores.shape)
    # print(bfull, scores, classes)
    bs = bfull[scores > .5]
    cs = classes[scores>.5]
    # s = im_name
    # draw_bboxes(arr,bs,cs).save("/tmp/"+s[0].split(".")[0] + ".png")


    with open(args.output,'w') as f:
        for i in range(bfull.shape[0]):
            for j in range(bfull[i].shape[0]):
                #box should be xmin ymin xmax ymax
                box = bfull[i, j]
                class_prediction = classes[i, j]
                score_prediction = scores[i, j]
                if int(class_prediction) != 0:
                    f.write('%d %d %d %d %d %f \n' % \
                        (box[0], box[1], box[2], box[3], int(class_prediction), score_prediction))

    print('done')

if __name__ == '__main__':
    main()


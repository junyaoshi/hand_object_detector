import os.path as osp
import os
import pdb
import time
import json

import numpy as np
import torch
import cv2

import _init_paths
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from demo_ss import _get_image_blob

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])


def set_up_bbox_extractor(
        cfg_file='cfgs/res101.yml',
        set_cfgs=None,
        cuda=True,
        load_dir="models",
        net='res101',
        dataset='pascal_voc',
        checksession=1,
        checkepoch=8,
        checkpoint=132028,
        class_agnostic=False,
):
    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    cfg.USE_GPU_NMS = cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = load_dir + "/" + net + "_handobj_100K" + "/" + dataset
    if not osp.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = osp.join(model_dir,
                         'faster_rcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))
    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])

    # initilize the network here.
    fasterRCNN = None
    if net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()
    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if cuda:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    return fasterRCNN


def extract_bbox_from_frame(
        fasterRCNN,
        image_path,
        json_path,
        cuda=True,
        thresh_hand=0.5,
        thresh_obj=0.5,
        class_agnostic=False,
        verbose=True
):
    if json_path is not None:
        json_dir = osp.dirname(json_path)
        os.makedirs(json_dir, exist_ok=True)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    if cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        cfg.CUDA = True
        fasterRCNN.cuda()

    fasterRCNN.eval()

    im = cv2.imread(image_path)
    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.resize_(1, 1, 5).zero_()
    num_boxes.resize_(1).zero_()
    box_info.resize_(1, 1, 5).zero_()

    det_tic = time.time()

    with torch.no_grad():
        (
            rois, cls_prob, bbox_pred,
            rpn_loss_cls, rpn_loss_box,
            RCNN_loss_cls, RCNN_loss_bbox,
            rois_label, loss_list
        ) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0]  # hand contact state info
        offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

        # get hand contact
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if cuda:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                if class_agnostic:
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        obj_dets, hand_dets = None, None
        for j in xrange(1, len(pascal_classes)):
            if pascal_classes[j] == 'hand':
                inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
            elif pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds],
                                      offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if pascal_classes[j] == 'targetobject':
                    obj_dets = cls_dets.cpu().numpy()
                if pascal_classes[j] == 'hand':
                    hand_dets = cls_dets.cpu().numpy()

    misc_toc = time.time()
    nms_time = misc_toc - misc_tic

    if verbose:
        print(f'Finished bounding box extraction for frame: {image_path}.')
        print(f'Detect time: {detect_time}; NMS time: {nms_time}')

    success = True
    json_info = None
    if hand_dets is None:
        if verbose:
            print('No bounding box detected for this frame')
        success = False
    else:
        json_info = {"image_path": image_path, "body_bbox_list": [[]]}

        if len(hand_dets) == 1:
            hand = [int(hand_dets[0][0]), int(hand_dets[0][1]),
                    int(hand_dets[0][2] - hand_dets[0][0]),
                    int(hand_dets[0][3] - hand_dets[0][1])]
            contact = int(hand_dets[0][5])
            if hand_dets[0][9] == 0:
                if verbose:
                    print('Detected left hand bounding box for this frame.')
                json_info["hand_bbox_list"] = [{"left_hand": hand, "right_hand": []}]
                json_info["contact_list"] = [{"left_hand": contact, "right_hand": -1}]
            elif hand_dets[0][9] == 1:
                if verbose:
                    print('Detected right hand bounding box for this frame.')
                json_info["hand_bbox_list"] = [{"left_hand": [],"right_hand": hand}]
                json_info["contact_list"] = [{"left_hand": -1, "right_hand": contact}]
            else:
                print(f"Error: unrecognized hand type: {hand_dets[0][9]}")
                success = False

        elif len(hand_dets) == 2:
            if verbose:
                print('Detected both left and right hand bounding boxes for this frame.')
            hand1 = [int(hand_dets[0][0]), int(hand_dets[0][1]),
                     int(hand_dets[0][2] - hand_dets[0][0]),
                     int(hand_dets[0][3] - hand_dets[0][1])]
            hand2 = [int(hand_dets[1][0]), int(hand_dets[1][1]),
                     int(hand_dets[1][2] - hand_dets[1][0]),
                     int(hand_dets[1][3] - hand_dets[1][1])]
            contact1 = int(hand_dets[0][5])
            contact2 = int(hand_dets[1][5])
            if hand_dets[0][9] == 0:
                json_info["hand_bbox_list"] = [{"left_hand": hand1, "right_hand": hand2}]
                json_info["contact_list"] = [{"left_hand": contact1, "right_hand": contact2}]
            elif hand_dets[0][9] == 1:
                json_info["hand_bbox_list"] = [{"left_hand": hand2, "right_hand": hand1}]
                json_info["contact_list"] = [{"left_hand": contact2, "right_hand": contact1}]
            else:
                print(f"Error: unrecognized hand type: {hand_dets[0][9]}")
                success = False

        else:
            print("More than two bounding boxes detected for this frame.")
            success = False

        if success:
            if verbose:
                print('Successfully detected hand bbox.')
            if json_path is not None:
                with open(json_path, 'w') as outfile:
                    json.dump(json_info, outfile)
                if verbose:
                    print(f'Saved extracted bounding box info to: {json_path}')
        else:
            if verbose:
                print('Hand bbox detection failed. No json file saved.')

    return json_info, im, success


if __name__ == '__main__':
    task_dir = '/home/junyao/Datasets/something_something_hand_demos_same_hand/move_up'
    image_path = osp.join(task_dir, 'frames/1/frame0.jpg')
    # json_path = osp.join(task_dir, 'bbs_json/1/frame0.json')
    json_path = None

    fasterRCNN = set_up_bbox_extractor()
    extract_bbox_from_frame(
        fasterRCNN=fasterRCNN,
        image_path=image_path,
        json_path=json_path,
    )

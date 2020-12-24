from utils.utils import *
from dataset.vocdataset import VOC_CLASSES
from dataset.cocodataset import COCO_CLASSES
from dataset.data_augment import ValTransform
from utils.vis_utils import vis
from time import time
import submission_builder
import numpy as np
import os
import sys
import argparse
import yaml
import cv2
from tqdm import tqdm
import torch
from torch.autograd import Variable
from colorama import Fore, Back, Style
from detection_package import *

writer = submission_builder.SubmissionWriter('submission', rvc)


seq = os.listdir(path)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', '--iteration', type=int, default=1)
    parser.add_argument('-nms1', '--nms_threshold1', type=float, default=1)
    parser.add_argument('-conf', '--conf_threshold', type=float, default=0.03)
    parser.add_argument('-conf2', '--conf2_threshold', type=float, default=0.035)
    parser.add_argument('-prob', '--prob_threshold', type=float, default=0.03)
    parser.add_argument('-box', '--box_threshold', type=int, default=0)
    parser.add_argument('-label1', '--label1', type=int, default=0)
    parser.add_argument('-sh', '--sharpening', type=float, default=2)
    parser.add_argument('-boxa', '--boxa', type=int, default=0)
    parser.add_argument('-albedo', '--albedo_process', type=bool, default=0)
    parser.add_argument('-Inter', '--Inter', type=bool, default=0)
    parser.add_argument('-text', '--text_write', type=bool, default=0)
    parser.add_argument('-saver', '--image_save', type=bool, default=1)
    parser.add_argument('-nms2', '--nms_threshold2', type=float, default=1)
    parser.add_argument('-folder', '--folder', type=str, default='test')
    parser.add_argument('--cfg', type=str, default='config/yolov3_baseline.cfg',
                        help='config file. see readme')
    parser.add_argument('-d', '--dataset', type=str, default='COCO')
    parser.add_argument('-i', '--img', type=str, default='./img/',)
    parser.add_argument('-c', '--checkpoint', default='./weights/YOLOv3-ASFF_800_43.9.pth', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('-s', '--test_size', type=int, default=800)
    parser.add_argument('--half', dest='half', action='store_true', default=False,
                        help='FP16 training')
    parser.add_argument('--rfb', dest='rfb', action='store_true', default=True,
                        help='Use rfb block')
    parser.add_argument('--asff', dest='asff', action='store_true', default=True,
                        help='Use ASFF module for yolov3')
    parser.add_argument('--use_cuda', type=bool, default=True)
    return parser.parse_args()


def demo():

    args = parse_args()
    cuda = torch.cuda.is_available() and args.use_cuda

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    dtclass = [[],[],[]]
    backbone=cfg['MODEL']['BACKBONE']
    test_size = (args.test_size, args.test_size)
    # test_size = (640, 480)

    if args.dataset == 'COCO':
        class_names = COCO_CLASSES #
        num_class=80 #
    elif args.dataset == 'VOC':
        class_names = VOC_CLASSES
        num_class=20
    else:
        raise Exception("Only support COCO or VOC model now!")

    if args.asff:
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
        else:
            from models.yolov3_asff import YOLOv3 #
            print(f'{Fore.YELLOW}HELLO{Style.RESET_ALL}') #
        model = YOLOv3(num_classes = num_class, rfb=args.rfb, asff=args.asff) #
    else:
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
        else:
            from models.yolov3_baseline import YOLOv3
        model = YOLOv3(num_classes = num_class, rfb=args.rfb)

    if args.checkpoint:
        cpu_device = torch.device("cpu") #
        ckpt = torch.load(args.checkpoint, map_location=cpu_device) #
        model.load_state_dict(ckpt) #
    if cuda:
        torch.backends.cudnn.benchmark = True #
        device = torch.device("cuda")
        model = model.to(device)

    if args.half:
        model = model.half()
    model = model.eval()
    dtype = torch.float16 if args.half else torch.float32

    sequence = 50
    rr = gg = bb = r = g = b = np.zeros(50, dtype=float)

    transform = ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229,0.224,0.225))

    for a, seq_1 in enumerate(seq):

        if args.image_save:
            if not os.path.exists(os.path.join('./results/' + seq_1)):
                os.makedirs(os.path.join('./results/' + seq_1))

        lst = os.listdir(path + seq_1)
        num = len(lst)

        for i in tqdm(range(0, num)):
            if args.albedo_process:
                args.img, args_img_2 = load_img(i, seq_1, args.albedo_process)
            else:
                args.img = load_img(i, seq_1, args.albedo_process)

            im = cv2.imread(args.img)
            if args.albedo_process:
                im_2 = cv2.imread(args_img_2)

            if i > (sequence - 1):
                for j in range(sequence - 1):
                    r[j] = r[j+1]
                    g[j] = g[j+1]
                    b[j] = b[j+1]

                r[sequence-1] = np.mean(im[:, :, 0] / 255)
                g[sequence-1] = np.mean(im[:, :, 1] / 255)
                b[sequence-1] = np.mean(im[:, :, 2] / 255)
                rr[sequence-1] = np.std(im[:, :, 0] / 255)
                gg[sequence-1] = np.std(im[:, :, 1] / 255)
                bb[sequence-1] = np.std(im[:, :, 2] / 255)
                transform = ValTransform(rgb_means=(((sum(r) / sequence)+0.485) / 2, ((sum(g) / sequence)+0.456) / 2,
                                                    ((sum(b) / sequence)+0.406) / 2),
                                         std=(((sum(rr) / sequence) + 0.229) / 2, ((sum(gg) / sequence) + 0.224)/2,
                                              ((sum(bb) / sequence) + 0.225) / 2))
            else:
                r[i] = np.mean(im[:, :, 0] / 255)
                g[i] = np.mean(im[:, :, 1] / 255)
                b[i] = np.mean(im[:, :, 2] / 255)
                rr[i] = np.std(im[:, :, 0] / 255)
                gg[i] = np.std(im[:, :, 1] / 255)
                bb[i] = np.std(im[:, :, 2] / 255)

            height, width, _ = im.shape
            ori_im = im.copy()

            # Image read
            im_input, _ = transform(im, None, test_size)
            if args.albedo_process:
                im_input_2, _ = transform(im_2, None, test_size)

            # Image load
            if cuda:
                im_input = im_input.to(device)
                if args.albedo_process:
                    im_input_2 = im_input_2.to(device)

            # Image to Variable
            im_input = Variable(im_input.type(dtype).unsqueeze(0))
            if args.albedo_process:
                im_input_2 = Variable(im_input_2.type(dtype).unsqueeze(0))

            iter_ = 0

            sum_put = None

            if args.albedo_process:
                for iter in range(args.iteration):
                    A = 0

                    outputs_1 = model(im_input, iter=iter)
                    outputs_2 = model(im_input_2, iter=iter)

                    outputs, ret_with_prob = postprocess(outputs_1, num_class, args.conf_threshold, args.nms_threshold1)
                    ret_with_prob = [torch.Tensor(ret_with_prob)]

                    # Integrate result of first image
                    if outputs[0] is not None:
                        if iter_ == 0:
                            sum_put = outputs[0]
                            sum_prob_put = ret_with_prob[0]
                            iter_ += 1
                        else:
                            sum_put = torch.cat([sum_put, outputs[0]], dim=0)
                            sum_prob_put = torch.cat([sum_prob_put, ret_with_prob[0]], dim=0)

                    else:
                        A = 1
                    outputs, ret_with_prob = postprocess(outputs_2, num_class, args.conf2_threshold, args.nms_threshold1)
                    ret_with_prob = [torch.Tensor(ret_with_prob)]

                    # Integrate result of second image
                    if A == 1 and outputs[0] is not None :
                        if iter_ == 0:
                            sum_put = outputs[0]
                            sum_prob_put = ret_with_prob[0]
                            iter_ += 1
                        else:
                            sum_put = torch.cat([sum_put, outputs[0]], dim=0)
                            sum_prob_put = torch.cat([sum_prob_put, ret_with_prob[0]], dim=0)

                    elif outputs[0] is not None:
                        sum_put = torch.cat([sum_put, outputs[0]], dim=0)
                        sum_prob_put = torch.cat([sum_prob_put, ret_with_prob[0]], dim=0)

            else:
                for iter in range(args.iteration):
                    outputs = model(im_input, iter=iter)
                    outputs, ret_with_prob = postprocess(outputs, num_class, args.conf_threshold, args.nms_threshold1)
                    ret_with_prob = [torch.Tensor(ret_with_prob)]

                    if outputs[0] is not None:
                        if iter_ == 0:
                            sum_put = outputs[0]
                            sum_prob_put = ret_with_prob[0]
                            iter_ += 1
                        else:
                            sum_put = torch.cat([sum_put, outputs[0]], dim=0)
                            sum_prob_put = torch.cat([sum_prob_put, ret_with_prob[0]], dim=0)

            outputs[0] = sum_put
            ret_with_prob[0] = sum_prob_put

            if outputs[0] is not None:
                sel_bbox, scores, cls = bbox_score_class(outputs, height, width, test_size)

                label = []
                for s in range(len(ret_with_prob[0])):
                    label.append(ret_with_prob[0][s][7:88])

                if args.label1 :
                    prob = onehot_encoder(cls)
                else :
                    prob = make_81(label, cls)
                meanmean_label, meanmean_bbox, meanmean_score = coco2rvc(prob, sel_bbox, scores, args.sharpening, args.label1)

                processed_bbox = width_height2corner(meanmean_bbox)
                obv, selected_bbox, selected_scores, cov_list = det2obv(processed_bbox, meanmean_label,
                                                                        args.nms_threshold2, args.box_threshold)

                final_detections, location = rvc_writer(selected_scores, selected_bbox, cov_list, args.prob_threshold)

                for t in range(len(selected_bbox)):
                    cv2.rectangle(im,
                                  (int(selected_bbox[t][0]), int(selected_bbox[t][1])),
                                  (int(selected_bbox[t][2]), int(selected_bbox[t][3])),
                                  (0, 0, 255), 1)

                         ##########################Inter frame processing##################
                if args.Inter == 1:
                    result = list()
                    dtclass[i % 3] = []

                    if len(final_detections) > 0:
                        for n in range(len(final_detections)):
                            dtclass[i % 3].extend([final_detections[n][0].index(max(final_detections[n][0]))])

                    if i>0 and len(copy) > 0:
                        if i==1:
                            for n in range(len(copy)):
                                if copy[n][0].index(max(copy[n][0])) in dtclass[1] :
                                    result.append(copy[n])
                        elif i>1 :
                            for n in range(len(copy)):
                                if copy[n][0].index(max(copy[n][0])) in dtclass[i%3] or copy[n][0].index(max(copy[n][0])) in dtclass[(i%3)-2] :
                                    result.append(copy[n])

                    if args.text_write and i>0 :
                        if len(result) > 0:                              ## 텍스트로 검출한거 정리하는 구문
                            if i != 0:
                                if not os.path.exists("submission/{}".format(args.folder)):
                                    os.makedirs("submission/{}".format(args.folder))
                                f = open("submission/{}/{}.txt".format(args.folder, seq_1), 'a')
                            f.write('\n %d ' %(i-1))
                            for n in range(len(result)):
                                f.write('%s %f, ' %(rvc[result[n][0].index(max(result[n][0]))], max(result[n][0])))
                        f.close()

                    if args.image_save:
                        cv2.imwrite('./results/' + seq_1 + '/' + str(i) + '.png', im)
                    copy = final_detections

                else:
                    if args.text_write and i > 0:
                        if len(final_detections) > 0:                              ## 텍스트로 검출한거 정리하는 구문
                            if i != 0:
                                if not os.path.exists("submission/{}".format(args.folder)):
                                    os.makedirs("submission/{}".format(args.folder))
                                f = open("submission/{}/{}.txt".format(args.folder, seq_1), 'a')
                            f.write('\n %d ' %(i-1))
                            for n in range(len(final_detections)):
                                f.write('%s %f, ' %(rvc[final_detections[n][0].index(max(final_detections[n][0]))], max(final_detections[n][0])))
                        f.close()

                    if args.image_save:
                        cv2.imwrite('./results/' + seq_1 + '/' + str(i) + '.png', im)

            if args.Inter == 1 :
                if i>0 and i < num -1 :
                    for detection in result:
                        if args.boxa :
                            x = round((detection[3]-detection[1])/40)
                            y = round((detection[4]-detection[2])/40)
                            detection[1]+=x
                            detection[2]+=y
                            detection[3]-=x
                            detection[4]-=y
                        writer.add_detection(detection)
                    writer.next_image()
                elif i==num-1:
                    for detection in result:
                        if args.boxa:
                            x = round((detection[3] - detection[1]) / 40)
                            y = round((detection[4] - detection[2]) / 40)
                            detection[1] += x
                            detection[2] += y
                            detection[3] -= x
                            detection[4] -= y
                        writer.add_detection(detection)
                    writer.next_image()
                    for detection in final_detections:
                        if args.boxa:
                            x = round((detection[3] - detection[1]) / 40)
                            y = round((detection[4] - detection[2]) / 40)
                            detection[1] += x
                            detection[2] += y
                            detection[3] -= x
                            detection[4] -= y
                        writer.add_detection(detection)
                    writer.next_image()
            else :
                for detection in final_detections:
                    if args.boxa:
                        x = round((detection[3] - detection[1]) / 40)
                        y = round((detection[4] - detection[2]) / 40)
                        detection[1] += x
                        detection[2] += y
                        detection[3] -= x
                        detection[4] -= y
                    writer.add_detection(detection)
                writer.next_image()

        writer.save_sequence(seq_1,args.folder)

if __name__ == '__main__':
    demo()
import numpy as np
import torch
import json

def print_per_class_iou(class_iou,
                        class_name_list_in,
                        line_width=6,
                        cell_width=12):
    class_name_list = class_name_list_in.copy()
    name_len = len(class_name_list)
    class_iou = list(class_iou)
    mean_iou = sum(class_iou)/name_len
    class_iou = ['{:.2%}'.format(i) for i in class_iou]
    if name_len % line_width != 0 and name_len//line_width > 1:
        lines = name_len//line_width + 1
        for k in range(line_width - name_len % line_width):
            class_iou.append(' ')
            class_name_list.append(' ')
    elif name_len//line_width > 1:
        lines = name_len//line_width
    else:
        lines = 1

    for k in range(lines):
        K = k*line_width
        class_name_list_tem = class_name_list[K:K+line_width]
        class_iou_tem = class_iou[K:K+line_width]
        name_list = ''.join(['|{:{align}{width}}'.format(
            i, align='^', width=cell_width) for i in class_name_list_tem])+'|'
        print(len(name_list)*'-')
        print(name_list)
        print(len(name_list)*'-')
        print(''.join(['|{:{align}{width}}'.format(
            i, align='^', width=cell_width) for i in class_iou_tem])+'|')

    print(len(name_list)*'-')
    acc_out = 'Mean IoU {:.2%}'.format(mean_iou)
    print('|' +
          '{:{align}{width}}'.format(acc_out, align='^',
                                     width=line_width*(cell_width+1)-1)
          + '|')
    print(len(name_list)*'-')


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def _evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, iu, mean_iu, fwavacc


# ------------------------------------------------------------------------------
# -----  OBSOLETE IMPLEMENTATION, NOT CONSISTENT WITH OFFICIAL EVALUATION ------
# ------------------------------------------------------------------------------
# def test_miou(
#     model, loader, upsample=None,
#     info_path=None, num_classes=19,
#     writer=None, print_results=True,
# ):
#     model.eval()
#     gts_all, predictions_all = [], []
#     with torch.no_grad():
#         for idx, data in enumerate(loader):
#             inputs, gts = data[0].cuda(), data[1]
#             N = inputs.size(0)

#             outputs = model(inputs)
#             if isinstance(outputs, (list, tuple)):
#                 outputs = outputs[0]
#             if upsample is not None:
#                 outputs = upsample(outputs)
#             predictions = outputs.data.max(1)[1].squeeze_(1).cpu()

#             gts_all.append(gts.numpy())
#             predictions_all.append(predictions.numpy())

#     gts_all = np.concatenate(gts_all)
#     predictions_all = np.concatenate(predictions_all)

#     acc, acc_cls, iu, mean_iu, fwavacc = _evaluate(
#         predictions_all, gts_all, num_classes
#     )
#     if print_results:
#         if info_path:
#             with open(info_path, 'r') as fp:
#                 info = json.load(fp)
#             name_classes = info['label']
#             print_per_class_iou(iu, name_classes, line_width=7)
#         else:
#             print(100*'-')
#             print(f'Mean IoU: {mean_iu:.3f}')
#             print(100*'-')
#     return mean_iu
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def get_confusion_matrix(gt_label, pred_label, class_num):
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label,i_pred_label] = label_count[cur_index]

    return confusion_matrix


def test_miou(
    model, loader, upsample=None,
    info_path=None, num_classes=19,
    writer=None, print_results=True,
):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for idx, data in enumerate(loader):
            inputs, gts = data[0].cuda(), data[1]
            N = inputs.size(0)

            outputs = model(inputs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            if upsample is not None:
                outputs = upsample(outputs)
            predictions = outputs.data.argmax(dim=1).cpu().numpy()
            gts = gts.numpy()
            for gt, prediction in zip(gts, predictions):
                ignore_index = gt != 255
                gt = gt[ignore_index]
                prediction = prediction[ignore_index]
                confusion_matrix += get_confusion_matrix(
                    gt, prediction, num_classes
                )

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    iu = (tp / np.maximum(1.0, pos + res - tp))
    mean_iu = iu.mean()

    if print_results:
        if info_path:
            with open(info_path, 'r') as fp:
                info = json.load(fp)
            name_classes = info['label']
            print_per_class_iou(iu, name_classes, line_width=7)
        else:
            print(100*'-')
            print(f'Mean IoU: {mean_iu:.3f}')
            print(100*'-')
    return mean_iu



def cls_evaluate(model, data_loader):
        model.eval()
        class_name_list = data_loader.dataset.classes
        num_classes = len(class_name_list)
        class_acc = np.zeros(num_classes)
        samples_per_class = np.zeros(num_classes)
        for batch_idx, batch in enumerate(data_loader):
            inputs, labels = batch[0].cuda(), batch[1].cuda()
            outputs = model(inputs)
            _, predict_labels = torch.max(outputs, 1)
            for index, predict_label in enumerate(predict_labels):
                tem_true_tensor = labels[index] == predict_label
                class_acc[labels[index]] += tem_true_tensor.item()
                samples_per_class[labels[index]] += 1
        acc = np.sum(class_acc)/np.sum(samples_per_class)
        return acc

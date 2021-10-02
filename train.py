from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dataset import MosMed
from model import UNet
import numpy as np
from torchnet import meter
from tensorboardX import SummaryWriter
import random

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

def cross_entropy2d(input, target, weight=None, size_average=True):
    #NxCxHxW
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim = 1)

    #log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    #log_p = log_p.view(-1, c)
    #print(log_p.size())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=size_average)
    return loss

def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls_avg = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls_avg,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu, acc_cls

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

train_epoches = 500
val_period = 5
SummaryWriterPath = 'datahackthon/log/segmentation'
SaveModelPath = 'datahackthon/model'
slides_path = './COVID19_1110/slides/'


model = UNet(1, 2).cuda()
#test = model(torch.zeros(1,1,512,512))
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
#loss = cross_entropy2d(test, torch.ones(1,512,512, dtype = int))
#print(loss)

writer = SummaryWriter(SummaryWriterPath)
train_loss_meter = meter.AverageValueMeter()
val_loss_meter = meter.AverageValueMeter()
accuracy_meter = meter.ClassErrorMeter(accuracy=True)
confusion_meter = meter.ConfusionMeter(2)
train_loss_meter.reset()
val_loss_meter.reset()
accuracy_meter.reset()
confusion_meter.reset()

train_set = MosMed(slides_path, split = 'train')
val_set = MosMed(slides_path, split = 'val')

train_loader = torch.utils.data.DataLoader(train_set, 16, shuffle=False, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, 16, shuffle=False, drop_last=True)

if 1:
    for epoch in range(train_epoches):
        if 1:
            for idx, data in enumerate(train_loader):
                #print('index',idx)
                input_image, target = data
                optimizer.zero_grad()
                #print(input_image.size())
                #print(target.size())
                output = model(input_image.cuda())
                loss = cross_entropy2d(output, target.cuda(), weight =  torch.FloatTensor([1, 475]).cuda())
                loss.backward()
                optimizer.step()
                train_loss_meter.add(loss.cpu().data)
                #print(loss)

        print('train loss', train_loss_meter.value()[0])
        writer.add_scalar('train loss', train_loss_meter.value()[0], epoch)

        if (epoch%val_period)==0:
            model.eval()
            gts = []
            preds = []
            with torch.no_grad():
                target_list = []
                predict_list = []
                for idx, data in enumerate(val_loader):
                    input_image, target = data
                    output = model(input_image.cuda())
                    prob = torch.nn.functional.softmax(output)
                    #[Batch, num_class]
                    predict_cls = torch.argmax(prob, axis = 1)
                    loss = cross_entropy2d(output, target.cuda(), weight =  torch.FloatTensor([1, 475]).cuda())
                    #print(predict_cls.size())
                    #accuracy_meter.add(output.view(-1), target.view(-1))
                    gts += list(target.numpy())
                    preds += list(predict_cls.data.cpu().numpy())
                    target_list.append(target.squeeze().tolist())
                    predict_list.append(predict_cls.squeeze().tolist())
                    #Todo Confusion matrix
                    val_loss_meter.add(loss.cpu().data)
                    #print(predict_cls)

            #print('epoches:', epoch)
            #print('prediction')
            #plt.imshow(predict_cls.data.cpu().numpy()[0])
            #plt.show()
            #print('labels')
            #plt.imshow(target.data.cpu().numpy()[0])
            #plt.show()
            #print(predict_cls.data.cpu().numpy()[0])
            #print(scores(gts, preds, 2))
            result,cls_iu, acc_cls = scores(gts, preds, 2)
            print(cls_iu)
            print('val loss', val_loss_meter.value()[0])
            writer.add_image('ground truth', target.data.cpu().numpy()[0][np.newaxis,:,:].repeat(3, axis = 0), epoch)
            writer.add_image('prediction', predict_cls.data.cpu().numpy()[0][np.newaxis,:,:].repeat(3, axis = 0), epoch)
            writer.add_image('input_image', input_image.data.numpy()[0].repeat(3, axis = 0), epoch)
            writer.add_scalar('validation loss', val_loss_meter.value()[0], epoch)
            writer.add_scalar('Background IoU', cls_iu[0], epoch)
            writer.add_scalar('GroundGlass IoU', cls_iu[1], epoch)
            #print('val accuracy', accuracy_meter.value())
            model.train()
        if (epoch%val_period)==0:
            torch.save(model.state_dict(), SaveModelPath + '/{}.model'.format(epoch))

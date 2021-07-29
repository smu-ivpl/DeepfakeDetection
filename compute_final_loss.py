from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve, auc, roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import log_loss


def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.clf()
    """
    This function prints and plots the confusion matrix very prettily.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    # Specify the tick marks and axis text
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # The data formatting
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Print the text of the matrix, adjusting text colour for display
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.savefig('confusion_matrix_{:03d}.png'.format(epoch))
    plt.savefig("confusion_matrix_vit_b7.pdf", bbox_inches='tight')
    plt.show()

loss = nn.BCEWithLogitsLoss()

test_path = '/home/yjheo/Deepfake/dataset/dfdc_facebook/test/labels.csv'
label = pd.read_csv(test_path)
predict = pd.read_csv('final/SOTA_last_weight.csv')
predict2 = pd.read_csv('final/ViT_Distill_last_weight.csv')

label = torch.from_numpy(label['label'].values).unsqueeze(dim=1).float()
predict = torch.from_numpy(predict['label'].values).unsqueeze(dim=1).float()
predict2 = torch.from_numpy(predict2['label'].values).unsqueeze(dim=1).float()

pyplot.plot(predict, predict2, marker='.',  linestyle = 'None',color='b',markersize=1,label='Our model')
pyplot.savefig("correlation_sota_our.pdf", bbox_inches="tight")
pyplot.show()
#pyplot.plot(lr_fpr, lr_tpr, marker='.', markersize=0.5,label='Our model')
total_loss = 0
total_loss2 = 0
#for i in range(len(label)):
#    total_loss += label[i]*torch.log(predict[i]) +(1-label[i])*torch.log(1-predict[i])
#    total_loss2 += label[i] * torch.log(predict2[i]) + (1 - label[i]) * torch.log(1 - predict2[i])
#total_loss /= len(label)
#total_loss2 /= len(label)
#total_loss = -torch.sum(label*torch.log(predict) + (1-label)*torch.log(1-predict)) / len(label)
#total_loss2 = -torch.sum(label*torch.log(predict2) + (1-label)*torch.log(1-predict2)) / len(label)
#print("1 : ", -total_loss)
#print("2 : ", -total_loss2)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(5000)]
# calculate AUC
auc = roc_auc_score(label, predict)
auc2 = roc_auc_score(label, predict2)
print('AUC: %.3f' % auc)
print('AUC2: %.3f' % auc2)
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(label, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(label, predict)
lr_fpr2, lr_tpr2, _ = roc_curve(label, predict2)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', markersize=0.5,label='Our model')
pyplot.plot(lr_fpr2, lr_tpr2, marker='.', markersize=0.5,label='SOTA')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.savefig("AUC_vit_b7.pdf", bbox_inches='tight')
pyplot.show()

lr_precision, lr_recall, _ = precision_recall_curve(label, predict)
lr_precision2, lr_recall2, _ = precision_recall_curve(label, predict2)
yhat = predict
yhat2 = predict2


print("1:",len(yhat[np.where(yhat >= 0.5)]))
yhat[np.where(yhat >= 0.55)] = 1.
yhat[np.where(yhat < 0.55)] = 0.

print("2:",len(yhat2[np.where(yhat2 >= 0.5)]))
yhat2[np.where(yhat2 >= 0.45)] = 1.
yhat2[np.where(yhat2 < 0.45)] = 0.
lr_f1 = f1_score(label.squeeze(1), yhat)
lr_f1_2 = f1_score(label.squeeze(1), yhat2)
# summarize scores
print('Our Model: f1=%.3f' % (lr_f1))
print('2: f1=%.3f' % (lr_f1_2))
# plot the precision-recall curves
no_skill = len(label[label==1]) / len(label)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', markersize=0.5,label='1')
pyplot.plot(lr_recall2, lr_precision2, marker='.', markersize=0.5,label='2')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.savefig("ROC_vit_b7.pdf", bbox_inches='tight')
pyplot.show()

cm = confusion_matrix(label.squeeze(1), predict)
cm2 = confusion_matrix(label.squeeze(1), predict2)
plot_confusion_matrix(cm, {"Real:0", "Fake:1"})
plot_confusion_matrix(cm2, {"Real:0", "Fake:1"})

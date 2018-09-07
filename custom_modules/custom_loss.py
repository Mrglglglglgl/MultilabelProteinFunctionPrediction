""" 
a class that represents the margin-based ranking loss criterion. 
Both the forward and backward passes are manually implemented
"""

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import numpy as np

class RankingLoss(Function): 

    @staticmethod
    def forward(ctx, input, target):
    	# set the margin value
        margin = 1.0

        L = torch.zeros(input.size()[0])
        all_labels_idx = np.arange(target.size()[1])
        input = torch.sigmoid(input)

        # filter out the data that do not contain any positive labels
        batch_normalizer_1 = torch.FloatTensor([np.sum(np.any(target, axis=1))])
        batch_normalizer_2 = torch.FloatTensor([np.sum(np.all(target, axis=1))])
        batch_normalizer = batch_normalizer_1 - batch_normalizer_2
        if batch_normalizer == 0:
            batch_normalizer = 1
        positive_indices = torch.zeros(target.size())
        negative_indices = torch.zeros(target.size())
        new_target = target.cpu().detach().data.numpy()[np.any(target == 1, axis = 1)]
        new_input = input.cpu().detach().data.numpy()[np.any(target == 1, axis = 1)]
        backup_target = new_target.copy()
        new_target = torch.FloatTensor(new_target[~np.all(new_target == 1, axis = 1)])
        new_input = torch.FloatTensor(new_input[~np.all(backup_target == 1, axis = 1)])
        J = torch.nonzero(new_target)

        # loop through the data and their labels
        for counter,i in enumerate(list(torch.LongTensor(np.setdiff1d(np.where(np.any(target,axis=1))[0] ,np.where(np.all(target,axis=1))[0])))):
            # create a list to store the labels
            list_of_positive_labels = []
            # msk indicates whether a label can be visited, True mean it can be visited
            # at the beginning all labels can be visited (non of them has been considered yet)
            msk = np.ones(target.size()[1], dtype = bool)
            # Find the positive labels for this example
            for element in range(J.size()[0]):
                if J[element][0] == i :
                    j = J[element][1]
                    list_of_positive_labels.append(j)
                    msk[j] = False
            # if you found any loop through them
            if list_of_positive_labels:
                for pos_index, positive_label in enumerate(list_of_positive_labels):
                    # the error is -1 at the beginning
                    sample_score_margin = -1
                    # msk hold the positive label indices (we cannot loop select them)
                    msk_2 = msk.copy()
                    # get the negative labels so we can loop through them
                    neg_labels_idx = all_labels_idx[msk_2]
                    # if there are any negative labels loop through them
                    if list(neg_labels_idx):
                        for neg_index, neg_idx in enumerate(list(neg_labels_idx)):
                                # mark the visited negative index as visited for this label
                                msk_2[neg_idx] = False
                                # get the updated negative indices
                                neg_labels_idx = all_labels_idx[msk_2]
                                # compute the error fot that particular negative index
                                sample_score_margin = input[i, neg_idx] - input[i, positive_label] + margin
                                label_normalizer[i] += 1
                                if sample_score_margin < 0:
                                    # checks if no violating examples have been found 
                                    continue
                                else:
                                    # if a violation is found compute add to the loss
                                    negative_indices[i, neg_idx] += 1
                                    positive_indices[i, positive_label] += 1
                                    L[i] += margin
        label_normalizer =  torch.sum(positive_indices !=0 , dim=1).float()
        label_normalizer[label_normalizer == 0] = 1.0                        
        loss = (-torch.sum(positive_indices*input, dim = 1) + torch.sum(negative_indices*input, dim = 1) + L) / label_normalizer
        ctx.save_for_backward(input, target)
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        ctx.batch_normalizer = batch_normalizer
        ctx.label_normalizer = label_normalizer

        return torch.sum(loss , dim = 0, keepdim = True) /batch_normalizer

    @staticmethod
    def backward(ctx, grad_output):

        input, target = ctx.saved_variables
        positive_indices = Variable(ctx.positive_indices, requires_grad = False) 
        negative_indices = Variable(ctx.negative_indices, requires_grad = False)
        label_normalizer = ctx.label_normalizer
        batch_normalizer = ctx.batch_normalizer
        grad_input = grad_output*(negative_indices - positive_indices)/label_normalizer.view(-1,1)

        return grad_input/batch_normalizer, None, None    

      
class RankLoss(nn.Module): 
    def __init__(self): 
        super(RankLoss, self).__init__()
        
    def forward(self, input, target): 
        return RankingLoss.apply(input, target)
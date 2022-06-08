from __future__ import print_function
from tkinter import E

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class ContrastiveRanking(nn.Module):
    def __init__(self, opt, gen_model):
        super().__init__()
        self.m = opt.m
        self.dataset = opt.dataset
        self.do_sum_in_log = opt.do_sum_in_log
        self.feature_size = 128

        self.backbone_q = gen_model(name=opt.model)
        self.backbone_k = gen_model(name=opt.model)
        for param_k, param_q in zip(self.backbone_k.parameters(), self.backbone_q.parameters()):
            param_k.data = param_q.data
            param_k.requires_grad = False
        self.register_buffer("memorybank_InfoNCE", torch.randn(opt.memorybank_size, self.feature_size))
        self.memorybank_InfoNCE = nn.functional.normalize(self.memorybank_InfoNCE, dim=1)
        self.register_buffer("memorybank_labels", torch.ones(opt.memorybank_size, dtype=torch.long) * -1)

        self.min_tau = opt.min_tau
        self.max_tau = opt.max_tau
        self.similarity_threshold = opt.similarity_threshold
        self.n_sim_classes = opt.n_sim_classes
        self.use_dynamic_tau = opt.use_dynamic_tau
        self.use_all_ranked_classes_above_threshold = self.similarity_threshold > 0
        self.use_same_and_similar_class = opt.use_same_and_similar_class
        self.one_loss_per_rank = opt.one_loss_per_rank
        self.mixed_out_in = opt.mixed_out_in

        ## Classes is Cifar 100
        if opt.dataset == 'cifar100':
            class_names = np.load('./cifar100_idx2className.npy', allow_pickle=True).item()
        elif opt.dataset == 'cifar10' or 'voc':
            class_names = {}
            for key, values in opt.class_to_idx.items():
                class_names[values] = key

        self.class_names = class_names

        self.set_super_cat_sims(class_names)

        self.criterion = ContrastiveRankingLoss()



    def set_super_cat_sims(self, class_names):
        ############
        ## class names = {0:apple, 1:aquar....}
        ## name2idx = {apple:0, aquarium:1, ...}
        ## self.class_sims_idx = {0: {'sim_class_idx2name': ['apple', mushroom', ...]}} -> convert to one-hot encoding, and similarity scores, {0: {'sim_class_idx2name': ['apple', mushroom', ...], 'sim_class_idx2indices': [0, 2, 11, ...], 'sim_class_val': tensor([1, 0.7500, 0.7500...]}}
        ## 1 for itself, 0.75 for all similar classes including, 0 for others
        ############

        if self.dataset == 'cifar100':
            if self.dataset == 'cifar100':
                cats = {'aquatic mammals': 	['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        'flowers' :['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        'people': ['baby', 'boy', 'girl', 'man', 'woman'],
                        'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        'small mammals': [	'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
                        }

            # else:
            #     ## motorbike and chair removed
            #     cats = {'Person': ['person'],
            #             'Animal': ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep'],
            #             'Vehicle': ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'train'],
            #             'Indoor': ['bottle', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']}


            ## Swap from 0:apple to apple:0
            name2idx = {}
            for idx in class_names:
                key = class_names[idx]
                name2idx[key] = idx

            ## For every word find all its similar entities in the super_cat dict above
            ## Then for the word, put it first, followed by its super_cats, then all other words
            ## Convert this to one hot

            self.class_sims_idx = {}
            for idx in class_names.keys():
                self.class_sims_idx[idx] = {}
                word = class_names[idx]
                #get supercat of word
                for supercat in cats.keys():
                    if word in cats[supercat]:
                        similar_cats = copy.copy(cats[supercat])
                        similar_cats.remove(word)
                        similar_cats = [word] + similar_cats

                #sort out supercats from list of all classes
                other_cats = list(class_names.values())
                for cat in similar_cats:
                    other_cats.remove(cat)
                self.class_sims_idx[idx]['sim_class_idx2name'] = similar_cats + other_cats

                sim_class_idx2indices = [name2idx[word] for word in self.class_sims_idx[idx]['sim_class_idx2name']]
                self.class_sims_idx[idx]['sim_class_idx2indices'] = torch.tensor(sim_class_idx2indices).type(
                    torch.long).cuda()
                self.class_sims_idx[idx]['sim_class_val'] = torch.cat(
                    [torch.ones((len(similar_cats), 1), dtype=torch.float32) * 0.75,
                    torch.zeros((len(other_cats), 1), dtype=torch.float32)], dim=0).squeeze()
                self.class_sims_idx[idx]['sim_class_val'][0] = 1

        
        elif self.dataset == 'cifar10' or self.dataset == 'voc':
            if self.dataset == 'cifar10':
                ## removed ship and train
                cats = {'airplane': ['bird', 'automobile'],
                        'automobile': ['airplane', 'horse'],
                        'bird': ['airplane', 'frog', 'deer', 'horse'],
                        'cat': ['dog', 'deer', 'horse', 'frog'],
                        'deer': ['dog', 'cat', 'bird', 'frog'],
                        'dog': ['cat', 'deer', 'horse', 'frog'],
                        'frog': ['deer', 'bird', 'cat'],
                        'horse': ['dog', 'deer', 'cat', 'frog'],
                        }

            else:
                cats = {'person': [],
                        'bird': ['aeroplane'],
                        'cat': ['dog', 'sheep'],
                        'cow': [],
                        'dog': ['cat'],
                        'horse': ['dog', 'cat'],
                        'sheep': [],
                        'aeroplane': ['bird', 'boat'],
                        'bicycle': [],
                        'boat': ['car', 'bus'],
                        'bus': ['car', 'boat'],
                        'car': ['bus'],
                        'train': ['car'],
                        'bottle': ['pottedplant'],
                        'diningtable': ['sofa'],
                        'pottedplant': ['bottle'],
                        'tvmonitor': ['bottle']
                        }

            ## Swap from 0:apple to apple:0
            name2idx = {}
            for idx in class_names:
                key = class_names[idx]
                name2idx[key] = idx


            ## For every word find all its similar entities in the super_cat dict above
            ## Then for the word, put it first, followed by its super_cats, then all other words
            ## Convert this to one hot

            self.class_sims_idx = {}
            for idx in class_names.keys():
                self.class_sims_idx[idx] = {}
                word = class_names[idx]
                #get supercat of word
                for supercat in cats.keys():
                    if word == supercat:
                        similar_cats = copy.copy(cats[supercat])
                        similar_cats = [word] + similar_cats

                #sort out supercats from list of all classes
                other_cats = list(class_names.values())
                for cat in similar_cats:
                    other_cats.remove(cat)
                self.class_sims_idx[idx]['sim_class_idx2name'] = similar_cats + other_cats

                sim_class_idx2indices = [name2idx[word] for word in self.class_sims_idx[idx]['sim_class_idx2name']]
                self.class_sims_idx[idx]['sim_class_idx2indices'] = torch.tensor(sim_class_idx2indices).type(
                    torch.long).cuda()
                self.class_sims_idx[idx]['sim_class_val'] = torch.cat(
                    [torch.ones((len(similar_cats), 1), dtype=torch.float32) * 0.75,
                    torch.zeros((len(other_cats), 1), dtype=torch.float32)], dim=0).squeeze()
                self.class_sims_idx[idx]['sim_class_val'][0] = 1



    # video_view1, video_view2 are batches of the same videos but independently augmented
    def forward(self, anchor, pos, labels):
        # compute scores
        l_pos, l_class_pos, l_neg, masks, below_threshold, dynamic_taus = self.compute_InfoNCE_classSimilarity(
            anchor=anchor, pos=pos, labels=labels)

        #initially l_neg and l_class pos are identical
        res = {}
        for i, mask in enumerate(masks):
            if (self.use_same_and_similar_class and not i == 0):
                mask = masks[-1]
                for j in range(len(masks)-1):
                    mask = mask | masks[j]
                ## Get the mask of all objects that are somewhat similar to anchor
                ## All images above threshold -inf in l_neg
                ## l_neg already has the good matches from previous batch as -inf
                l_neg[mask & ~below_threshold[i]] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()
                #keep only members of current class
                ## All images not in mask have higher similarity score, so throw away
                l_class_pos_cur[~mask] = -float("inf")
                ## Other images in mask that have lower score need not be considered as positive 
                # throw out those batches for which the similarity between ranking class and label class is below threshold
                l_class_pos_cur[below_threshold[i]] = -float("inf")

            elif self.use_all_ranked_classes_above_threshold or (self.use_same_and_similar_class and i == 0):
                # mask out from negatives only if they are part of the class and this class has a similarity to
                # label class above the similarity threshold
                l_neg[mask & ~below_threshold[i]] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()
                l_class_pos_cur[~mask] = -float("inf")
                l_class_pos_cur[below_threshold[i]] = -float("inf")

            else:
                ## Set cos similarity of all positions from mask in neg as -inf
                ## Set cos similarity of all positions not from mask in pos as -inf 
                ## This simply means that initally there were no negative pairs, and we have ony positive pairs 
                l_neg[mask] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()
                l_class_pos_cur[~mask] = -float("inf")
            taus = dynamic_taus[i].view(-1, 1)

            if i == 0:
                ## l_pos is similarity of anchor with positive transformation
                ## l_class_pos_cur is similarity of anchor with memorybank
                l_class_pos_cur = torch.cat([l_pos, l_class_pos_cur], dim=1)

            if self.mixed_out_in and i == 0:
                loss = self.sum_out_log(l_class_pos_cur, l_neg, taus)
            elif self.do_sum_in_log and not(self.mixed_out_in and i ==0):
                loss = self.sum_in_log(l_class_pos_cur, l_neg, taus)
            else:
                loss = self.sum_out_log(l_class_pos_cur, l_neg, taus)

            result = {'score': None,
                      'target': None,
                      'loss': loss}
            res['class_similarity_ranking_class' + str(i)] = result

            if (self.use_same_and_similar_class and not i == 0):
                break


        return self.criterion(res, labels)

    def sum_in_log(self, l_pos, l_neg, tau):
        logits = torch.cat([l_pos, l_neg], dim=1) / tau
        logits = F.softmax(logits, dim=1)
        sum_pos = logits[:, 0:l_pos.shape[1]].sum(1)
        sum_pos = sum_pos[sum_pos > 1e-7]
        if len(sum_pos) > 0:
            loss = - torch.log(sum_pos).mean()
        else:
            loss = torch.tensor([0.0]).cuda()
        return loss

    def sum_out_log(self, l_pos, l_neg, tau):
        l_pos = l_pos / tau
        l_neg = l_neg / tau
        l_pos_exp = torch.exp(l_pos)
        l_neg_exp_sum = torch.exp(l_neg).sum(dim=1).unsqueeze(1)
        all_scores = (l_pos_exp / (l_pos_exp + l_neg_exp_sum))
        all_scores = all_scores[all_scores > 1e-7]
        if len(all_scores) > 0:
            loss = - torch.log(all_scores).mean()
        else:
            loss = torch.tensor([0.0]).cuda()
        return loss

    def get_similar_labels(self, labels):
        ## Select objects with similarity greater than threshold

        # in this case use top n classes
        labels = labels.cpu().numpy()

        sim_class_labels = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.long)
        sim_class_sims = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.float)
        sim_leq_thresh = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.bool)
        for i, label in enumerate(labels):
            sim_class_labels[i, :] = self.class_sims_idx[label]['sim_class_idx2indices']
            sim_class_sims[i, :] = self.class_sims_idx[label]['sim_class_val']
            sim_leq_thresh[i, :] = self.class_sims_idx[label]['sim_class_val'] >= self.similarity_threshold ## return True when value is above threshold
        # remove columns in which no sample has a similarity  equal to or larger than the selected threshold
        at_least_one_leq_thrsh = torch.sum(sim_leq_thresh, dim=0) > 0
        sim_class_labels = sim_class_labels[:, at_least_one_leq_thrsh]
        sim_leq_thresh = sim_leq_thresh[:, at_least_one_leq_thrsh]

        ## Get the top n similar classes
        sim_class_labels = sim_class_labels[:, :self.n_sim_classes]
        sim_class_sims = sim_class_sims[:, :self.n_sim_classes]

        # negate sim_leq_thresh to get a mask that can be applied to set all values below thresh to -inf
        sim_leq_thresh = ~sim_leq_thresh[:, :self.n_sim_classes]

        ###################
        ## sim_class_labels = [[0, 1, 2...], ...] ## labels of all similar classes
        ## sim_leq_thresh = [[False, False, ..., True, ...]] ## True when the similarity is less than threshold
        ## sim_class_sims = [[1, 0.75, ..], ...] ## Degree of similarity based on ranking
        ###################

        return sim_class_labels, sim_leq_thresh, sim_class_sims

    def compute_InfoNCE_classSimilarity(self, anchor, pos, labels, enqueue=True):
        ## l_pos is just dot product along rows of the two features for each pair of (anchor, positive)
        l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)
        similar_labels, below_threshold, class_sims = self.get_similar_labels(labels)
        masks = []
        threshold_masks = []
        dynamic_taus = []
        for i in range(similar_labels.shape[1]):
            ####################################
            ## None adds new dimension
            ## When the classname in memorybank and the similar labels match set that to one else 0
            ## Memory bank size 4096
            ## Repeat this for every n_similar_classes, i.e, masks has length of n_similar_classes, with each one of size (512, 4096) 
            ####################################
            mask = (self.memorybank_labels[:, None] == similar_labels[None, :, i]).transpose(0, 1)
            masks.append(mask)
            if self.use_all_ranked_classes_above_threshold:
                threshold_masks.append(below_threshold[None, :, i].transpose(0, 1).repeat(1, mask.shape[1]))
                ## all the values below similarity threshold get True, above get False
            dynamic_taus.append(self.get_dynamic_tau(class_sims[:, i]))
        #########
        ## dynamic_taus = [[0.1, 0.1, ... (512 times for each image in dataset)], ... (for n similar classes)]
        ## starts with min_tau for similarity 1, and goes to max_tau for similarity 0
        #########

        ## This is to get one score for a particular rank, i.e all images in tha rank are used
        if self.one_loss_per_rank:
            similarity_scores = reversed(class_sims.unique(sorted=True))
            similarity_scores = similarity_scores[similarity_scores > -1]
            new_masks = []
            new_taus = []
            ## For each image similar to the anchor with similarity_scores s
            for s in similarity_scores:
                new_taus.append(self.get_dynamic_tau(torch.ones_like(dynamic_taus[0]) * s))
                mask_all_siblings = torch.zeros_like(masks[0], dtype=torch.bool)
                ## Go through all the similar_labels, and check if they have similiarity scores s
                for i in range(similar_labels.shape[1]):
                    same_score = class_sims[:, i] == s
                    ## If so add them to your mask. This will be the set of positive pairs based on rank ie p_k 
                    if any(same_score):
                        mask_all_siblings[same_score] = mask_all_siblings[same_score] | masks[i][same_score]
                ## new masks will have the masks sorted by similarity scores, and all labels in one set will have same score 
                ## new_masks = [[False, False, .. True, ..], [False, ..]] True when similar score object found in bank else False
                new_masks.append(mask_all_siblings)
            masks = new_masks
            ## Tau will now have only k items where k is the number of distinct ranks
            dynamic_taus = new_taus

        ## l_class_pos is just matrix multiplication of each anchor with all elements in the bank to give total cosine similarity
        l_class_pos = torch.einsum('nc,ck->nk', [anchor, self.memorybank_InfoNCE.transpose(0, 1).clone()])
        l_neg = l_class_pos.clone()

        if self.training and enqueue:
            self.enqueue(pos, labels)

        return l_pos, l_class_pos, l_neg, masks, threshold_masks, dynamic_taus

    def enqueue(self, feature, labels):
        ## We store only the features from backbone_k
        m_dim = feature.shape[0]
        f = feature.detach()
        ## remove last batch_size image features every time
        self.memorybank_InfoNCE = torch.cat((f, self.memorybank_InfoNCE[:-m_dim, :]), dim=0)
        self.memorybank_labels = torch.cat((labels, self.memorybank_labels[:-m_dim]), dim=0)
        return self.memorybank_InfoNCE

    def update_weights(self):
        dict = {}
        for name, param in self.backbone_q.named_parameters():
            dict[name] = param
        for name, param_k in self.backbone_k.named_parameters():
            if name in dict:
                param_k.data = self.m * param_k.data + (1 - self.m) * dict[name].data


    def get_dynamic_tau(self, similarities):
        ## Returns a list of tau that are equally spaced between min_tau, max_tau
        dissimilarities = 1 - similarities
        d_taus = self.min_tau + (dissimilarities - 0) / (1 - 0) * (self.max_tau - self.min_tau)
        return d_taus

    def visualize_layers(self, writer_train, epoch):
        self.backbone_q.module.visualize_layers(writer_train, epoch)


class ContrastiveRankingLoss:
    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        loss = 0.0
        for key, val in outputs.items():
            if 'loss' in val:
                loss = loss + val['loss']
            else:
                loss = loss + self.cross_entropy(val['score'], val['target'])
        loss = loss / float(len(outputs))
        return loss
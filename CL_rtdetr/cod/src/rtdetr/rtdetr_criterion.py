

"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""

import  numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

import math
import random
from pytorch_metric_learning import miners, losses
from torch.nn import Parameter
# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register



def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P) ) # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()
    return T
def generate_ETF(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    
    #print(orth_vec.shape,"orth_vec   shape")
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(num_classes / (num_classes - 1)))
    
    
    return etf_vec.T

# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
  
#Hierachical Proxy
class_taxonomy_parent_lv3={0: [138], 1: [80], 2: [144, 204, 205], 3: [3], 4: [24], 5: [140, 141], 6: [172, 174], 7: [11, 13, 37, 68, 98, 124, 131, 132, 133, 134, 135], 8: [192], 9: [75, 125], 10: [219], 11: [209], 12: [136], 13: [198], 14: [73, 74], 15: [187], 16: [130], 17: [176], 18: [139], 19: [7], 20: [85], 21: [201], 22: [16, 17, 18, 19, 20], 23: [185], 24: [52], 25: [199], 26: [90], 27: [50], 28: [146], 29: [217], 30: [193], 31: [207], 32: [10], 33: [123], 34: [150], 35: [178], 36: [195], 37: [173], 38: [220], 39: [26, 83, 143, 191], 40: [188], 41: [145, 167], 42: [151], 43: [28, 84, 89, 152, 153, 154, 155, 156], 44: [27], 45: [78, 79, 216], 46: [214], 47: [137, 211], 48: [210], 49: [165], 50: [6, 12, 31, 36, 40, 48, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65], 51: [41, 42, 43], 52: [8, 9, 14, 25, 29, 39, 45, 71, 81, 93, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 121], 53: [0, 1, 2], 54: [189], 55: [34], 56: [157], 57: [148], 58: [97], 59: [175], 60: [160], 61: [96, 177], 62: [169], 63: [161, 162], 64: [159], 65: [142], 66: [213], 67: [122], 68: [117, 118, 119, 120, 158], 69: [186], 70: [206], 71: [95], 72: [51], 73: [35], 74: [163], 75: [164], 76: [215], 77: [15, 166], 78: [21], 79: [149], 80: [32, 126, 127, 128], 81: [86], 82: [168], 83: [190], 84: [170], 85: [179], 86: [66], 87: [47, 94], 88: [46, 88, 183, 218], 89: [23], 90: [203], 91: [87], 92: [38], 93: [129], 94: [100], 95: [91, 208], 96: [4], 97: [82], 98: [67, 69, 70], 99: [22, 33], 100: [180], 101: [197], 102: [92], 103: [182], 104: [181], 105: [212], 106: [147], 107: [5, 99], 108: [184], 109: [77], 110: [202], 111: [194], 112: [44, 171], 113: [196], 114: [200], 115: [30], 116: [72, 76], 117: [49]}

class_taxonomy_parent_lv2={0: [3, 11, 13, 24, 37, 68, 73, 74, 75, 80, 98, 124, 125, 131, 132, 133, 134, 135, 136, 138, 140, 141, 144, 172, 174, 192, 198, 204, 205, 209, 219], 1: [7, 10, 16, 17, 18, 19, 20, 50, 52, 85, 90, 123, 130, 139, 146, 150, 173, 176, 178, 185, 187, 193, 195, 199, 201, 207, 217, 220], 2: [0, 1, 2, 6, 8, 9, 12, 14, 25, 26, 27, 28, 29, 31, 34, 36, 39, 40, 41, 42, 43, 45, 48, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 71, 78, 79, 81, 83, 84, 89, 93, 95, 96, 97, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 137, 142, 143, 145, 148, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 165, 167, 169, 175, 177, 186, 188, 189, 191, 206, 210, 211, 213, 214, 216], 3: [4, 5, 15, 21, 22, 23, 30, 32, 33, 35, 38, 44, 46, 47, 49, 51, 66, 67, 69, 70, 72, 76, 77, 82, 86, 87, 88, 91, 92, 94, 99, 100, 126, 127, 128, 129, 147, 149, 163, 164, 166, 168, 170, 171, 179, 180, 181, 182, 183, 184, 190, 194, 196, 197, 200, 202, 203, 208, 212, 215, 218]}

num_of_data_in_parent_lv3={0: 117, 1: 47, 2: 5325, 3: 754, 4: 67, 5: 816, 6: 419, 7: 948, 8: 635, 9: 185, 10: 75, 11: 144, 12: 110, 13: 72, 14: 258, 15: 109, 16: 94, 17: 97, 18: 79, 19: 138, 20: 238, 21: 77, 22: 282, 23: 107, 24: 75, 25: 94, 26: 7, 27: 160, 28: 44, 29: 120, 30: 74, 31: 270, 32: 2169, 33: 2211, 34: 103, 35: 83, 36: 146, 37: 60, 38: 303, 39: 761, 40: 57, 41: 111, 42: 57, 43: 495, 44: 74, 45: 1071, 46: 455, 47: 2503, 48: 107, 49: 93, 50: 6136, 51: 179, 52: 11752, 53: 2031, 54: 56, 55: 47, 56: 356, 57: 113, 58: 1182, 59: 101, 60: 118, 61: 411, 62: 415, 63: 253, 64: 1451, 65: 131, 66: 49, 67: 51, 68: 1456, 69: 215, 70: 72, 71: 2713, 72: 50, 73: 169, 74: 620, 75: 352, 76: 52, 77: 1537, 78: 205, 79: 88, 80: 576, 81: 58, 82: 78, 83: 141, 84: 66, 85: 70, 86: 180, 87: 163, 88: 975, 89: 123, 90: 247, 91: 447, 92: 154, 93: 1722, 94: 210, 95: 595, 96: 130, 97: 962, 98: 535, 99: 133, 100: 571, 101: 266, 102: 254, 103: 497, 104: 77, 105: 165, 106: 622, 107: 489, 108: 537, 109: 96, 110: 82, 111: 131, 112: 353, 113: 159, 114: 166, 115: 113, 116: 345, 117: 50}

num_of_data_in_parent_lv2={0: 9972, 1: 7140, 2: 35072, 3: 15611}

final={0: [3, 7, 4, 14, 9, 1, 12, 0, 5, 2, 6, 8, 13, 11, 10], 1: [19, 32, 22, 27, 24, 20, 26, 33, 16, 18, 28, 34, 37, 17, 35, 23, 15, 30, 36, 25, 21, 31, 29, 38], 2: [53, 50, 52, 39, 44, 43, 55, 51, 45, 71, 61, 58, 68, 67, 47, 65, 41, 57, 42, 56, 64, 60, 63, 49, 62, 59, 69, 40, 54, 70, 48, 66, 46], 3: [96, 107, 77, 78, 99, 89, 115, 80, 73, 92, 112, 88, 87, 117, 72, 86, 98, 116, 109, 97, 81, 91, 95, 102, 94, 93, 106, 79, 74, 75, 82, 84, 85, 100, 104, 103, 108, 83, 111, 113, 101, 114, 110, 90, 105, 76]}

def generate_orth(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    col_norms = torch.norm(orth_vec, dim=0, keepdim=True)  
    orth_vec_normalized = orth_vec / col_norms  
    return orth_vec.T
def generate_new_ort_mat(old_orth_vec,feat_in, num_classes):
    old_orth_vec, _ = np.linalg.qr(old_orth_vec)
    new_columns = np.random.randn(feat_in, num_classes)

    for i in range(num_classes):
        for j in range(old_orth_vec.shape[1]):
            new_columns[:, i] -= np.dot(old_orth_vec[:, j], new_columns[:, i]) * old_orth_vec[:, j]
        for j in range(i):
            new_columns[:, i] -= np.dot(new_columns[:, j], new_columns[:, i]) * new_columns[:, j]

        norm = np.linalg.norm(new_columns[:, i])
        if norm > 1e-10:  
            new_columns[:, i] /= norm
    new_orth_vec = np.hstack((old_orth_vec, new_columns))
    new_orth_vec = torch.tensor(new_orth_vec).float()
    new_num_classes = old_orth_vec.shape[1] + num_classes
    assert torch.allclose(torch.matmul(new_orth_vec.T, new_orth_vec), torch.eye(new_num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(new_orth_vec.T, new_orth_vec) - torch.eye(new_num_classes))))
    col_norms = torch.norm(new_orth_vec, dim=0, keepdim=True)  
    orth_vec_normalized = new_orth_vec / col_norms  
    return new_orth_vec
# a=generate_orth(256,4).T
# l2_norms = torch.norm(a.T, dim=1)

# print("L2 Norms of Vectors:", l2_norms)
# b=generate_new_ort_mat(a,256,6)

# l2_norms = torch.norm(b.T, dim=1)

# print("L2 Norms of Vectors:", l2_norms)
# print(a)
# print(b)

def generate_GOF(orth_vec, level):
    if level==2:
        target_norms = torch.tensor([num_of_data_in_parent_lv2[j] for j in range(4)]).float()
        GOF = orth_vec.T * target_norms
    if level==3:
        target_norms = torch.tensor([num_of_data_in_parent_lv3[j] for j in range(118)]).float()
        GOF = orth_vec.T * target_norms
    return GOF.T
    

    
#abc
class ProxyNCA(torch.nn.Module):
    def __init__(self, 
        nb_classes,
        sz_embedding,
        smoothing_const = 0.1,
        scaling_x = 1,
        scaling_p = 3,
        level = 2
    ):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        
        if level == 3:
            self.proxies = torch.nn.Parameter(layer_3, requires_grad=False)
            print("level", level)
        if level == 2:
            self.proxies = torch.nn.Parameter(layer_2, requires_grad=False)
            print("level", level)

        #self.proxies = Parameter(torch.randn(nb_classes, sz_embedding) / 8)
        
        #self.proxies = Parameter(generate_GOF(generate_orth(sz_embedding, nb_classes), level))   
        
        #print(self.proxies.shape()) #torch.Size([8, 300, 221])
        # Set requires_grad to False
        #self.proxies.requires_grad = False
        
        
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p

    def forward(self, X, T):
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.scaling_x
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        return loss.mean()
    

def classify_to_parent_ids_lv2(id):
    for i, j in enumerate (class_taxonomy_parent_lv2):
        if id in class_taxonomy_parent_lv2[i]:
            return j
def classify_to_parent_ids_lv3(id):
    for i, j in enumerate (class_taxonomy_parent_lv3):
        if id in class_taxonomy_parent_lv3[i]:
            return j
    
class_to_parent_lv2 = {3: 0, 11: 0, 13: 0, 24: 0, 37: 0, 68: 0, 73: 0, 74: 0, 75: 0, 80: 0, 98: 0, 124: 0, 125: 0, 131: 0, 132: 0, 133: 0, 134: 0, 135: 0, 136: 0, 138: 0, 140: 0, 141: 0, 144: 0, 172: 0, 174: 0, 192: 0, 198: 0, 204: 0, 205: 0, 209: 0, 219: 0, 7: 1, 10: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 50: 1, 52: 1, 85: 1, 90: 1, 123: 1, 130: 1, 139: 1, 146: 1, 150: 1, 173: 1, 176: 1, 178: 1, 185: 1, 187: 1, 193: 1, 195: 1, 199: 1, 201: 1, 207: 1, 217: 1, 220: 1, 0: 2, 1: 2, 2: 2, 6: 2, 8: 2, 9: 2, 12: 2, 14: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 31: 2, 34: 2, 36: 2, 39: 2, 40: 2, 41: 2, 42: 2, 43: 2, 45: 2, 48: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2, 64: 2, 65: 2, 71: 2, 78: 2, 79: 2, 81: 2, 83: 2, 84: 2, 89: 2, 93: 2, 95: 2, 96: 2, 97: 2, 101: 2, 102: 2, 103: 2, 104: 2, 105: 2, 106: 2, 107: 2, 108: 2, 109: 2, 110: 2, 111: 2, 112: 2, 113: 2, 114: 2, 115: 2, 116: 2, 117: 2, 118: 2, 119: 2, 120: 2, 121: 2, 122: 2, 137: 2, 142: 2, 143: 2, 145: 2, 148: 2, 151: 2, 152: 2, 153: 2, 154: 2, 155: 2, 156: 2, 157: 2, 158: 2, 159: 2, 160: 2, 161: 2, 162: 2, 165: 2, 167: 2, 169: 2, 175: 2, 177: 2, 186: 2, 188: 2, 189: 2, 191: 2, 206: 2, 210: 2, 211: 2, 213: 2, 214: 2, 216: 2, 4: 3, 5: 3, 15: 3, 21: 3, 22: 3, 23: 3, 30: 3, 32: 3, 33: 3, 35: 3, 38: 3, 44: 3, 46: 3, 47: 3, 49: 3, 51: 3, 66: 3, 67: 3, 69: 3, 70: 3, 72: 3, 76: 3, 77: 3, 82: 3, 86: 3, 87: 3, 88: 3, 91: 3, 92: 3, 94: 3, 99: 3, 100: 3, 126: 3, 127: 3, 128: 3, 129: 3, 147: 3, 149: 3, 163: 3, 164: 3, 166: 3, 168: 3, 170: 3, 171: 3, 179: 3, 180: 3, 181: 3, 182: 3, 183: 3, 184: 3, 190: 3, 194: 3, 196: 3, 197: 3, 200: 3, 202: 3, 203: 3, 208: 3, 212: 3, 215: 3, 218: 3}

class_to_parent_lv3 = {138: 0, 80: 1, 144: 2, 204: 2, 205: 2, 3: 3, 24: 4, 140: 5, 141: 5, 172: 6, 174: 6, 11: 7, 13: 7, 37: 7, 68: 7, 98: 7, 124: 7, 131: 7, 132: 7, 133: 7, 134: 7, 135: 7, 192: 8, 75: 9, 125: 9, 219: 10, 209: 11, 136: 12, 198: 13, 73: 14, 74: 14, 187: 15, 130: 16, 176: 17, 139: 18, 7: 19, 85: 20, 201: 21, 16: 22, 17: 22, 18: 22, 19: 22, 20: 22, 185: 23, 52: 24, 199: 25, 90: 26, 50: 27, 146: 28, 217: 29, 193: 30, 207: 31, 10: 32, 123: 33, 150: 34, 178: 35, 195: 36, 173: 37, 220: 38, 26: 39, 83: 39, 143: 39, 191: 39, 188: 40, 145: 41, 167: 41, 151: 42, 28: 43, 84: 43, 89: 43, 152: 43, 153: 43, 154: 43, 155: 43, 156: 43, 27: 44, 78: 45, 79: 45, 216: 45, 214: 46, 137: 47, 211: 47, 210: 48, 165: 49, 6: 50, 12: 50, 31: 50, 36: 50, 40: 50, 48: 50, 53: 50, 54: 50, 55: 50, 56: 50, 57: 50, 58: 50, 59: 50, 60: 50, 61: 50, 62: 50, 63: 50, 64: 50, 65: 50, 41: 51, 42: 51, 43: 51, 8: 52, 9: 52, 14: 52, 25: 52, 29: 52, 39: 52, 45: 52, 71: 52, 81: 52, 93: 52, 101: 52, 102: 52, 103: 52, 104: 52, 105: 52, 106: 52, 107: 52, 108: 52, 109: 52, 110: 52, 111: 52, 112: 52, 113: 52, 114: 52, 115: 52, 116: 52, 121: 52, 0: 53, 1: 53, 2: 53, 189: 54, 34: 55, 157: 56, 148: 57, 97: 58, 175: 59, 160: 60, 96: 61, 177: 61, 169: 62, 161: 63, 162: 63, 159: 64, 142: 65, 213: 66, 122: 67, 117: 68, 118: 68, 119: 68, 120: 68, 158: 68, 186: 69, 206: 70, 95: 71, 51: 72, 35: 73, 163: 74, 164: 75, 215: 76, 15: 77, 166: 77, 21: 78, 149: 79, 32: 80, 126: 80, 127: 80, 128: 80, 86: 81, 168: 82, 190: 83, 170: 84, 179: 85, 66: 86, 47: 87, 94: 87, 46: 88, 88: 88, 183: 88, 218: 88, 23: 89, 203: 90, 87: 91, 38: 92, 129: 93, 100: 94, 91: 95, 208: 95, 4: 96, 82: 97, 67: 98, 69: 98, 70: 98, 22: 99, 33: 99, 180: 100, 197: 101, 92: 102, 182: 103, 181: 104, 212: 105, 147: 106, 5: 107, 99: 107, 184: 108, 77: 109, 202: 110, 194: 111, 44: 112, 171: 112, 196: 113, 200: 114, 30: 115, 72: 116, 76: 116, 49: 117}

def map_2D_to_parent(tensor, class_map):
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            class_id = tensor[i, j].item()
            if class_id != 221 and class_id in class_map:
                tensor[i, j] = class_map[class_id]
            if class_map == "class_to_parent_lv2":
                if class_id == 221:
                    tensor[i, j] = 4
            if class_map == "class_to_parent_lv3":
                if class_id == 221:
                    tensor[i, j] = 118
    return tensor

def map_1D_to_parent(tensor, class_map):
    mapped_tensor = tensor.clone()

    for i in range(mapped_tensor.shape[0]):
        class_id = mapped_tensor[i].item()
        if class_id != 221 and class_id in class_map:
            parent_class = class_map[class_id]
            mapped_tensor[i] = parent_class
        if class_map == "class_to_parent_lv2":
            if class_id == 221:
                mapped_tensor[i] = 4
        if class_map == "class_to_parent_lv3":
            if class_id == 221:
                mapped_tensor[i] = 118

    return mapped_tensor

     
def next_layer_matrix (old_matrix, taxonomy):
    new_matrix = np.empty((0, 256))
    for i, j in enumerate(taxonomy):
        a = []
        a = taxonomy[j]
        
        x = []
        for k in a:
            x.append(old_matrix[k])
        x = np.array(x)
        mean_vector = np.mean(x, axis = 0)
        new_matrix = np.vstack([new_matrix, mean_vector])
    return new_matrix
    
layer_3 = generate_orth(256, 118)
layer_2 = next_layer_matrix (layer_3, final)

np.savetxt('layer_3.txt', layer_3, fmt='%.6f')  # Adjust format as needed
np.savetxt('layer_2.txt', layer_2, fmt='%.6f')  # Adjust format as needed

layer_2 = torch.tensor(layer_2).float().cuda()
layer_3 = torch.tensor(layer_3).float().cuda()



print("Layer matrices saved to layer_3.txt and layer_2.txt")
    
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 

    
    
    
    
    
import numpy as np


@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma
        
        
        
        self.proxy_lv3 = ProxyNCA(118, 256, level = 3).cuda()
        
        self.proxy_lv2 = ProxyNCA(4, 256, level = 2).cuda()

        #self.proxy = Proxy_Anchor(80, 256).cuda()


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1].float()
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        #fixing
        #print("target", len(targets))
        #print(targets)
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        
        
        #print("src_logits = outputs['pred_logits']",src_logits)
        
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        
        
        target_parent_lv2 = torch.tensor([classify_to_parent_ids_lv2(id.item()) for id in target_classes_o])
        target_parent_lv3 = torch.tensor([classify_to_parent_ids_lv3(id.item()) for id in target_classes_o])
        #print("target_classes_o",target_classes_o)
        #print("target_parent_lv2",target_parent_lv2)
#         print(target_classes_o)
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        
        #print(src_logits.shape)

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}
    #abc
    
    def loss_proxy(self, outputs, targets, indices, num_boxes, log=True):
        #fixing
        #assert 'origin_logits_l6' in outputs
#         for i, j in enumerate(outputs):
#             print(j)
        src_logits = outputs['origin_logits_l6'].float()
        a = 0

        if src_logits.shape != torch.Size([8, 300, 221]):
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            #print("target_classes_o",target_classes_o.shape)
            # Create target_classes with correct size and device
            target_classes = torch.full(
                src_logits.shape[:2], 
                fill_value=len(class_taxonomy_parent_lv3), 
                dtype=torch.int64, 
                device=src_logits.device
            )
            target_classes[idx] = target_classes_o

            target_classes = map_2D_to_parent(target_classes, class_to_parent_lv3)
            target = F.one_hot(target_classes, num_classes=len(class_taxonomy_parent_lv3) + 1)[..., :-1].float()

            # Compute loss
            for i in range(src_logits.shape[0]):
                a += self.proxy_lv3(src_logits[i], target[i]) / max(num_boxes, 1e-6)
        
        
        src_logits_lv2 = outputs['origin_logits_l5'].float()
        
        if src_logits_lv2.shape != torch.Size([8, 300, 221]):
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            #print("target_classes_o",target_classes_o.shape)
            # Create target_classes with correct size and device
            target_classes = torch.full(
                src_logits_lv2.shape[:2], 
                fill_value=len(class_taxonomy_parent_lv2), 
                dtype=torch.int64, 
                device=src_logits_lv2.device
            )
            target_classes[idx] = target_classes_o

            target_classes = map_2D_to_parent(target_classes, class_to_parent_lv2)
            target = F.one_hot(target_classes, num_classes=len(class_taxonomy_parent_lv2) + 1)[..., :-1].float()
            # Compute loss
            for i in range(src_logits_lv2.shape[0]):
                a += self.proxy_lv2(src_logits_lv2[i], target[i]) / max(num_boxes, 1e-6)
        

        loss = a
        return {'loss_proxy': loss}

    

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'proxy':self.loss_proxy,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        #fixing
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    loss = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
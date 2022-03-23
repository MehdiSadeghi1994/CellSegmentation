import numpy as np

SMOOTH = 1e-6

def dice_coefficients(label1, label2, labels=None):
        if labels is None:
            labels = np.unique(np.hstack((np.unique(label1), np.unique(label2))))
        dice_coefs = []
        for label in labels:
            match1 = (label1 == label)
            match2 = (label2 == label)
            denominator = 0.5 * (np.sum(match1.astype(np.float)) + np.sum(match2.astype(np.float)))
            numerator = np.sum(np.logical_and(match1, match2).astype(np.float))
            if denominator == 0:
                dice_coefs.append(0.)
            else:
                dice_coefs.append(numerator / denominator)
        return dice_coefs




def iou(outputs: np.array, labels: np.array):

        intersection_1 = (outputs & labels).sum((1, 2))
        union_1 = (outputs | labels).sum((1, 2))

        intersection_0 = (np.logical_not(outputs) & np.logical_not(labels)).sum((1,2))
        union_0 = (np.logical_not(outputs) | np.logical_not(labels)).sum((1, 2))
        
        iou_1 = (intersection_1 + SMOOTH) / (union_1 + SMOOTH)
        iou_0 = (intersection_0 + SMOOTH) / (union_0 + SMOOTH)

        iou = 0.5*(iou_1+iou_0)
        # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
        return iou 



def precision(outputs: np.array, labels: np.array):
        intersection = (outputs & labels).sum((1, 2))
        precision = (intersection + SMOOTH) / (np.sum(np.abs(outputs), axis=(1,2)) + SMOOTH) 
        return precision             


def recall(outputs: np.array, labels: np.array):
        intersection = (outputs & labels).sum((1, 2))
        recall = (intersection + SMOOTH) / (np.sum(np.abs(labels), axis=(1,2)) + SMOOTH) 
        return recall                                                            
    


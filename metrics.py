import torch
import cv2

#the extraction of classification values (tp, tn, fp, fn) is batch elem and channel wise executed
#it is handeled this way, to enable analysis what impact which channel has
#in order to achive this .sum(dim=2).sum(dim=2) is often used
#the first .sum(dim=2) reduces the tensor shape from (b, c, h, w) to (b, c, w)
#the second .sum(dim=2) reduces the tensor shape from (b, c, w) to (b, c)

class Metric:
    def __init__(self, use_cuda=False):
        self.set_device(use_cuda)
        
    def set_device(self, use_cuda):
        if use_cuda:
            self.use_device = self.use_cuda
        else:
            self.use_device = self.use_cpu
    def use_cuda(self, pred, mask):
        return pred.cuda(), mask.cuda()
    def use_cpu(self, pred, mask):
        return pred.cpu(), mask.cpu()

    def init(self, pred, mask):
        if pred is None or mask is None:
            return
        assert pred.shape == mask.shape, "got pred shape {} instead of mask shape {}".format(pred.shape, mask.shape)

        #detach tensors
        pred = pred.detach().cpu().clone().round()
        mask = mask.detach().cpu().clone().round()

        self.extract_classification_values(pred, mask)
        
    def extract_classification_values(self, pred, mask):
        self.calc_false_values(pred, mask, (pred - mask).abs())

        self.calc_true_values(mask)

    def calc_false_values(self, pred, mask, abs_diff):
        #gets values that are only present in prediction
        self.fp = (abs_diff * pred).sum(dim=2).sum(dim=2)
        #gets values that are only present in mask
        self.fn = (abs_diff * mask).sum(dim=2).sum(dim=2)

    def calc_true_values(self, mask):
        self.tp = mask.sum(dim=2).sum(dim=2) - self.fn
        #for ThetaMetric and its subclasses the amount of tp elements in prediction and mask can be different
        #here the tp amount in mask is used because the prediction shall be evaluated relative to the mask and not the other way around

        self.tn = (torch.ones(mask.shape)).sum(dim=2).sum(dim=2) - self.tp - self.fp - self.fn
        #an alternative tn calculation:
        #self.tn = (1. - mask).sum(dim=2).sum(dim=2) - self.fp
        #using this formula with ThetaMetric and its subclasses the following condition could become true:
        #|tp elems| + |tn elems| + |fp elems| + |fn elems| != |mask elems|
        #which could result in metrics above 1
        #that is due to the possibility of different tp amounts in prediction and mask

    def get_classification_values(self, pred=None, mask=None):
        self.init(pred, mask)
        return self.tp, self.tn, self.fp, self.fn
    
    def dice(self, pred=None, mask=None):
        return self.f_measure(1, pred, mask)
    def jaccard(self, pred=None, mask=None):
        self.init(pred, mask)
        return (self.tp / (self.tp + self.fp + self.fn)).mean()
    def precision(self, pred=None, mask=None):
        self.init(pred, mask)
        return (self.tp / (self.tp + self.fp)).mean()
    def recall(self, pred=None, mask=None):
        self.init(pred, mask)
        return (self.tp / (self.tp + self.fn)).mean()
    def accuracy(self, pred=None, mask=None):
        self.init(pred, mask)
        return ((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)).mean()
    def f_measure(self, beta, pred=None, mask=None):
        self.init(pred, mask)
        return (((beta ** 2 + 1) * self.tp) / ((beta ** 2 + 1) * self.tp + (beta ** 2) * self.fn + self.fp)).mean()

class DistanceMetric(Metric):
    def __init__(self, use_cuda=False):
        super().__init__(use_cuda)
    
    def extract_classification_values(self, pred, mask):
        weighted_pred, weighted_mask = self.get_weighted_distances(pred, mask)

        self.calc_false_values(weighted_pred, weighted_mask, (pred - mask).abs())

        self.calc_true_values(mask)

    def get_weighted_distances(self, pred, mask):
        p_shape = pred.shape
        assert len(p_shape) == 4

        #invert mask and pred
        inv_pred = (1. - pred).type(torch.uint8).cpu().numpy()
        inv_mask = (1. - mask).type(torch.uint8).cpu().numpy()

        #init distance maps
        pred_distance_heat = torch.zeros(p_shape)
        mask_distance_heat = torch.zeros(p_shape)
        #get distance maps batch elem and channel wise
        for b in range(p_shape[0]):
            for c in range(p_shape[1]):
                pred_distance_heat[b,c] = torch.tensor(cv2.distanceTransform(inv_pred[b,c], cv2.DIST_L2, 5))
                mask_distance_heat[b,c] = torch.tensor(cv2.distanceTransform(inv_mask[b,c], cv2.DIST_L2, 5))

        #get distance of prediction pixels to mask pixels
        weighted_pred = pred * mask_distance_heat

        #get distance of mask pixels to prediction pixels
        weighted_mask = mask * pred_distance_heat

        return weighted_pred, weighted_mask

class ThetaMetric(DistanceMetric):
    def __init__(self, theta, use_cuda=False):
        super().__init__(use_cuda)
        self.theta = theta
    def extract_classification_values(self, pred, mask):
        weighted_pred, weighted_mask = self.get_weighted_distances(pred, mask)

        #pred weight > theta: false positive
        #get bool mask of fp elems
        fp_mask = weighted_pred > self.theta

        #mask weight > theta: false negative
        #get bool mask of fn elems
        fn_mask = weighted_mask > self.theta

        #count elems in fp_mask and fn_mask
        self.fp = fp_mask.sum(dim=2).sum(dim=2)
        self.fn = fn_mask.sum(dim=2).sum(dim=2)

        self.calc_true_values(mask)

        return weighted_pred, weighted_mask, fp_mask, fn_mask

class DistanceThetaMetric(ThetaMetric):
    def __init__(self, theta, use_cuda=False):
        super().__init__(theta, use_cuda)
    def extract_classification_values(self, pred, mask):
        theta = self.theta

        weighted_pred, weighted_mask, fp_mask, fn_mask = super().extract_classification_values(pred, mask)

        #extract fp elems, shift with theta, sum elems up
        self.fp = (weighted_pred * fp_mask - theta * fp_mask).sum(dim=2).sum(dim=2)

        #extract fn elems, shift with theta, sum elems up
        self.fn = (weighted_mask * fn_mask - theta * fn_mask).sum(dim=2).sum(dim=2)

class BorderMetric(Metric):
    def __init__(self, border_extractor, border_thicness=1, metric_object=None, use_cuda=False):
        super().__init__(use_cuda)
        if metric_object is not None:
            self.metric_object = metric_object
            self.super_extractor = self.metric_object.get_classification_values
            self.metric_object.set_device(use_cuda)
        else:
            self.super_extractor = self.get_classification_values
        self.border_extractor = border_extractor
        self.border_thicness = border_thicness

    def extract_classification_values(self, pred, mask):
        #extract borders
        pred = self.border_extractor(pred, self.border_thicness)
        mask = self.border_extractor(mask, self.border_thicness)

        #run classification value extraction in metric object if present or self if not
        self.tp, self.tn, self.fp, self.fn = self.super_extractor(pred, mask)
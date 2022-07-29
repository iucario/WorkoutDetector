# BSN: Boundary Sensitive Network for Temporal Action Proposal Generation
# http://arxiv.org/abs/1806.02964
# https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


def bi_loss(scores: Tensor, anchors: Tensor, threshold: float) -> Tuple[Tensor, int]:
    eps = 1e-6
    scores = scores.view(-1).cuda()
    anchors = anchors.contiguous().view(-1)

    pmask = (scores > threshold).float().cuda()
    num_positive = torch.sum(pmask)
    num_entries = len(scores)
    ratio = num_entries / num_positive

    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = coef_0 * (ratio - 1)
    loss = coef_1 * pmask * torch.log(anchors + eps) + coef_0 * (
        1.0 - pmask) * torch.log(1.0 - anchors + eps)
    loss = -torch.mean(loss)
    num_sample = [torch.sum(pmask), ratio]
    return loss, num_sample


def TEM_loss_calc(anchors_action, anchors_start, anchors_end, match_scores_action,
                  match_scores_start, match_scores_end, opt):

    loss_action, num_sample_action = bi_loss(match_scores_action, anchors_action, opt)
    loss_start_small, num_sample_start_small = bi_loss(match_scores_start, anchors_start,
                                                       opt)
    loss_end_small, num_sample_end_small = bi_loss(match_scores_end, anchors_end, opt)

    loss_dict = {
        "loss_action": loss_action,
        "num_sample_action": num_sample_action,
        "loss_start": loss_start_small,
        "num_sample_start": num_sample_start_small,
        "loss_end": loss_end_small,
        "num_sample_end": num_sample_end_small
    }
    #print loss_dict
    return loss_dict


def TEM_loss_function(y_action, y_start, y_end, TEM_output, opt):
    anchors_action = TEM_output[:, 0, :]
    anchors_start = TEM_output[:, 1, :]
    anchors_end = TEM_output[:, 2, :]
    loss_dict = TEM_loss_calc(anchors_action, anchors_start, anchors_end, y_action,
                              y_start, y_end, opt)

    cost = 2 * loss_dict["loss_action"] + loss_dict["loss_start"] + loss_dict["loss_end"]
    loss_dict["cost"] = cost
    return loss_dict


def PEM_loss_function(anchors_iou, match_iou, model, opt):
    match_iou = match_iou.cuda()
    anchors_iou = anchors_iou.view(-1)
    u_hmask = (match_iou > opt["pem_high_iou_thres"]).float()
    u_mmask = ((match_iou <= opt["pem_high_iou_thres"]) &
               (match_iou > opt["pem_low_iou_thres"])).float()
    u_lmask = (match_iou < opt["pem_low_iou_thres"]).float()

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = model.module.u_ratio_m * num_h / (num_m)
    r_m = torch.min(r_m, torch.Tensor([1.0]).cuda())[0]
    u_smmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).cuda()
    u_smmask = u_smmask * u_mmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = model.module.u_ratio_l * num_h / (num_l)
    r_l = torch.min(r_l, torch.Tensor([1.0]).cuda())[0]
    u_slmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).cuda()
    u_slmask = u_slmask * u_lmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    iou_weights = u_hmask + u_smmask + u_slmask
    iou_loss = F.smooth_l1_loss(anchors_iou, match_iou)
    iou_loss = torch.sum(iou_loss * iou_weights) / torch.sum(iou_weights)

    return iou_loss


def temporal_iou(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoU score between a groundtruth bbox and the proposals.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        list[float]: List of iou scores.
    """
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    union_len = len_anchors - inter_len + gt_max - gt_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def temporal_iop(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoP score between a groundtruth bbox and the proposals.

    Compute the IoP which is defined as the overlap ratio with
    groundtruth proportional to the duration of this proposal.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        list[float]: List of intersection over anchor scores.
    """
    len_anchors = np.array(proposal_max - proposal_min)
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def soft_nms(proposals, alpha, low_threshold, high_threshold, top_k):
    """Soft NMS for temporal proposals.

    Args:
        proposals (np.ndarray): Proposals generated by network.
        alpha (float): Alpha value of Gaussian decaying function.
        low_threshold (float): Low threshold for soft nms.
        high_threshold (float): High threshold for soft nms.
        top_k (int): Top k values to be considered.

    Returns:
        np.ndarray: The updated proposals.
    """
    proposals = proposals[proposals[:, -1].argsort()[::-1]]
    tstart = list(proposals[:, 0])
    tend = list(proposals[:, 1])
    tscore = list(proposals[:, -1])
    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 0 and len(rscore) <= top_k:
        max_index = np.argmax(tscore)
        max_width = tend[max_index] - tstart[max_index]
        iou_list = temporal_iou(tstart[max_index], tend[max_index], np.array(tstart),
                                np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list) / alpha)

        for idx, _ in enumerate(tscore):
            if idx != max_index:
                current_iou = iou_list[idx]
                if current_iou > low_threshold + (high_threshold -
                                                  low_threshold) * max_width:
                    tscore[idx] = tscore[idx] * iou_exp_list[idx]

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    rstart = np.array(rstart).reshape(-1, 1)
    rend = np.array(rend).reshape(-1, 1)
    rscore = np.array(rscore).reshape(-1, 1)
    new_proposals = np.concatenate((rstart, rend, rscore), axis=1)
    return new_proposals


class TEM(nn.Module):
    """Temporal Evaluation Model for Boundary Sensitive Network.

    Args:
        temporal_dim (int): Length of snippets. I guess.
        tem_feat_dim (int): Feature dimension.
        tem_hidden_dim (int): Hidden layer dimension.
        tem_match_threshold (float): Temporal evaluation match threshold.
        loss_cls (dict): Config for building loss.
            Default: ``dict(type='BinaryLogisticRegressionLoss')``.
        loss_weight (float): Weight term for action_loss. Default: 2.
        conv1_ratio (float): Ratio of conv1 layer output. Default: 1.0.
        conv2_ratio (float): Ratio of conv2 layer output. Default: 1.0.
        conv3_ratio (float): Ratio of conv3 layer output. Default: 0.01.
    """

    def __init__(self,
                 temporal_dim: int,
                 tem_feat_dim: int,
                 tem_hidden_dim: int = 512,
                 tem_match_threshold: float = 0.5,
                 boundary_ratio: float = 0.1,
                 loss_weight: float = 2,
                 conv1_ratio: float = 1,
                 conv2_ratio: float = 1,
                 conv3_ratio: float = 0.01):
        super().__init__()

        self.temporal_dim = temporal_dim
        self.boundary_ratio = boundary_ratio
        self.match_threshold = tem_match_threshold
        self.output_dim = 3
        self.loss_weight = loss_weight
        self.conv1_ratio = conv1_ratio
        self.conv2_ratio = conv2_ratio
        self.conv3_ratio = conv3_ratio

        self.conv1 = nn.Conv1d(in_channels=tem_feat_dim,
                               out_channels=tem_hidden_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1)
        self.conv2 = nn.Conv1d(in_channels=tem_hidden_dim,
                               out_channels=tem_hidden_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1)
        self.conv3 = nn.Conv1d(in_channels=tem_hidden_dim,
                               out_channels=self.output_dim,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.anchors_tmins, self.anchors_tmaxs = self._temporal_anchors()

    def _temporal_anchors(self, tmin_offset=0., tmax_offset=1.) -> Tuple[list, list]:
        """Generate temporal anchors.

        Args:
            tmin_offset (int): Offset for the minimum value of temporal anchor.
                Default: 0.
            tmax_offset (int): Offset for the maximum value of temporal anchor.
                Default: 1.

        Returns:
            tuple[List[float], List[float]]: The minimum and maximum values of temporal
                anchors.
        """
        temporal_gap = 1. / self.temporal_dim
        anchors_tmins = []
        anchors_tmaxs = []
        for i in range(self.temporal_dim):
            anchors_tmins.append(temporal_gap * (i + tmin_offset))
            anchors_tmaxs.append(temporal_gap * (i + tmax_offset))

        return anchors_tmins, anchors_tmaxs

    def _forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw features.
        Returns:
            torch.Tensor: probability scores of action, start and end.
                Shape: (N, 3).
        """
        x = F.relu(self.conv1_ratio * self.conv1(x))
        x = F.relu(self.conv2_ratio * self.conv2(x))
        x = torch.sigmoid(self.conv3_ratio * self.conv3(x))
        return x

    def train_step(self, raw_feature: Tensor, label_action: Tensor, label_start: Tensor,
                   label_end: Tensor, gt_bbox):
        """Define the computation performed at every call when training."""
        label_action, label_start, label_end = (self.generate_labels(gt_bbox))
        device = raw_feature.device
        label_action = label_action.to(device)
        label_start = label_start.to(device)
        label_end = label_end.to(device)
        tem_output = self._forward(raw_feature)
        score_action = tem_output[:, 0, :]
        score_start = tem_output[:, 1, :]
        score_end = tem_output[:, 2, :]

        loss_action = bi_loss(score_action, label_action, self.match_threshold)
        loss_start_small = bi_loss(score_start, label_start, self.match_threshold)
        loss_end_small = bi_loss(score_end, label_end, self.match_threshold)
        loss_dict = {
            'loss_action': loss_action * self.loss_weight,
            'loss_start': loss_start_small,
            'loss_end': loss_end_small
        }

        return loss_dict

    def test_step(self, raw_feature, video_meta):
        """Define the computation performed at every call when testing."""
        tem_output = self._forward(raw_feature).cpu().numpy()
        batch_action = tem_output[:, 0, :]
        batch_start = tem_output[:, 1, :]
        batch_end = tem_output[:, 2, :]

        video_meta_list = [dict(x) for x in video_meta]

        video_results = []

        for batch_idx, _ in enumerate(batch_action):
            video_name = video_meta_list[batch_idx]['video_name']
            video_action = batch_action[batch_idx]
            video_start = batch_start[batch_idx]
            video_end = batch_end[batch_idx]
            video_result = np.stack((video_action, video_start, video_end,
                                     self.anchors_tmins, self.anchors_tmaxs),
                                    axis=1)
            video_results.append((video_name, video_result))
        return video_results

    def generate_labels(self, gt_bbox: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate training labels.
        Start and end labels are padded to calculate overlap between the prediction.
        
        Args:
            gt_bbox (np.ndarray): Ground truth bounding boxes. Shape: (N, 2). With
                item [start, end].
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Action, start and end labels. 
        """
        match_score_action_list = []
        match_score_start_list = []
        match_score_end_list = []
        for every_gt_bbox in gt_bbox:
            gt_tmins = every_gt_bbox[:, 0].cpu().numpy()
            gt_tmaxs = every_gt_bbox[:, 1].cpu().numpy()

            gt_lens = gt_tmaxs - gt_tmins
            gt_len_pad = np.maximum(1. / self.temporal_dim, self.boundary_ratio * gt_lens)

            gt_start_bboxs = np.stack(
                (gt_tmins - gt_len_pad / 2, gt_tmins + gt_len_pad / 2), axis=1)
            gt_end_bboxs = np.stack(
                (gt_tmaxs - gt_len_pad / 2, gt_tmaxs + gt_len_pad / 2), axis=1)

            match_score_action = []
            match_score_start = []
            match_score_end = []

            for anchor_tmin, anchor_tmax in zip(self.anchors_tmins, self.anchors_tmaxs):
                match_score_action.append(
                    np.max(temporal_iop(anchor_tmin, anchor_tmax, gt_tmins, gt_tmaxs)))
                match_score_start.append(
                    np.max(
                        temporal_iop(anchor_tmin, anchor_tmax, gt_start_bboxs[:, 0],
                                     gt_start_bboxs[:, 1])))
                match_score_end.append(
                    np.max(
                        temporal_iop(anchor_tmin, anchor_tmax, gt_end_bboxs[:, 0],
                                     gt_end_bboxs[:, 1])))
            match_score_action_list.append(match_score_action)
            match_score_start_list.append(match_score_start)
            match_score_end_list.append(match_score_end)
        match_score_action_list = torch.Tensor(match_score_action_list)
        match_score_start_list = torch.Tensor(match_score_start_list)
        match_score_end_list = torch.Tensor(match_score_end_list)
        return match_score_action_list, match_score_start_list, match_score_end_list


def post_processing(result, video_info, soft_nms_alpha, soft_nms_low_threshold,
                    soft_nms_high_threshold, post_process_top_k,
                    feature_extraction_interval):
    """Post process for temporal proposals generation.

    Args:
        result (np.ndarray): Proposals generated by network.
        video_info (dict): Meta data of video. Required keys are
            'duration_frame', 'duration_second'.
        soft_nms_alpha (float): Alpha value of Gaussian decaying function.
        soft_nms_low_threshold (float): Low threshold for soft nms.
        soft_nms_high_threshold (float): High threshold for soft nms.
        post_process_top_k (int): Top k values to be considered.
        feature_extraction_interval (int): Interval used in feature extraction.

    Returns:
        list[dict]: The updated proposals, e.g.
            [{'score': 0.9, 'segment': [0, 1]},
             {'score': 0.8, 'segment': [0, 2]},
            ...].
    """
    if len(result) > 1:
        result = soft_nms(result, soft_nms_alpha, soft_nms_low_threshold,
                          soft_nms_high_threshold, post_process_top_k)

    result = result[result[:, -1].argsort()[::-1]]
    video_duration = float(video_info['duration_frame'] // feature_extraction_interval *
                           feature_extraction_interval
                          ) / video_info['duration_frame'] * video_info['duration_second']
    proposal_list = []

    for j in range(min(post_process_top_k, len(result))):
        proposal = {}
        proposal['score'] = float(result[j, -1])
        proposal['segment'] = [
            max(0, result[j, 0]) * video_duration,
            min(1, result[j, 1]) * video_duration
        ]
        proposal_list.append(proposal)
    return proposal_list


class PEM(nn.Module):
    """Proposals Evaluation Model for Boundary Sensitive Network.

    Please refer `BSN: Boundary Sensitive Network for Temporal Action
    Proposal Generation <http://arxiv.org/abs/1806.02964>`_.

    Code reference
    https://github.com/wzmsltw/BSN-boundary-sensitive-network

    Args:
        pem_feat_dim (int): Feature dimension.
        pem_hidden_dim (int): Hidden layer dimension.
        pem_u_ratio_m (float): Ratio for medium score proprosals to balance
            data.
        pem_u_ratio_l (float): Ratio for low score proprosals to balance data.
        pem_high_temporal_iou_threshold (float): High IoU threshold.
        pem_low_temporal_iou_threshold (float): Low IoU threshold.
        soft_nms_alpha (float): Soft NMS alpha.
        soft_nms_low_threshold (float): Soft NMS low threshold.
        soft_nms_high_threshold (float): Soft NMS high threshold.
        post_process_top_k (int): Top k proposals in post process.
        feature_extraction_interval (int):
            Interval used in feature extraction. Default: 16.
        fc1_ratio (float): Ratio for fc1 layer output. Default: 0.1.
        fc2_ratio (float): Ratio for fc2 layer output. Default: 0.1.
        output_dim (int): Output dimension. Default: 1.
    """

    def __init__(self,
                 pem_feat_dim,
                 pem_hidden_dim,
                 pem_u_ratio_m,
                 pem_u_ratio_l,
                 pem_high_temporal_iou_threshold,
                 pem_low_temporal_iou_threshold,
                 soft_nms_alpha,
                 soft_nms_low_threshold,
                 soft_nms_high_threshold,
                 post_process_top_k,
                 feature_extraction_interval=16,
                 fc1_ratio=0.1,
                 fc2_ratio=0.1,
                 output_dim=1):
        super().__init__()

        self.feat_dim = pem_feat_dim
        self.hidden_dim = pem_hidden_dim
        self.u_ratio_m = pem_u_ratio_m
        self.u_ratio_l = pem_u_ratio_l
        self.pem_high_temporal_iou_threshold = pem_high_temporal_iou_threshold
        self.pem_low_temporal_iou_threshold = pem_low_temporal_iou_threshold
        self.soft_nms_alpha = soft_nms_alpha
        self.soft_nms_low_threshold = soft_nms_low_threshold
        self.soft_nms_high_threshold = soft_nms_high_threshold
        self.post_process_top_k = post_process_top_k
        self.feature_extraction_interval = feature_extraction_interval
        self.fc1_ratio = fc1_ratio
        self.fc2_ratio = fc2_ratio
        self.output_dim = output_dim

        self.fc1 = nn.Linear(in_features=self.feat_dim,
                             out_features=self.hidden_dim,
                             bias=True)
        self.fc2 = nn.Linear(in_features=self.hidden_dim,
                             out_features=self.output_dim,
                             bias=True)

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = torch.cat(list(x))
        x = F.relu(self.fc1_ratio * self.fc1(x))
        x = torch.sigmoid(self.fc2_ratio * self.fc2(x))
        return x

    def forward_train(self, bsp_feature, reference_temporal_iou):
        """Define the computation performed at every call when training."""
        pem_output = self._forward(bsp_feature)
        reference_temporal_iou = torch.cat(list(reference_temporal_iou))
        device = pem_output.device
        reference_temporal_iou = reference_temporal_iou.to(device)

        anchors_temporal_iou = pem_output.view(-1)
        u_hmask = (reference_temporal_iou > self.pem_high_temporal_iou_threshold).float()
        u_mmask = (
            (reference_temporal_iou <= self.pem_high_temporal_iou_threshold) &
            (reference_temporal_iou > self.pem_low_temporal_iou_threshold)).float()
        u_lmask = (reference_temporal_iou <= self.pem_low_temporal_iou_threshold).float()

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = self.u_ratio_m * num_h / (num_m)
        r_m = torch.min(r_m, torch.Tensor([1.0]).to(device))[0]
        u_smmask = torch.rand(u_hmask.size()[0], device=device)
        u_smmask = u_smmask * u_mmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = self.u_ratio_l * num_h / (num_l)
        r_l = torch.min(r_l, torch.Tensor([1.0]).to(device))[0]
        u_slmask = torch.rand(u_hmask.size()[0], device=device)
        u_slmask = u_slmask * u_lmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        temporal_iou_weights = u_hmask + u_smmask + u_slmask
        temporal_iou_loss = F.smooth_l1_loss(anchors_temporal_iou, reference_temporal_iou)
        temporal_iou_loss = torch.sum(
            temporal_iou_loss * temporal_iou_weights) / torch.sum(temporal_iou_weights)
        loss_dict = dict(temporal_iou_loss=temporal_iou_loss)

        return loss_dict

    def forward_test(self, bsp_feature, tmin, tmax, tmin_score, tmax_score, video_meta):
        """Define the computation performed at every call when testing."""
        pem_output = self._forward(bsp_feature).view(-1).cpu().numpy().reshape(-1, 1)

        tmin = tmin.view(-1).cpu().numpy().reshape(-1, 1)
        tmax = tmax.view(-1).cpu().numpy().reshape(-1, 1)
        tmin_score = tmin_score.view(-1).cpu().numpy().reshape(-1, 1)
        tmax_score = tmax_score.view(-1).cpu().numpy().reshape(-1, 1)
        score = np.array(pem_output * tmin_score * tmax_score).reshape(-1, 1)
        result = np.concatenate((tmin, tmax, tmin_score, tmax_score, pem_output, score),
                                axis=1)
        result = result.reshape(-1, 6)
        video_info = dict(video_meta[0])
        proposal_list = post_processing(result, video_info, self.soft_nms_alpha,
                                        self.soft_nms_low_threshold,
                                        self.soft_nms_high_threshold,
                                        self.post_process_top_k,
                                        self.feature_extraction_interval)
        output = [dict(video_name=video_info['video_name'], proposal_list=proposal_list)]
        return output

    def forward(self,
                bsp_feature,
                reference_temporal_iou=None,
                tmin=None,
                tmax=None,
                tmin_score=None,
                tmax_score=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(bsp_feature, reference_temporal_iou)

        return self.forward_test(bsp_feature, tmin, tmax, tmin_score, tmax_score,
                                 video_meta)

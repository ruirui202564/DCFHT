from skmultiflow.trees import HoeffdingTreeClassifier
import numpy as np
import time
from copy import deepcopy
from tqdm import tqdm
from findAttr import findAttrAll, updateAttr, delSilent, updateModelTree, updateModelTreeIdx, get_ord_indices, delAttrTree
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

class OCFHT_drift_new():
    def __init__(self,
                 classifier,
                 created_on,
                 drift_detection_method: BaseDriftDetector,
                 warning_detection_method: BaseDriftDetector,
                 is_background_learner: bool = False,
                 is_drift: bool = False):
        self.classifier = classifier
        self.created_on = created_on
        self.instances_seen = 0
        self.is_background_learner = is_background_learner
        self.is_drift = is_drift

        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method

        self.last_drift_on = 0
        self.last_warning_on = 0
        self.nb_drifts_detected = 0
        self.nb_warnings_detected = 0

        self.uni_fea = []
        self.attributeTree = []
        self.err_count = 0
        self.correct_cnt = 0
        self.correct_idx = []
        self.all_instances = 0

        self.drift_detection = None
        self.warning_detection = None
        self.background_learner = None
        self._use_drift_detector = False
        self._use_background_learner = False

        # Initialize drift and warning detectors
        if drift_detection_method is not None:
            self._use_drift_detector = True
            self.drift_detection = deepcopy(drift_detection_method)

        if warning_detection_method is not None:
            self._use_background_learner = True
            self.warning_detection = deepcopy(warning_detection_method)

    def reset(self, instances_seen):
        if self._use_background_learner and self.background_learner is not None:
            self.classifier = self.background_learner.classifier
            self.warning_detection = self.background_learner.warning_detection
            self.drift_detection = self.background_learner.drift_detection
            self.created_on = self.background_learner.created_on
            self.uni_fea = self.background_learner.uni_fea
            self.attributeTree = self.background_learner.attributeTree
            self.instances_seen = self.background_learner.instances_seen
            self.feature_sta = self.background_learner.feature_sta
            self.background_learner = None
        else:
            self.classifier.reset()
            self.created_on = instances_seen
            self.drift_detection.reset()

    def OCFHT_drift_OVFM(self, X, Y):
        num_sample = X.shape[0]

        # err_count = 0
        # correct_cnt = 0
        acc_all = []
        y_pred = []
        y_actual = []
        # correct_idx=[]
        # uni_fea = []
        all_ord_indices = get_ord_indices(X)

        start_time = time.time()
        for t in tqdm(range(num_sample)): 
            self.all_instances += 1
            x_t = X[t,:]
            y_t = Y[t]
            y_t = np.array([y_t])
            idx_t = np.where(~np.isnan(x_t))
            idx_t = idx_t[0]

            x_t_ = x_t
            x_t = x_t[idx_t]
            if y_t[0] == -1:
                y_t[0] = 0
            y_actual.append(y_t)            

            x_t = x_t.reshape(1,-1)
            pred_t, pred_result, idx_t_new, x_t_new = self.predict(x_t, y_t, idx_t, x_t_, all_ord_indices)
            y_pred.append(pred_t)
            acc_t = self.correct_cnt / (t+1)
            acc_all.append(acc_t)
            
            if self.is_drift:
                # self.drift_warn(x_t_new, y_t, idx_t_new, pred_result, acc_t, x_t_) # warning、drift
                self.drift_acc(x_t_new, y_t, idx_t_new, pred_result, acc_t, x_t_, all_ord_indices)
            else:
                self.partial_fit_back_acc(x_t_new, y_t, idx_t_new)

        end_Time = time.time()
        runtime = end_Time - start_time
        return self.classifier, self.err_count, self.correct_cnt, runtime, acc_all

    def predict(self, x_t, y_t, idx_t, x_t_, all_ord_indices):
        self.instances_seen += 1
        ord = []
        if self.instances_seen == 1:
            self.uni_fea = idx_t
            idx_t_new = idx_t
            x_t_new = x_t
        else:
            active = np.intersect1d(idx_t, self.uni_fea)
            if len(active) == 0:
                new = idx_t
            else:
                new = np.setdiff1d(idx_t, active)
            silent = np.setdiff1d(self.uni_fea, active)
            self.uni_fea = np.union1d(self.uni_fea, idx_t)
            if len(new) == 0:
                idx_t_new = idx_t
                x_t_new = x_t
            else:
                idx_t_new = np.concatenate((active, new))
                idx_t_new = idx_t_new.astype(np.int32)
                x_t_new = x_t_[idx_t_new]
                x_t_new = x_t_new.reshape(1, -1)
            
            if len(silent) > 0:
                self.classifier._tree_root = delSilent(self.classifier._tree_root, silent)
            self.classifier._tree_root = updateModelTreeIdx(self.classifier._tree_root, idx_t_new)

        for item in idx_t_new:
            if item in all_ord_indices:
                ord_id = np.where(idx_t_new == item)[0][0]
                ord.append(ord_id)
        self.classifier.nominal_attributes = ord

        pred_t = self.classifier.predict(x_t_new) 
        
        if y_t == pred_t:
            self.correct_cnt += 1
            pred_result = 0
            self.correct_idx.append(self.all_instances)
        else:
            self.err_count += 1
            pred_result = 1
        return pred_t, pred_result, idx_t_new, x_t_new

    def drift_warn(self, x_t_new, y_t, idx_t_new, pred_result, acc_t, x_t_):
        if self.instances_seen == 1:
            self.classifier.partial_fit(x_t_new, y_t)
            self.attributeTree = findAttrAll(self.classifier._active_leaf_node_cnt, self.classifier._tree_root, idx_t_new)
            self.classifier._tree_root = updateModelTree(self.classifier._tree_root, self.attributeTree.tree)
        else:
            self.classifier.partial_fit(x_t_new, y_t)
            attributeTree_new = findAttrAll(self.classifier._active_leaf_node_cnt, self.classifier._tree_root, idx_t_new)
            self.attributeTree = updateAttr(self.attributeTree, attributeTree_new)
            self.classifier._tree_root = updateModelTree(self.classifier._tree_root, self.attributeTree.tree)

            if self.background_learner:
                self.background_learner.classifier.nominal_attributes = self.classifier.nominal_attributes
                self.background_learner.partial_fit_back_warn(x_t_new, y_t, idx_t_new, x_t_)

            if self._use_drift_detector and not self.is_background_learner:
                if self._use_background_learner:
                    self.warning_detection.add_element(acc_t)
                    if self.warning_detection.detected_change():
                        self.last_warning_on = self.instances_seen
                        self.nb_warnings_detected += 1
                        background_learner = HoeffdingTreeClassifier()
                        self.background_learner = OCFHT_drift_new(background_learner,
                                                              self.all_instances,
                                                              self.drift_detection_method,
                                                              self.warning_detection_method,
                                                              True,
                                                              self.is_drift)
                        self.warning_detection.reset()

                # Update the drift detection
                self.drift_detection.add_element(acc_t)
                if self.drift_detection.detected_change():
                    self.last_drift_on = self.instances_seen
                    self.nb_drifts_detected += 1
                    self.reset(self.instances_seen)

    def partial_fit_back_warn(self, x_t, y_t, idx_t, x_t_):
        self.instances_seen += 1
        if self.instances_seen == 1:
            self.uni_fea = idx_t
            idx_t_new = idx_t
            x_t_new = x_t
            self.classifier.partial_fit(x_t_new, y_t)
            self.attributeTree = findAttrAll(self.classifier._active_leaf_node_cnt,
                                            self.classifier._tree_root,
                                            idx_t_new)
            self.classifier._tree_root = updateModelTree(self.classifier._tree_root,
                                                        self.attributeTree.tree)
        else:
            self.instances_seen += 1
            active = np.intersect1d(idx_t, self.uni_fea) 
            if len(active) == 0:
                new = idx_t
            else:
                new = np.setdiff1d(idx_t, active)
            silent = np.setdiff1d(self.uni_fea, active)
            self.uni_fea = np.union1d(self.uni_fea, idx_t)
            if len(new) == 0:
                idx_t_new = idx_t
                x_t_new = x_t
            else:
                idx_t_new = np.concatenate((active, new))
                x_t_new = x_t_[idx_t_new]
                x_t_new = x_t_new.reshape(1, -1)

            if len(silent) > 0:
                self.classifier._tree_root = delSilent(self.classifier._tree_root, silent)
            self.classifier._tree_root = updateModelTreeIdx(self.classifier._tree_root, idx_t_new)

            self.classifier.partial_fit(x_t_new, y_t)
            attributeTree_new = findAttrAll(self.classifier._active_leaf_node_cnt,
                                            self.classifier._tree_root,
                                            idx_t_new)
            self.attributeTree = updateAttr(self.attributeTree, attributeTree_new)
            self.classifier._tree_root = updateModelTree(self.classifier._tree_root, self.attributeTree.tree)

    def drift_acc(self, x_t_new, y_t, idx_t_new, pred_result, acc_t, x_t_, all_ord_indices): 
        if self.instances_seen == 1:
            self.classifier.partial_fit(x_t_new, y_t)
            self.attributeTree = findAttrAll(self.classifier._active_leaf_node_cnt, self.classifier._tree_root, idx_t_new)
            self.classifier._tree_root = updateModelTree(self.classifier._tree_root, self.attributeTree.tree)
        else:
            self.classifier.partial_fit(x_t_new, y_t)
            attributeTree_new = findAttrAll(self.classifier._active_leaf_node_cnt, self.classifier._tree_root, idx_t_new)
            self.attributeTree = updateAttr(self.attributeTree, attributeTree_new)
            self.classifier._tree_root = updateModelTree(self.classifier._tree_root, self.attributeTree.tree)

            if self.background_learner:
                pred_res_back_t, pred_res_back, idx_t1, x_t1 = self.background_learner.predict(x_t_new, y_t, idx_t_new, x_t_, all_ord_indices)
                # acc_back = self.background_learner.correct_cnt / self.background_learner.instances_seen
                acc_back = (self.background_learner.correct_cnt + self.correct_cnt) / self.all_instances
                self.background_learner.partial_fit_back_acc(x_t1, y_t, idx_t1)

            if self._use_drift_detector and not self.is_background_learner:
                if self._use_background_learner:
                    self.warning_detection.add_element(pred_result) # acc_t
                    if self.warning_detection.detected_change():
                        self.last_warning_on = self.all_instances
                        self.nb_warnings_detected += 1
                        background_learner = HoeffdingTreeClassifier()
                        self.background_learner = OCFHT_drift_new(background_learner,
                                                              self.all_instances,
                                                              self.drift_detection_method,
                                                              self.warning_detection_method,
                                                              True,
                                                              self.is_drift)
                        self.warning_detection.reset()

            if self.background_learner:
                if self.background_learner.instances_seen > 50 and acc_back > acc_t:
                    self.last_drift_on = self.all_instances
                    self.nb_drifts_detected += 1
                    self.reset(self.all_instances)

    def partial_fit_back_acc(self, x_t, y_t, idx_t):
        if self.instances_seen == 1: 
            self.classifier.partial_fit(x_t, y_t)
            self.attributeTree = findAttrAll(self.classifier._active_leaf_node_cnt,
                                             self.classifier._tree_root,
                                             idx_t)
            self.classifier._tree_root = updateModelTree(self.classifier._tree_root,
                                                         self.attributeTree.tree)
        else:
            self.classifier.partial_fit(x_t, y_t) # 如果split_idx不存在当前样本中，则这里的split_idx仍然为-1
            attributeTree_new = findAttrAll(self.classifier._active_leaf_node_cnt,self.classifier._tree_root, idx_t) # 如果split_idx不存在当前样本中，则这里的split_idx仍然为-1
            self.attributeTree = updateAttr(self.attributeTree, attributeTree_new)
            self.classifier._tree_root = updateModelTree(self.classifier._tree_root, self.attributeTree.tree)

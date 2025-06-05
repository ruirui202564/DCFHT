from skmultiflow.trees.nodes import LearningNodeNBAdaptive
from skmultiflow.trees.nodes import SplitNode
from skmultiflow.trees import HoeffdingTreeClassifier
import numpy as np
import copy

class overTree():
    def __init__(self, active_nodes_cnt, tree):
        self.active_nodes_cnt = active_nodes_cnt
        self.tree = tree

class parent():
    def __init__(self, split_attr, split_vanish):
        self.children = []
        self.split_attr = split_attr
        self.split_vanish = split_vanish

class leaf():
    def __init__(self, attr_observe):
        self.attribute = attr_observe

def findAttrAll(active_nodes_cnt, tree, idx_t):
    attr = findAttr(tree, idx_t)
    attrTree = overTree(active_nodes_cnt, attr)
    return attrTree

def findAttr(tree, idx_t):
    if isinstance(tree, SplitNode):
        idx = tree._split_test._att_idx
        if idx in range(len(idx_t)):
            attr = parent(idx_t[idx], 0)
        else:
            attr = parent(idx, 1)

        for i in range(len(tree._children)):
            new_child = findAttr(tree._children[i], idx_t)
            attr.children.append(new_child)
    else: 
        para_fea = {}
        attribute = tree._attribute_observers
        for key, value in attribute.items():
            new_key = idx_t[key]
            para_fea[new_key] = value
        attr = leaf(para_fea)

    return attr

def updateAttr(attrTree, attrTree_new):
    attrTree.active_nodes_cnt = attrTree_new.active_nodes_cnt
    if isinstance(attrTree.tree, leaf) & isinstance(attrTree_new.tree, parent):
        attrTree.tree = attrTree_new.tree
    elif isinstance(attrTree.tree, leaf) & isinstance(attrTree_new.tree, leaf):
        attrTree.tree.attribute.update(attrTree_new.tree.attribute)
    elif isinstance(attrTree.tree, parent) & isinstance(attrTree_new.tree, leaf):
        attrTree.tree = attrTree_new.tree
    else:
        attrTree.tree.split_vanish += attrTree_new.tree.split_vanish
        if attrTree.tree.split_vanish >= 1:
            para_fea = {}#findAllLeafAttr(attrTree.tree, {}) #
            attrTree.tree = leaf(para_fea)
            attrTree.active_nodes_cnt -= 1
        else:
            if attrTree_new.tree.split_attr >= 5:
                attrTree.tree.split_attr = attrTree_new.tree.split_attr
            attrTree_child = len(attrTree.tree.children)
            for i in range(len(attrTree_new.tree.children)):
                if i+1 > attrTree_child:
                    attrTree.tree.children.append(attrTree_new.tree.children[i])
                else:
                    attrTree.tree.children[i] = update_child(attrTree.tree.children[i], attrTree_new.tree.children[i])
    return attrTree

def update_child(tree, tree_new):
    if isinstance(tree, leaf) & isinstance(tree_new, parent):
        tree = tree_new
    elif isinstance(tree, leaf) & isinstance(tree_new, leaf):
        tree.attribute.update(tree_new.attribute)
    else:
        tree.split_vanish += tree_new.split_vanish
        if tree.split_vanish >= 1:
            para_fea = {}#findAllLeafAttr(tree, {}) #
            tree = leaf(para_fea)
        else:
            if tree_new.split_attr >= 0: 
                tree.split_attr = tree_new.split_attr
            attrTree_child = len(tree.children)
            for i in range(len(tree_new.children)):
                if i+1 > attrTree_child:
                    tree.children.append(tree_new.children[i])
                else:
                    tree.children[i] = update_child(tree.children[i], tree_new.children[i])
    return tree

def findAllLeafAttr(tree, attr):
    if isinstance(tree, leaf):
        attr.update(tree.attribute)
    else:
        for i in range(len(tree.children)):
            tree.children[i] = findAllLeafAttr(tree.children[i], attr)

    return attr

def delSilent(modelTree, silent):
    for i in range(len(silent)):
        modelTree = delSil(modelTree, silent[i])
    return modelTree

def delSil(modelTree, sil):
    if isinstance(modelTree, SplitNode):
        for i in range(len(modelTree._children)):
            modelTree._children[i] = delSil(modelTree._children[i], sil)
    else:
        if modelTree is not None and hasattr(modelTree, '_attribute_observers'):
            attribute = modelTree._attribute_observers
            if sil in attribute:
                del attribute[sil]
            modelTree._attribute_observers = attribute
    return modelTree

def updateModelTree(modelTree, attrTree):
    if isinstance(modelTree, SplitNode) & isinstance(attrTree, parent):
        modelTree._split_test._att_idx = int(attrTree.split_attr)
        for i in range(len(modelTree._children)):
            modelTree._children[i] = updateModelTree(modelTree._children[i], attrTree.children[i])
    elif isinstance(modelTree, SplitNode) & isinstance(attrTree, leaf):
        model = LearningNodeNBAdaptive({})
        model._attribute_observers = copy.deepcopy(attrTree.attribute)
        modelTree = model
    else: 
        modelTree._attribute_observers = attrTree.attribute
        new = {}
        sorted_keys = sorted(modelTree._attribute_observers.keys())
        for key in sorted_keys:
            new[key] = modelTree._attribute_observers[key]
        modelTree._attribute_observers = new

    return modelTree

def updateModelTreeIdx(modelTree, idx_t):
    if isinstance(modelTree, SplitNode):
        if modelTree._split_test._att_idx in idx_t:
            idx = modelTree._split_test._att_idx
            modelTree._split_test._att_idx = int(np.argwhere(idx_t == idx))
        else:
            modelTree._split_test._att_idx = -1
        for i in range(len(modelTree._children)):
            modelTree._children[i] = updateModelTreeIdx(modelTree._children[i], idx_t)
    else:
        if modelTree is not None and hasattr(modelTree, '_attribute_observers'):
            i = 0
            new = {}
            for key, value in modelTree._attribute_observers.items():
                new[i] = value
                i = i + 1
            modelTree._attribute_observers = new

    return modelTree

def get_ord_indices(X):
    max_ord=15
    indices = []
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) <= max_ord:
            indices.append(i)
    return indices

def delAttrTree(attrTree, delFea):
    for i in range(len(delFea)):
        attrTree = delAttr(attrTree, delFea[i])
    return attrTree

def delAttr(attrTree, fea):
    if isinstance(attrTree, parent):
        for i in range(len(attrTree.children)):
            attrTree.children[i] = delAttr(attrTree.children[i], fea)
    else:
        attribute = attrTree.attribute
        if fea in attribute:
            del attribute[fea]
        attrTree.attribute = attribute
    return attrTree

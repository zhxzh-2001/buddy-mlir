# ===- fuse_ops.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Construct op fusion pattern.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import PlaceholderOp, OpType,Op
from .. import DeviceType

# TODO: classify op type for op fusion
# OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType]
# OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.ConcatType]
# OP_TYPE_FUSABLE_BY_SPECIFIC_PASS = []
# ANCHOR_OP_TYPE = []

from ..type import TensorDType
from torch.fx.node import Node as torchnode
from torch.fx.immutable_collections import immutable_list
from ..operation import OutputOp, PlaceholderOp,FusedOp

import logging

logging.basicConfig(level=logging.INFO,filename='logfile.log')
logger = logging.getLogger("[BuddyGraphFuseOps]")



class GraphNode:
    def __init__(self):
        self.name = None
        self.outputs = []
        self.index = 0
        self.ref = None
        self.extern_ref = 0
        self.pattern = OpType.Unfusable


class LinkNode:
    def __init__(self):
        self.value = None
        self.pattern = 0
        self.next = None


class Group:
    def __init__(self):
        self.parent = None
        self.pattern = 0
        self.root_ref = None
        self.master_ref = None
        self.name = None
        self.num_nodes = 1
        self.is_fused = False
        self.fuseop_ref = None
        
    def FindRoot(self):
        if self.parent == None:
            return self
        else:
            root = self
            while root.parent != None:
                root = root.parent
            while self != root:
                parent = self.parent
                self.parent = root
                self = parent
        return root


class BuddyTopoGraph:
    def __init__(self, graph: Graph):
        self.edge_node_dict = {}
        self.post_dfs_order = []
        self.visited_list = []
        self.added_dict = {}
        self.root_flag = 1
        self.root_flag_1 = 1
        self._graph = graph

    def FindNode(self, node_name, nodes):
        for node in nodes:
            if node_name in self._graph.node_table:
                return self._graph.node_table[node_name], "node"
        if node_name == "placeholder":
            return "placeholder", "var"
        elif node_name is None:
            return "None", "var"
        elif isinstance(
            node_name, (TensorDType, int, float, bool)
        ): 
            return type(node_name), "var"
        elif isinstance(node_name,immutable_list):
            if isinstance(node_name[0],torchnode):
                result = []
                for i in node_name:
                    result.append(self._graph.node_table[i.name])
                return result,"node_list"
            else:
                return type(node_name), "var"
        logger.info("cannot find node {0}".format(node_name))

    def Update(self, node, parent, pattern):
        """
        create new graph node with edge and then add to edge_node_dict
        """
        if node.name in self.edge_node_dict.keys():
            current = self.edge_node_dict[node.name]
        else:
            current = GraphNode()
        if node.name in self._graph.node_table:
            if parent is not None:
                link = LinkNode()
                if parent.name not in self.edge_node_dict.keys():
                    logger.error(
                        "cannot find node {0} in edge dict, prob this is the last node".format(
                            parent.name
                        )
                    )
                    exit(1)
                parent = self.edge_node_dict[parent.name]
                link.value = parent
                link.pattern = pattern
                current.name = node.name
                current.outputs.append(link)
            else:
                current.name = node.name
                current.extern_ref = 1
        return current

    def AddNode(self, node, node_pattern):
        if node.name not in self.edge_node_dict.keys():
            logger.error(
                "cannot find node {0} in edge dict, prob this is the last node".format(
                    node.name
                )
            )
            exit(1)
        current = self.edge_node_dict[node.name]
        current.index = len(self.post_dfs_order)
        current.ref = node
        current.pattern = node_pattern
        if node.name not in self.added_dict.keys():
            self.post_dfs_order.append(current)
            self.added_dict[node.name] = current.index
        else:
            index = self.added_dict[node.name]
            self.post_dfs_order[index] = current

    def VisitExpr(self, node):
        """
        build model DAG graph
        """
        # logger.info("-----------------------")
        # logger.info("visitexpr:{0} : {1}".format(node.name,node))
        # logger.info("args:{0}".format(node.args))
        if node in self.visited_list:
            return
        if self.root_flag:
            edge_root_node = self.Update(node, None, OpType.Unfusable)
            self.edge_node_dict[node.name] = edge_root_node
            self.root_flag = 0
        op_pattern = node._op_type
        for input_s in node.args:
            edge_pattern = op_pattern
            if input_s == "PlaceholderOp":
                break
            input_node, node_type = self.FindNode(
                input_s, self._graph._ops_registry
            )
            if node_type == "node":
                #!!!!!!! 若输入输出shape不变，算子类型可退化为kelementwise ⬇️
                edge_node = self.Update(input_node, node, edge_pattern)
                self.edge_node_dict[input_node.name] = edge_node
                self.VisitExpr(input_node)
                self.visited_list.append(input_node)
            elif node_type == "var":
                self.visited_list.append(input_node)
            elif node_type == "node_list":
                for in_node in input_node:
                    edge_node = self.Update(in_node,node,edge_pattern)
                    self.edge_node_dict[in_node.name] = edge_node
                    self.VisitExpr(in_node)
                    self.visited_list.append(in_node)   
        self.AddNode(node, op_pattern)
        return


class DominatorTree:
    def __init__(self, graph: Graph):
        super().__init__()
        self.groups : List[Group] = []
        self.tree_nodes = []
        self._graph = graph
        self.cached_op_groups = {}
    class TreeNode:
        def __init__(self):
            self.name = None
            self.parent = None
            self.depth = 0
            self.pattern = None
            self.index = 0
            self.gnode = None

    def InitGropus(self, graph):
        size = len(graph.post_dfs_order)
        for index in range(size):
            graph_node = graph.post_dfs_order[index]
            group_node = Group()
            group_node.pattern = graph_node.pattern
            group_node.root_ref = graph_node.ref
            group_node.name = graph_node.name
            if group_node.pattern == OpType.ReduceType:
                group_node.master_ref = graph_node.ref
            self.groups.append(group_node)
            self.cached_op_groups[graph_node.name] = [graph_node.ref]
            logger.info("{0} : {1}".format(graph_node.name,graph_node.ref))

    def CombinePattern(self, lhs, rhs):
        if lhs.value > rhs.value:
            return lhs
        return rhs

    def LeastCommonAncestorMulEdges(self, lhs, rhs, edge_pattern):
        while lhs != rhs:
            if lhs == None:
                return None
            if rhs == None:
                return None
            if lhs.depth < rhs.depth:
                edge_pattern = self.CombinePattern(edge_pattern, rhs.pattern)
                rhs = rhs.parent
            elif rhs.depth < lhs.depth:
                edge_pattern = self.CombinePattern(edge_pattern, lhs.pattern)
                lhs = lhs.parent
            else:
                edge_pattern = self.CombinePattern(edge_pattern, lhs.pattern)
                edge_pattern = self.CombinePattern(edge_pattern, rhs.pattern)
                lhs = lhs.parent
                rhs = rhs.parent
        return lhs

    def LeastCommonAncestor(self, edges, edge_pattern, index):
        if len(edges) <= index:
            return None
        link_head = edges[index]

        def get_node(father_node):
            oindex = father_node.index
            return self.tree_nodes[oindex]

        parent = get_node(link_head.value)
        edge_pattern = link_head.value.pattern
        index = index + 1
        for i in range(index, len(edges)):
            link = edges[index]
            parent = self.LeastCommonAncestorMulEdges(
                parent, get_node(link.value), edge_pattern
            )
            edge_pattern = self.CombinePattern(edge_pattern, link.value.pattern)
        return parent

    def GetNode(self, graph_node, graph):
        tree_node = self.TreeNode()
        #!!!!!!
        tree_node.gnode = graph_node
        if graph_node.extern_ref == 1:
            tree_node.name = graph_node.name
            tree_node.depth = 1
            tree_node.parent = None
            tree_node.pattern = "Unfusable"
            tree_node.parent_gnode = graph_node
        else:
            # find the LCAs of all outputs.
            pattern = OpType.ElementwiseType
            tree_node.name = graph_node.name
            parent = self.LeastCommonAncestor(graph_node.outputs, pattern, 0)
            tree_node.depth = parent.depth + 1 if parent else 1
            tree_node.parent = parent
            tree_node.pattern = pattern
            parent_gnode = None
            for node in graph:
                if node.name == parent.name:
                    parent_gnode = node
            assert parent_gnode is not None
            tree_node.parent_gnode = parent_gnode
        return tree_node

    def PostDom(self, graph):
        size = len(graph.post_dfs_order)
        self.tree_nodes = [None] * size
        for i in range(size, 0, -1):
            self.tree_nodes[i - 1] = self.GetNode(
                graph.post_dfs_order[i - 1], graph.post_dfs_order
            )

    def DominatorPartition(self, graph):
        self.InitGropus(graph)
        self.PostDom(graph)


class GraphPartioner:
    def __init__(self, graph: Graph):
        self.fuse = None
        self.visited = []
        self._graph = graph

    def CheckPath_(self, src, sink, fcond, tree):
        if src.name in self.visited:
            return True
        self.visited.append(src.name)
        gnode = tree.groups[src.index]
        assert gnode is not None
        gnode = gnode.FindRoot()
        if not fcond(gnode.pattern, src == sink):
            return False
        if src == sink:
            return True
        for link in src.outputs:
            if not self.CheckPath_(link.value, sink, fcond, tree):
                return False
        return True

    def CheckPath(self, src, sink, fcond, tree):
        assert src.extern_ref == 0, "root node, error"
        self.visited = []
        assert src != sink
        for link in src.outputs:
            if not self.CheckPath_(link.value, sink, fcond, tree):
                return False
        return True

    def MergeFromTo(self, child, parent,tree):
        child = child.FindRoot()
        parent = parent.FindRoot()
        if child == parent:
            return
        parent.num_nodes += child.num_nodes
        tree.cached_op_groups[parent.name][:0] = tree.cached_op_groups[child.name]
        del tree.cached_op_groups[child.name] 
        child.parent = parent
        if child.master_ref is not None:
            assert parent.master_ref is None
            parent.master_ref = child.master_ref
            parent.pattern = child.pattern
        else:
            assert parent.master_ref is not None
            child.master_ref = parent.master_ref
            child.pattern = parent.pattern

    def CommitFuse_(self, src, sink, target, tree):
        if src == sink:
            return
        if src.name in self.visited:
            return
        self.visited.append(src.name)
        gnode = tree.groups[src.index]
        assert gnode is not None
        self.MergeFromTo(gnode, target,tree)
        for link in src.outputs:
            self.CommitFuse_(link.value, sink, target, tree)

    def CommitFuse(self, src, sink, tree):
        target = tree.groups[sink.index]
        logger.info(
            "[Merge] {0} + {1} -> {2}".format(src.name, sink.name, target.name)
        )
        self.visited = []
        assert src != sink
        self.CommitFuse_(src, sink, target, tree)

    def RunFuse(self, graph, tree):
        def fcond0(kind, issink):
            return kind.value <= OpType.BroadcastType.value

        for phase in range(0, 1):
            for i in range(0, len(tree.groups)):
                graph_node = graph.post_dfs_order[i]
                dom_node = tree.tree_nodes[i]
                group_node = tree.groups[i]
                if dom_node != None and group_node.pattern == OpType.ReduceType:
                    if phase != 0:
                        continue
                    if (
                        dom_node.parent != None
                        and dom_node.pattern == OpType.ElementwiseType
                    ):
                        if self.CheckPath(
                            graph_node, dom_node.parent_gnode, fcond0, tree
                        ):
                            self.CommitFuse(
                                graph_node, dom_node.parent_gnode, tree
                            )
            
    def FinishFuse(self,tree):
        self._graph.op_groups = {}
        cnt = 0
        for node_name in tree.cached_op_groups.keys():
            if (len(tree.cached_op_groups[node_name]) == 1):
                if isinstance(tree.cached_op_groups[node_name][0],PlaceholderOp):
                    continue
                self._graph.maingroup.append(tree.cached_op_groups[node_name][0])
            else:
                fops = FusedOp();
                fops.name = "fused_"+str(cnt)
                fops.fused_ops = tree.cached_op_groups[node_name]
                for i in fops.fused_ops:
                    i._is_fused = True
                    i.fuse_ref = fops
                self._graph.maingroup.append(fops)
                self._graph.node_table[fops.name] = fops
                self._graph.op_groups[fops.name] = fops.fused_ops
            cnt += 1
        #construct each fusedop:
        # Identify inputs for each op_group
        for name in self._graph.op_groups.keys():
            inputs = []
            for op in self._graph.op_groups[name]:
                for parent in op._parents:
                    if (
                        self._graph.node_table[parent]
                        not in self._graph.op_groups[name]
                        and 
                        parent not in inputs
                    ):
                        inputs.append(parent)
            self._graph.node_table[name]._parents = inputs  
        
        for name in self._graph.op_groups.keys():
            outputs = []
            for op in self._graph.op_groups[name]:
                for child in op._children:
                    if (self._graph.node_table[child] not in self._graph.op_groups[name]):
                        outputs.append(op.name)
                        self._graph.node_table[name]._children.append(child)
                   
        print("totol fuse:{0}".format(cnt))
        logger.info("------------maingroup pattern----------------")
        for i,n in enumerate( self._graph.maingroup ):
            logger.info("maingroup.op_groups[{0}]:{1}".format(i,n.name))
            logger.info("node message:{0}".format(n))
            if isinstance(n, FusedOp):
                logger.info("fused_ops:{0}".format(n.fused_ops))
                for cnt,j in enumerate(n.fused_ops):
                    logger.info("cnt op:")
                    logger.info("parents:{0}".format(j._parents))
                    logger.info("childs{0}:".format(j._children))
            logger.info("-----------------------------------------")
        pass

def my_fuse_ops_test(graph: Graph):
    graph._ops_registry["OutputOp"] = OutputOp
    graph._ops_registry["PlaceholderOp"] = PlaceholderOp
    topo_graph = BuddyTopoGraph(graph)
    topo_graph.VisitExpr(graph.body[-1])
    for i,node in enumerate(topo_graph.visited_list):
        if isinstance(node,Op):
            logger.info("{0} : {1}".format(i,node.name))
        else:
            logger.info("{0} : {1}".format(i,node))
        
    post_dom_tree = DominatorTree(graph)
    post_dom_tree.DominatorPartition(topo_graph)
    
    fuse_op_object = GraphPartioner(graph)
    fuse_op_object.RunFuse(topo_graph, post_dom_tree)
    fuse_op_object.FinishFuse(post_dom_tree)

def simply_fuse(graph: Graph):
    """
    Function to fuse all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """

    new_op_group = []
    device = DeviceType.UNKNOW
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    cnt = 0
    logger.info("-------------------------------------")
    for i,op in enumerate(new_op_group):
        cnt+=1
        logger.info("op{0}: {1}".format(i,op))
        logger.info("op parents:{0}".format(op._parents))
        logger.info("op children:{0}".format(op._children))
    logger.info("-------------------------------------")
    print("toto op number:{0}!!!!!!!!!".format(cnt))
    # graph.op_groups["subgraph0"] = new_op_group
    # graph.group_map_device = {"subgraph0": device}
    
    my_fuse_ops_test(graph)
    #graph.op_groups.clear()
    print("!!!!!!!!!!!!!!!!!!!!finish fuse test!!!!!!!!!!!!")
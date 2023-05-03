class GraphNode():
    def __init__(self, left_child, right_child, content=None, is_terminal=False):
        self.left_child=left_child
        self.right_child=right_child
        self.content=content
        self.is_terminal=is_terminal
        self.parent=None

    def add_parent(self, parent):
        self.parent=parent

    def add_index(self, idx):
        self.index = idx

def create_node(tree):
    if type(tree) == str:
        return GraphNode(None, None, content=tree, is_terminal=True)
    else:
        return GraphNode(create_node(tree[0]), create_node(tree[1]))

class Graph():
    def __init__(self, tree):
        '''
            Tuple object
        '''
        root = create_node(tree)
        self.reachable = {}
        self.get_all_reachable(root)
        self.add_idxs(root, 0)
        ### construct parent points
        self.parent = {root: None}
        self.add_parent_pointers(root)

        self.idx_dict = {}
        for node in self.parent:
            if node.is_terminal:
                self.idx_dict[node.index] = node
            self.reachable[node] = set(self.reachable[node])            
        ### now we know what nodes are adjacent to a given node...
        self.root = root

    def get_distances(self, idx):
        # get distance of every other leaf node from this leaf node
        node = self.idx_dict[idx]
        all_distances = []
        visited = set()

        self.dfs(node, visited, 0, all_distances)
        return all_distances

    def lca(self, idx1, idx2):
        node1 = self.idx_dict[idx1]
        node2 = self.idx_dict[idx2]

        dist = self.lca_helper(node1, node2, 0)
        return dist

    def get_all_reachable(self, node):
        if node.is_terminal:
            self.reachable[node] = [node]
        else:
            lchild = node.left_child
            rchild = node.right_child
            l = self.get_all_reachable(lchild)
            r = self.get_all_reachable(rchild)
            self.reachable[node] = l + r + [node]
        return self.reachable[node]


    def lca_helper(self, node1, node2, dist):
        p1 = self.parent[node1]
        if p1 == None:
            return -1
        elif node2 in self.reachable[p1]:
            return dist
        else:
            return self.lca_helper(p1, node2, dist+1)

    def get_constituents(self, max_depth=10000000):
        '''
            For every node in the graph get:
            - the leftmost leaf of the right child: and create a split of 
            (left side leaves) (right side leaves) except the leftmost leaf of the right side
            - the rightmost leaf of the left child: and create a split of 
            (left side leaves) (right side leaves) except the rightmost leaf the left side is left out

            as the output we return two lists of tuples where the first is a set of indices for constituents that will change more, 
            the second is the index of the word we are going to change, and the third is the indices of constituents that will change less 
        '''
        node2leaves = {}
        self.visit(self.root, node2leaves, 0)

        all_constituents_left = []
        all_constituents_right = []
        for node in node2leaves:
            l, r, depth = node2leaves[node]
            if depth > max_depth:
                continue
            if len(l) > 1 and len(r) > 0:
                ### l[:-1] will change more than r if we change l[-1]
                all_constituents_left.append((l[:-1], l[-1], r))
            if len(r) > 1 and len(l) > 0:
                ### r[1:] will change more than l if we change r[0]
                all_constituents_right.append((r[1:], r[0], l))

        return all_constituents_left + all_constituents_right

    def visit(self, node, node2leaves, curr_depth):
        if not node.is_terminal:
            lchild = node.left_child
            rchild = node.right_child
            self.visit(lchild, node2leaves, curr_depth+1)
            self.visit(rchild, node2leaves, curr_depth+1)
            if lchild in node2leaves:
                l1, r1, _ = node2leaves[lchild]
            else:
                assert(lchild.is_terminal)
                l1 = [lchild.index]
                r1 = []
            
            if rchild in node2leaves:
                l2, r2, _ = node2leaves[rchild]
            else:
                assert(rchild.is_terminal)
                r2 = [rchild.index]
                l2 = []

            node2leaves[node] = (l1 + r1), (l2 + r2), curr_depth
        return 

    def dfs(self, node, visited, idx, all_distances):
        if node in visited:
            return
        else:
            visited.add(node)
            if not node.is_terminal:
                lchild = node.left_child
                self.dfs(lchild, visited, idx+1, all_distances)
                rchild = node.right_child
                self.dfs(rchild, visited, idx+1, all_distances)
            else:
                all_distances.append((node.index, idx))
            parent = self.parent[node]
            if parent:
                self.dfs(parent, visited, idx+1, all_distances)
        return

    def add_parent_pointers(self, root):
        if root.is_terminal:
            return
        else:
            lchild = root.left_child
            rchild = root.right_child
            self.parent[lchild] = root
            self.parent[rchild] = root
            self.add_parent_pointers(lchild)
            self.add_parent_pointers(rchild)

    def add_idxs(self, root, idx):
        if root.is_terminal:
            root.add_index(idx)
            return idx+1
        else:
            idx2 = self.add_idxs(root.left_child, idx)
            idx3 = self.add_idxs(root.right_child, idx2)
            return idx3

if __name__ == '__main__':
    parse = (('its', ((('a', 'place'), ('where', 'your')), ('parents', 'wouldnt'))), ((('even', 'care'), (('if', 'you'), ('stayed', 'out'))), (('late', 'biking'), (('with', 'your'), 'friends'))))
    graph = create_node(parse)
    graph_obj = Graph(parse)
    #distances = graph_obj.get_distances(5)
    out = graph_obj.get_constituents()
    print(graph_obj.lca(1, 10))
    print(out)

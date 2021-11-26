**When do we use BFS**

#### Traversal in Graph

- Level Order Traversal
- Connected component
- Topological Sorting

A **graph** is described by G = <E, V>, i.e. a graph is consist of a set of edges and vertices.

#### Shortest Path (SP) in Simple Graph

- Get the SP from point A to B = Level order traversal from A to B
- Simple graph: each edge has same length

#### Iteration solution for all possible results *

Details explanation in DFS.



**Cheat info**

Stack --> DFS

Queue --> BFS



# Problems

### 102. Binary Tree Level Order Traversal

[LeetCode](https://leetcode.com/problems/binary-tree-level-order-traversal/), [LintCode](https://www.lintcode.com/problem/binary-tree-level-order-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        # 1. put root into queue
        queue = deque([root])
        ans = []
        # 2. while queue is not empty
        while queue:
            level = []
            # for current level, append all nodes into level list
            # append all nodes' children into queue
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(level)
        return ans
```



### 297. Serialize and Deserialize Binary Tree

[LeetCode](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/), [LintCode](https://www.lintcode.com/problem/7/)

**Solution**:

```python
from collections import deque
class Codec:
    def serialize(self, root):
        if not root:
            return ""
        queue = deque([root])
        ans = []
        while queue:
            for i in range(len(queue)):
                node = queue.popleft()
                if node is None:
                    ans.append("#")
                else:
                    ans.append(node.val)
                    if node.left:
                        queue.append(node.left)
                    else:
                        queue.append(None)
                    if node.right:
                        queue.append(node.right)
                    else:
                        queue.append(None)
        return ans

    def deserialize(self, data):
        if not data:
            return None
        nodes = [TreeNode(i) if i is not "#" else None for i in data]
        node_idx = -1
        for i, node in enumerate(nodes):
            if node:
                node_idx += 1
            else:
                continue
            if nodes[node_idx*2+1]:
                node.left = nodes[node_idx*2+1]
            if nodes[node_idx*2+2] is not "#":
                node.right = nodes[node_idx*2+2]
        return nodes[0]
```



### 107. Binary Tree Level Order Traversal II

[LeetCode](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)

**Solution**: DFS, append each level nodes to the start of the result, i.e. use a `deque` and append nodes from left.

```python
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        ans = deque([])
        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.appendleft(level)
        return ans
```



### 103. Binary Tree Zigzag Level Order Traversal

[LeetCode](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

**Solution**: In addition to Problem 102, we introduce a indicator `is_left_right` initiated as `True`, while it is true, we append nodes to `level` from left, otherwise we append nodes to `level` from right.

```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        ans = []
        is_left_right = True
        while queue:
            level = deque([])
            for _ in range(len(queue)):
                node = queue.popleft()
                if is_left_right:
                    level.append(node.val)
                else:
                    level.appendleft(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(level)
            is_left_right = not is_left_right
        return ans
```





### 137. Clone Graph

[LintCode](https://www.lintcode.com/problem/clone-graph/description)

```python
"""
Definition for a undirected graph node
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""

class Solution:
    """
    @param: node: A undirected graph node
    @return: A undirected graph node
    """
    def cloneGraph(self, node):
        if not node:
            return node
        
        head = node
        nodes = set([node])
        queue = collections.deque([node])
        
        # bfs - get all nodes from node
        while queue:
            node = queue.popleft()
            for neighbor in node.neighbors:
                if neighbor not in nodes:
                    queue.append(neighbor)
                    nodes.add(neighbor)
        
        # node - new node dict 
        node_newnode = {}
        
        # create new nodes based on label values
        for node in nodes:
            node_newnode[node] = UndirectedGraphNode(node.label)
            
        # feed in new neighbors to new nodes
        for node in nodes:
            new_node = node_newnode[node]
            for neighbor in node.neighbors:
                new_node.neighbors.append(node_newnode[neighbor])
                
        return node_newnode[head]
```


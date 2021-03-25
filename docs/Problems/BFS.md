**When do we use BFS**

#### Traversal in Graph

- Level Order Traversal
- Connected component
- Topological Sorting

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


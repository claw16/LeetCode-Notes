# Binary Tree

### Traverse a Binary Tree

There are many ways to traverse a binary tree:

* Level order - BFS
* Pre-order
* In-order
* Post-order

Example codes for pre-order, in-order, and post-order, respectively.

```python
# 将根作为root，空list作为result传入，即可得到整棵树的遍历结果
def traverse(root, result):
    if not root:
        return
    result.append(root.val)
    traverse(root.left, result)
    traverse(root.right, result)
```

```python
def traverse(root, result):
    if not root:
        return
    traverse(root.left, result)
    result.append(root.val) # 注意访问根节点放到了遍历左子树的后面
    traverse(root.right, result)
```

```python
def traverse(root, result):
    if not root:
        return
    traverse(root.left, result)
    traverse(root.right, result)
    result.append(root.val) # 注意访问根节点放到了最后
```


# Heap

### Definition

1. A heap is a special case of a complete binary tree.

2. Every node in a heap is always greater (max-heap) than its descendants



 ### Representation

Usually we use an array to represent a heap. A[0] is the root. The children of A[i] are A[2*i+1] and A[2*i+2], respectively.



### Operations

#### Insertion O(lg n)

1. Insert new node to the end of the heap.

2. Check the new node and its ancestor to verify whether this (sub-)tree meets all properties of a heap's definition. 

   a. if Yes, done.

   b. if No, swap the new node and its ancestor. Repeat step 2.

#### Deletion

1. Swap the target node with the last node in the heap.
2. Delete the target node.
3. Check the children and ancestor of the swapped node, if either of them does not meet the definition of a heap, recursively swap down or up until the heap becomes legal.

Note: in step 3, the swapping direction must be always swapping up or always swapping down.

#### Heap construction

1. Initialize a binary tree by the given keys.
2. Starting from the lowest non-leaf node, check the heap condition. If it does not meet the condition, keep swapping it with its largest child, until heap condition is met.
3. Repeat step 2.

### Min / Max Heap O(1)

The minimum value of a min-heap is in its root.



### Python Doc: heaqp

Explanation of this module can be found [here](https://docs.python.org/3/library/heapq.html).



### Example 1:

Take [612. K Closest Points](https://www.lintcode.com/problem/k-closest-points/description) as an example. The problem requires us to output K closest points in terms of distance to the origin point. If two or more points have the same distance, sort by x-axis; and if they are the same in x-axis, sort by y-axis.

We can use a heap store (distance, x, y), it sorts the tuple by distance, if distances are the same, it then sorts by x, and so forth.



### Example 2:

Overwrite the compare function of a node so that we can directly push ListNode into heapq.

```python
ListNode.__lt__ = lambda x, y: (x.val < y.val)
```


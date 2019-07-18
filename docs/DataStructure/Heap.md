# Heap

### Definition

1. A heap is a special case of a complete binary tree.

2. Every node in a heap is always greater (or smaller) than its descendants



 ### Representation

Usually we use an array to represent a heap. A[0] is the root. The children of A[i] are A[2*i+1] and A[2*i+2], respectively.



### Operations

#### Insertion

1. Insert new node to the end of the heap.

2. Check the new node and its ancestor to verify whether this (sub-)tree meets all properties of a heap's definition. 

   a. if Yes, done.

   b. if No, swap the new node and its ancestor. Repeat step 2.

#### Deletion

1. Swap the target node with the last node in the heap.
2. Delete the target node.
3. Check the children and ancestor of the swapped node, if either of them does not meet the definition of a heap, recursively swap down or up until the heap becomes legal.

#### Heap construction

1. Initialize a binary tree by the given keys.
2. Starting from the lowest non-leaf node, check the heap condition. If it does not meet the condition, keep swapping it with its largest child, until heap condition is met.
3. Repeat step 2.



### Python Doc: heaqp

Explanation of this module can be found [here](https://docs.python.org/3/library/heapq.html).
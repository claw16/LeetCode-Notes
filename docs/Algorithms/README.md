# Algorithms

### Quick Sort

Given an array, define two pointers - left and right, pointing to the first and last element in the array, respectively. Randomly select an element (value, not index) as the pivot. Move `left` to the right, and `right` to the left, make elements that are smaller than the pivot to the left of pivot, those are greater than the pivot to the right of the pivot. Then recursively call the function itself to re-arrange the sub-arrays to the left and right of the pivot.

##### Note:

* Base case: when start >= end, meaning that the sub-array has only one element, no need to sort.
* Use >= or <= when comparing something to `start` or `end`.
* Use > or < when comparing value to the pivot value.
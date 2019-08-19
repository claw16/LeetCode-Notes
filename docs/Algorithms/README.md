# Algorithms

### Quick Sort

Given an array, define two pointers - left and right, pointing to the first and last element in the array, respectively. Randomly select an element (value, not index) as the pivot. Move `left` to the right, and `right` to the left, make elements that are smaller than the pivot to the left of pivot, those are greater than the pivot to the right of the pivot. Then recursively call the function itself to re-arrange the sub-arrays to the left and right of the pivot.

##### Note:

* Base case: when start >= end, meaning that the sub-array has only one element, no need to sort.
* Use >= or <= when comparing something to `left` or `right`.
* Use > or < when comparing value to the pivot value.

```python
class Solution:
    def sortIntegers(self, A):
        # write your code here
        if not A or len(A) == 0:
            return
        
        self.quick_sort(A, 0, len(A) - 1)
    
    def quick_sort(self, A, start, end):
        if start >= end:
            return
        left, right = start, end
        pivot = A[(start + end) // 2]
        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1
            while left <= right and A[right] > pivot:
                right -= 1
            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
        self.quick_sort(A, start, right)
        self.quick_sort(A, left, end)

```



### Binary Search

General template:

```python
class Solution:
    # @param nums: The integer array
    # @param target: Target number to find
    # @return the first position of target in nums, position start from 0 
    def binarySearch(self, nums, target):
        if not nums:
            return -1
        start, end = 0, len(nums) - 1
        # 用 start + 1 < end 而不是 start < end 的目的是为了避免死循环
        # 在 first position of target 的情况下不会出现死循环
        # 但是在 last position of target 的情况下会出现死循环
        # 样例：nums=[1，1] target = 1
        # 为了统一模板，我们就都采用 start + 1 < end，就保证不会出现死循环
        while start + 1 < end:
            # python 没有 overflow 的问题，直接 // 2 就可以了
            # java和C++ 最好写成 mid = start + (end - start) / 2
            # 防止在 start = 2^31 - 1, end = 2^31 - 1 的情况下出现加法 overflow
            mid = (start + end) // 2
            
            # > , =, < 的逻辑先分开写，然后在看看 = 的情况是否能合并到其他分支里
            if nums[mid] < target:
                # 写作 start = mid + 1 也是正确的
                # 只是可以偷懒不写，因为不写也没问题，不会影响时间复杂度
                # 不写的好处是，万一你不小心写成了 mid - 1 你就错了
                start = mid
            elif nums[mid] == target:
                end = mid
            else: 
                # 写作 end = mid - 1 也是正确的
                # 只是可以偷懒不写，因为不写也没问题，不会影响时间复杂度
                # 不写的好处是，万一你不小心写成了 mid + 1 你就错了
                end = mid
                
        # 因为上面的循环退出条件是 start + 1 < end
        # 因此这里循环结束的时候，start 和 end 的关系是相邻关系（1和2，3和4这种）
        # 因此需要再单独判断 start 和 end 这两个数谁是我们要的答案
        # 如果是找 first position of target 就先看 start，否则就先看 end
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        
        return -1
```


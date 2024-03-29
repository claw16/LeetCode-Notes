# LintCode Problems

### 415. 有效回文串

[题目描述](https://www.lintcode.com/problem/valid-palindrome/description)

##### 题解 1：

思路：

1. 将string中所有不是数字和字母的char去除。
2. 不断比较字符串第一个和最后一个char，一旦不相等，直接返回False；如果相等，删除首位char，继续比较，直到字符串长度小于等于1，返回True。

```python
class Solution:
    def isPalindrome(self, s):
        # write your code here
        s = ''.join(char for char in s if char.isalnum()).lower()
        while len(s) > 1:
            if s[0] != s[-1]: return False
            else: s = s[1:-1]
        return True
```

##### 题解 2

思路：

1. 同上
2. 使用双指针start和end，分别指向string的首位。当start和end的位置没有重合甚至交叉的时候，比较两个位置的char是否相同，如果否，返回False；如果相同，start右移一格，end左移一格，继续比较。跳出循环后返回True。

```python
class Solution:
    def isPalindrome(self, s):
        # write your code here
        s = ''.join(char for char in s if char.isalnum()).lower()
        start, end = 0, len(s) - 1
        while start < end:
            if s[start] != s[end]: return False
            start += 1
            end -= 1
        return True
```



### 627. 最长回文串

[题目描述](https://www.lintcode.com/problem/longest-palindrome/description)

##### 题解 1

思路：用一个字典记录字符串中每种字符出现的次数。初始化一个变量`ans = 0`用来记录结果。如果字符出现次数大于1且为奇数，ans中加上`次数-1`；如果字符出现次数大于1且为偶数，ans中加上`次数`。如果ans和字符串长度相等，直接返回ans；否则返回ans+1。

```python
class Solution:
	def longestPalindrome(self, s):
        # write your code here
        char_numbers = {}
        for char in s:
            if char not in char_numbers: char_numbers[char] = 1
            else: char_numbers[char] += 1
        ans = 0
        for _, num in char_numbers.items():
            if num > 1 and num % 2 == 0: ans += num
            elif num > 1 and num % 2 != 0: ans += num-1
        return ans + 1 if len(s) > ans else ans
```



### 891. 有效回文 II

[题目描述](https://www.lintcode.com/problem/valid-palindrome-ii/description)

##### 题解思路：

应用双向双指针，逐字符检查。第一次出现left和right指针对应的字符不一样的情况，跳过左边或者右边的一个字符，即执行一次`left += 1`或者`right -= 1`，继续检查。

```python
class Solution:
    def validPalindrome(self, s):
        left, right = self.two_ptr(s, 0, len(s) - 1)
        if left >= right:
            return True
        # left < right --> try to delete 1 character
        return self.is_palindrome(s, left + 1, right) or self.is_palindrome(s, left, right - 1)
        
    def is_palindrome(self, s, left, right):
        left, right = self.two_ptr(s, left, right)
        return left >= right
        
    def two_ptr(self, s, left, right):
        while left < right:
            if s[left] != s[right]:
                return left, right
            left += 1
            right -= 1
        return left, right
```





### 13.字符串查找

[题目描述](https://www.lintcode.com/problem/implement-strstr/description)

##### 题解 1

* 边界条件：如果source==target，直接返回0
* 遍历source，在每个位置比较与target长度相等的子串跟target是否相同，如果是，直接返回当前位置index；如果否，继续遍历，最终返回False。

```python
class Solution:
    def strStr(self, source, target):
        if source == target: return 0
        n_target = len(target)
        for i in range(len(source)):
            if source[i:i+n_target] == target: return i
        return -1
```



##### 题解 2

[九章解法](https://www.jiuzhang.com/solution/implement-strstr/#tag-highlight-lang-python)



### 127. 拓扑排序

[题目描述](https://www.lintcode.com/problem/topological-sorting/description)

```python
"""
Definition for a Directed graph node
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""


class Solution:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # calculate in-degree for each node
        indegrees = {node:0 for node in graph}
        for node in graph:
            for neighbor in node.neighbors:
                indegrees[neighbor] += 1
        
        queue = collections.deque([])
        
        for node in graph:
            if indegrees[node] == 0:
                queue.append(node)
        
        ans = []
        
        while queue:
            node = queue.popleft()
            ans.append(node)
            
            for neighbor in node.neighbors:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
                    
        return ans
```





### 7. 二叉树的序列化和反序列化

[题目描述](https://www.lintcode.com/problem/serialize-and-deserialize-binary-tree/description)

!!! 需要反复做

```python
class Solution:

    def serialize(self, root):
        # write your code here
        if root is None:
            return ""
        
        data = []
        queue = collections.deque([root])
        
        while queue:
            node = queue.popleft()
            data.append(str(node.val) if node else '#')
            
            if node:
                queue.append(node.left)
                queue.append(node.right)
        
        return ' '.join(data) 

    def deserialize(self, data):
        # write your code here
        if not data:
            return None
        
        bfs_order = []
        n_nodes = 0
        for nodeval in data.split():
            if nodeval == '#':
                bfs_order.append(None)
            else:
                n_nodes += 1
                bfs_order.append(TreeNode(int(nodeval)))
        
        root = bfs_order[0]
        index = 0
        nodes = [root]
        
        while index < n_nodes:
            nodes[index].left = bfs_order[index*2 + 1]
            nodes[index].right = bfs_order[index*2 + 2]
            
            if nodes[index].left:
                nodes.append(nodes[index].left)
            if nodes[index].right:
                nodes.append(nodes[index].right)
            
            index += 1
                
        return root
```





### 120. 单词接龙

[题目描述](https://www.lintcode.com/problem/word-ladder/description)

```python
class Solution:
    def ladderLength(self, start, end, dict):
        # add end into dict, because as the question stated,
        # start and end are not in dict.
        dict.add(end)
        
        queue = collections.deque([start])
        distance = {start: 1}
        
        while queue:
            cur_word = queue.popleft()
            if cur_word == end:
                return distance[cur_word]
            
            for next_word in self.get_next_word(cur_word, dict):
                if next_word not in distance:
                    distance[next_word] = distance[cur_word] + 1
                    queue.append(next_word)
        return 0
        
    def get_next_word(self, word, dict):
        next_words = []
        for i in range(len(word)):
            left = word[:i]
            right = word[i+1 :]
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char == word[i]:
                    continue
                if left + char + right in dict:
                    next_words.append(left + char + right)
        return next_words
```



### 66. 二叉树的前序遍历

[题目描述](https://www.lintcode.com/problem/binary-tree-preorder-traversal/description)

```python
class Solution: 
    def preorderTraversal(self, root):
        # write your code here
        
        self.ans = []
        self.traverse(root)
        return self.ans
        
    def traverse(self, root):
        if not root:
            return
        self.ans.append(root.val)
        self.traverse(root.left)
        self.traverse(root.right)
```



### 73. 前序遍历和中序遍历树构造二叉树

[题目描述](https://www.lintcode.com/problem/construct-binary-tree-from-preorder-and-inorder-traversal/description)

思路：

利用前序和中序遍历的特性。preorder的第一个元素一定是树的root。找到这个root在inorder中的位置，那么root左边的元素就是左子树，右边的是右子树。递归一下就可以得到答案。

```python
class Solution:
    def buildTree(self, preorder, inorder):
        if not inorder: return None
        root = TreeNode(preorder[0])
        rootPos = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : 1 + rootPos], inorder[ : rootPos])
        root.right = self.buildTree(preorder[rootPos + 1 : ], inorder[rootPos + 1 : ])
        return root
```



### 97. 二叉树的最大深度

[题目描述](https://www.lintcode.com/problem/maximum-depth-of-binary-tree/description)

##### 思路：

If root is empty, return 0.

Recursively visited left and right child node, return the height that is greater and plus 1 (1 comes from the increased height by the root).

```python
class Solution:
    def maxDepth(self, root):
        if root is None:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```



### 93. 平衡二叉树

[题目描述](https://www.lintcode.com/problem/balanced-binary-tree/description)

##### 思路：

Use the idea of [Q97](https://www.lintcode.com/problem/maximum-depth-of-binary-tree/description), calculate the heights of left and right children, if the different is less than or equal to 1, return True; otherwise return False.

```python
class Solution:
    def isBalanced(self, root):
        is_balance, _ = self.validate(root)
        return is_balance
    
    def validate(self, root):
        if not root:
            return True, 0
            
        is_balance, left_height = self.validate(root.left)
        if not is_balance:
            return False, 0
        is_balance, right_height = self.validate(root.right)
        if not is_balance:
            return False, 0
        return abs(left_height - right_height) <= 1,  max(left_height, right_height) + 1
```



### 95. 验证二叉查找树

[题目描述](https://www.lintcode.com/problem/validate-binary-search-tree/description)

##### 题解1：

利用BST的性质：对BST进行in-order traversal，将得到一个sorted array。中序遍历，如果当前节点值大于上一个节点值，则True；否侧False。利用Dummy node，节省代码量。

```python
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def isValidBST(self, root):
        # write your code here
        dummy = TreeNode(-sys.maxsize)
        dummy.right = root
        stack = [dummy]
        last = dummy
        
        while stack:
            node = stack.pop()
            if last != dummy and node.val <= last.val:
                return False
            #else:
            last = node
            if node.right:
                root = node.right
                while root:
                    stack.append(root)
                    root = root.left
        return True
```





### 521. 去除重复元素

[题目描述](https://www.lintcode.com/problem/remove-duplicate-numbers-in-array/description)

##### 思路：

先将数组排序。利用两个指针，`i`从位置1开始遍历数组，`ans`从位置1开始，一旦发现下一个数字跟当前数字相同，`ans`保持不动，直到`i`找到一个新的数字，将新数字覆盖到`ans`的位置，`ans`右移一步。

```python
class Solution:
    # @param {int[]} nums an array of integers
    # @return {int} the number of unique integers
    def deduplication(self, nums):
        # Write your code here
        if len(nums) == 0:
            return 0
        nums.sort()
        ans = 1
        for i in range(1, len(nums)):
            if nums[i-1] != nums[i]:
                nums[ans] = nums[i]
                ans += 1
        return ans
```



### 604. 滑动窗口内数的和

[题目描述](https://www.lintcode.com/problem/window-sum/description)

##### 思路：

初始化数组window，把第一个k个数的和放进去，然后从位置1开始遍历，注意遍历结尾位置的定义。每到一个新位置，以前一个和的值为基准，减去出去的数字，加上进来的数字，保存为当前窗口的和，加入window数组。

```python
class Solution:
    def winSum(self, nums, k):
        # write your code here
        if k == 0:
            return nums
        window = [sum(nums[0 : k])]
        for i in range(1, len(nums) - k + 1):
            window.append(window[i-1] - nums[i-1] + nums[i-1+k])
        return window
```



### 228. 链表的中点

[题目描述](https://www.lintcode.com/problem/window-sum/description)

##### 思路：

利用快慢指针。每次，慢一步，快两步。当快到达末尾时，慢刚好是中间点，返回慢指针。

[题目描述](https://www.lintcode.com/problem/middle-of-linked-list/description)

```python
class Solution:
    """
    @param head: the head of linked list.
    @return: a middle node of the linked list
    """
    def middleNode(self, head):
        # write your code here
        if not head:
            return None
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```



### 607. Two Sum III - Data structure design

[LintCode](https://www.lintcode.com/problem/two-sum-iii-data-structure-design/description)

##### Solution I:

- `add() O(1)`: simply append to the end of an array.
- `find() O(nlogn)`: sort array and use 2 pointers.

```python
class TwoSum:
    """
    @param number: An integer
    @return: nothing
    """
    def __init__(self):
        self.arr = []
    
    def add(self, number):
        # write your code here
        self.arr.append(number)
    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """
    def find(self, value):
        # write your code here
        if len(self.arr) < 2:
            return False
            
        start, end = 0, len(self.arr) - 1
        self.arr.sort()
        
        while start + 1 < end:
            if self.arr[start] + self.arr[end] == value:
                return True
            if self.arr[start] + self.arr[end] < value:
                start += 1
            if self.arr[start] + self.arr[end] > value:
                end -= 1
        if self.arr[start] + self.arr[end] == value:
            return True
        return False
```

##### Solution II:

- `add() O(1)`: add to a dict, added number is the key, and its appearance is the corresponding value.
- `find() O(n)`: traverse the dict, if `target - current_number` is also in the dict; or it appears more than once in the dict, return True.

```python
class TwoSum:
    """
    @param number: An integer
    @return: nothing
    """
    def __init__(self):
        self.dict = {}
    
    def add(self, number):
        # write your code here
        if number in self.dict:
            self.dict[number] += 1
        else:
            self.dict[number] = 1
            
    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """
    def find(self, value):
        # write your code here
        for num in self.dict:
            if value - num in self.dict and (value - num != num or self.dict[num] > 1):
                return True
        return False
```



### 539. Move Zeros

[LintCode](https://www.lintcode.com/problem/move-zeroes/description)

##### Solution:

```python
class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """
    def moveZeroes(self, nums):
        # write your code here
        slow = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                if nums[slow] != nums[i]:
                    nums[slow] = nums[i]
                slow += 1
        
        for j in range(slow, len(nums)):
            if nums[j] != 0:
                nums[j] = 0
```



### 464. Sort Integers II (quicksort)

[LintCode](https://www.lintcode.com/problem/sort-integers-ii/description)

```python
class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers2(self, A):
        # write your code here
        start, end = 0, len(A) - 1
        self.quicksort(A, start, end)
        
    def quicksort(self, A, start, end):
        if start >= end:
            return
        left, right = start, end
        pivot = A[(left + right) // 2]
        while left <= right:
            while A[left] < pivot:
                left += 1
            while A[right] > pivot:
                right -= 1
            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
        self.quicksort(A, start, right)
        self.quicksort(A, left, end)
```



### 143. Sort Colors II

[LintCode](https://www.lintcode.com/problem/sort-colors-ii/description)

##### Solution:

The idea is almost the same with quicksort, pay extra attention to the `start` and `end` intervals and `while` conditions.

```python
class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sortColors2(self, colors, k):
        # write your code here
        start, end = 0, len(colors) - 1
        self.quicksort(colors, start, end, 1, k)
        
    def quicksort(self, colors, start, end, color_from, color_to):
        if start == end or color_from == color_to:
            return
        left, right = start, end
        color = (color_from + color_to) // 2
        while left <= right:
            while colors[left] <= color:
                left += 1
            while colors[right] > color:
                right -= 1
            if left <= right:
                colors[left], colors[right] = colors[right], colors[left]
                left += 1
                right -= 1
        self.quicksort(colors, start, right, color_from, color)
        self.quicksort(colors, left, end, color + 1, color_to)
```



### 57. 3Sum

[LintCode](https://www.lintcode.com/problem/3sum/description)

##### Solution:

Assume a candidate solution is `[a, b, c]`, we have `a + b + c = 0`, or `-a = b + c`. Then this question becomes a TwoSum question whose target is `-a`.

```python
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, numbers):
        # write your code here
        numbers.sort()
        ans = []
        for i in range(len(numbers) - 2):
            # ignore duplicate of a
            if i and numbers[i] == numbers[i - 1]:
                continue
            # a must be negative
            if numbers[i] > 0:
                continue
            self.two_sum(numbers, i + 1, len(numbers) - 1, ans)
        return ans
            
    def two_sum(self, numbers, start, end, ans):
        target = - numbers[start - 1]
        while start < end:
            if numbers[start] + numbers[end] == target:
                ans.append([-target, numbers[start], numbers[end]])
                start += 1
                end -= 1
                # ignore duplicates of b and c
                while start < end and numbers[start] == numbers[start - 1]:
                    start += 1
                while start < end and numbers[end] == numbers[end + 1]:
                    end -= 1
            elif numbers[start] + numbers[end] > target:
                end -= 1
            else:
                start += 1
```



### 5. Kth Largest Element

[LintCode](https://www.lintcode.com/problem/kth-largest-element/description)

##### Solution:

An important principle of partition in quicksort: after 1 round of partition, the pivot is located at its final position of the array.

Based on this principle, every call of partition, we check if the position of the pivot is the position we want. For example, if we want 3rd smallest element, and if pivot index is 2, then pivot is the solution.

```python
class Solution:
    """
    @param n: An integer
    @param nums: An array
    @return: the Kth largest element
    """
    def kthLargestElement(self, n, nums):
        return self.partition(nums, 0, len(nums) - 1, len(nums) - n)
        
    def partition(self, nums, start, end, n):
        if start == end:
            return nums[n]
        left, right = start, end
        pivot = nums[(left + right) // 2]
        while left <= right:
            while nums[left] < pivot and left <= right:
                left += 1
            while nums[right] > pivot and left  <= right:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        if left <= n:
            return self.partition(nums, left, end, n)
        if right >= n:
            return self.partition(nums, start, right, n)
        return nums[n]
```





### 31. Partition Array

[LintCode](https://www.lintcode.com/problem/partition-array/description)

##### Solution:

To find the solution, there is no need to sort the array, only partition once the array is good.

```python
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: As description
    @return: The index after partition
    """
    def partitionArray(self, nums, k):
        start, end = 0, len(nums) - 1
        while start <= end:
            while start <= end and nums[start] < k:
                start += 1
            while start <= end and nums[end] >= k:
                end -= 1
            if start <= end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        return start
```



### 608. Two Sum II - Input array is sorted

[LintCode](https://www.lintcode.com/problem/two-sum-ii-input-array-is-sorted/description)

##### Solution:

`start` and `end` go from the left and right side of the array. If their sum is equal to `target`, we found the solution. If sum is smaller than `target`, keep `end` still and move `start` to the right. If sum is greater, move `end` to the left.

```python
class Solution:
    """
    @param nums: an array of Integer
    @param target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, nums, target):
        # write your code here
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            if nums[start] + nums[end] == target:
                return [start + 1, end + 1]
            if nums[start] + nums[end] > target:
                end -= 1
            if nums[start] + nums[end] < target:
                start += 1
        if nums[start] + nums[end] == target:
            return [start + 1, end + 1]
```





### 585. 山脉序列中的最大值

[题目描述](https://www.lintcode.com/problem/maximum-number-in-mountain-sequence/description)

##### 思路:

- Define two pointers, `start` and `end`.
- While `start`is not to the next of `end`, define `mid`. Compare `mid` with its previous element, if `mid - 1 ` is greater, meaning the peak must be to the left of mid, move `end` to `mid`; otherwise move `start` to `mid`.
- The final state will be that `start` is to the next of `end`, and one of them is the peak. Return the larger one.

```python
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        start, end = 0, len(nums) - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid - 1] > nums[mid]:
                end = mid
            else:
                start = mid
        return max(nums[start], nums[end])
```



### 447. 在大数组中查找

[题目描述](https://www.lintcode.com/problem/search-in-a-big-sorted-array/description)

##### 思路：

利用倍增法的思路，因为现在不知道数组末尾的位置，我们就用一个指针从头开始，每次位置下标增大一倍，直到当前的值大于`target`，就意味着`target`的位置在当前下标位置和上一个下标位置之间。确定了范围之后用二分法查找即可。

```python
"""
Definition of ArrayReader
class ArrayReader(object):
    def get(self, index):
    	# return the number on given index, 
        # return 2147483647 if the index is invalid.
"""
class Solution:
    """
    @param: reader: An instance of ArrayReader.
    @param: target: An integer
    @return: An integer which is the first index of target.
    """
    def searchBigSortedArray(self, reader, target):
        # write your code here
        
        if reader.get(0) == target:
            return 0
        if reader.get(0) > target:
            return -1
        end = 1
        while reader.get(end) < target:
            end *= 2
        start = end // 2
        
        while start + 1 < end:
            mid = (start + end) // 2
            if reader.get(mid) > target:
                end = mid
            if reader.get(mid) < target:
                start = mid
            if reader.get(mid) == target:
                end = mid
        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        return -1
```



### 460. Find K Closest Elements

[题目描述](https://www.lintcode.com/problem/find-k-closest-elements/description)

##### 思路：

- Locate the first position called `left` that is greater than the `target` but less than or equal to `target` by the function `get_lower_bound`, which uses binary search.
- Define `right` as the next right position of `left`.
- Use `left` and `right` as two back-to-back pointers who go out in different directions. In each step, get the one that is closer to the target and append its value into the result list, then move the corresponding pointer one step further. We use a function `is_left_closer` to indicate that `left` is closer to the `target` if it returns True, `right` is closer if otherwise. Here is how the function works:
  - If `left`is less than 0, meaning that `left` is out of bound, return False.
  - if `right` exceeds the right bound, meaning that `right` is located at the last position of the given list, then of course `left` is closer no matter what numbers are to the left.
  - If both pointers are in bound, compare the values and return True if `left` is closer, False if otherwise.



```python
class Solution:
    """
    @param A: an integer array
    @param target: An integer
    @param k: An integer
    @return: an integer array
    """
    def kClosestNumbers(self, A, target, k):
        # write your code here
        left = self.get_lower_bound(A, target)
        right = left + 1
        ans = []
        for _ in range(k):
            if self.is_left_closer(A, target, left, right):
                ans.append(A[left])
                left -= 1
            else:
                ans.append(A[right])
                right += 1
        return ans
        
    def get_lower_bound(self, nums, target):
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > target:
                end = mid
            if nums[mid] <= target:
                start = mid
        if self.distance(nums, start, target) <= self.distance(nums, end, target):
            return start
        return end
        
    def distance(self, nums, index, target):
        return abs(nums[index] - target)
        
    def is_left_closer(self, A, target, left, right):
        if left < 0:
            return False
        if right > len(A) - 1:
            return True
        return self.distance(A, left, target) <= self.distance(A, right, target) 
```



### 428. Pow(x, n)

[题目描述](https://www.lintcode.com/problem/powx-n/description)

##### 思路：

First we only consider `n >= 0`. In this case we also need to consider the cases that n is even or odd.

If n is even, `pow(x, n) = pow(x*x, n/2)`. If n is odd, `pow(x, n) = pow(x, n - 1) * x`.

Now we consider the case that `n < 0`. In this case, we first make `n = -n`, asume n is positive and perform the steps described above. Then return `1/result`.

##### Recursion solution:

```python
class Solution:
    """
    @param x {float}: the base number
    @param n {int}: the power number
    @return {float}: the result
    """
    def myPow(self, x, n):
        # write your code here
        if n >= 0:
            return self.helper(x, n)
        if n < 0:
            n = -n
            return 1/self.helper(x, n)
    
    def helper(self, x, n):
        if n == 0: # base case 0
            return 1
        if n == 1: # base case 1
            return x
        if n % 2 == 0:
            return self.helper(x*x, n/2)
        else:
            return self.helper(x, n - 1) * x
```

##### Non-recursion solution:

```python
class Solution:
    """
    @param x {float}: the base number
    @param n {int}: the power number
    @return {float}: the result
    """
    def myPow(self, x, n):
        # write your code here
        if n < 0:
            x = 1/x
            n = - n
            
        ans = 1
        temp = x
        
        while n != 0:
            if n % 2 == 1:
                ans *= temp
            temp *= temp
            n //= 2
        return ans
```



### 159. Find Minimum in Rotated Sorted Array

[题目描述](https://www.lintcode.com/problem/find-minimum-in-rotated-sorted-array/description)

##### 思路：

The original array is sorted, and its rotated version is our input array. If we put the values into an x-y axis graph, say example `4 5 6 7 0 1 2`. It looks like this:

```
			7
		6
	5
4
---------------------------
						2
					1
				0
```

As can be seen from the graph, any value that is greater than `2`, the last number of the array, will be in the upper portion, which is not our interest. Any value that is less than `2` will be closer to our target. Use these two assumptions to keep shorten our searching range, until the `start` and `end` pointers are located next to each other, the smaller one of them is our answer.

```python
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        start, end = 0, len(nums) - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > nums[end]:
                start = mid
            else:
                end = mid
        return min(nums[start], nums[end])
```



### 140. Fast Power

[题目描述](https://www.lintcode.com/problem/fast-power/description)

##### 思路：

Similar to 428 Pow(x, n), use a faster way to calculate `a^n`, then modules `b`. Note that we can mod `b` in the middle steps to avoid overflow if `a` and `n` are very large numbers.

```python
class Solution:
    """
    @param a: A 32bit integer
    @param b: A 32bit integer
    @param n: A 32bit integer
    @return: An integer
    """
    def fastPower(self, a, b, n):
        # write your code here
        power_a = 1
        
        while n != 0:
            if n % 2 == 1:
                power_a *= a % b
            a *= a % b
            n //= 2
        return power_a % b
```



### 75. Find Peak Element

[题目描述](https://www.lintcode.com/problem/find-peak-element/description)

##### 思路：

- As stated, the second last element is greater than the last one, as a result the last element must not be a peak; and the second element is greater than the first one, i.e. the first element is not a peak as well. Consequently we set `start = 0, end = len(A) - 2`.
- Once we find a position that either side of it is greater, then we say we found a peak. Otherwise we go to the greater side to find. 
- If both sides are greater, we go to any side; note that in this case we cannot shorten the range to `start = mid - 1, end = mid + 1`, since this case indicates that we found a valey, if we shorten the range like that, we shall never find a peak. We can only pick one side.

```python
class Solution:
    """
    @param A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, A):
        # write your code here
        start, end = 1, len(A) - 2
        
        while start + 1 < end:
            mid = (start + end) // 2
            if A[mid] > A[mid + 1] and A[mid] > A[mid - 1]:
                return mid
            if A[mid] < A[mid + 1]:
                start = mid + 1
                continue
            if A[mid] < A[mid - 1]:
                end = mid - 1
                continue
        if A[start] > A[end]:
            return start
        return end
```



### 74. First Bad Version

[题目描述](https://www.lintcode.com/problem/first-bad-version/description)

思路：

Use binary search. 

```python
#class SVNRepo:
#    @classmethod
#    def isBadVersion(cls, id)
#        # Run unit tests to check whether verison `id` is a bad version
#        # return true if unit tests passed else false.
# You can use SVNRepo.isBadVersion(10) to check whether version 10 is a 
# bad version.
class Solution:
    """
    @param n: An integer
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        # write your code here
        start, end = 1, n
        while start + 1 < end:
            mid = (start + end) // 2
            if SVNRepo.isBadVersion(mid):
                end = mid
            else:
                start = mid
        if SVNRepo.isBadVersion(start):
            return start
        return end
```



### 62. Search in Rotated Sorted Array

[题目描述](https://www.lintcode.com/problem/first-bad-version/description)

##### 思路1：一次二分查找

If the value at `mid` position is greater than the value at `start` position, do a further comparison: is target in between `start` and `mid`, if yes, move `end` to `mid` position, then all elements within `start` and `end` are in an ascending order, continue the binary search we wil find the answer; if no, move `start` to `mid`. The same idea for the other part.

```python
class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """
    def search(self, A, target):
        # write your code here
        if not A:
            return -1
        start, end = 0, len(A) - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if A[mid] > A[start]:
                if A[mid] >= target >= A[start]:
                    end = mid
                else:
                    start = mid
            if A[mid] < A[end]:
                if A[mid] <= target <= A[end]:
                    start = mid
                else:
                    end = mid
            if A[mid] == target:
                return mid
        if A[start] == target:
            return start
        if A[end] == target:
            return end
        return -1
```



##### 思路2：两次二分查找

- First check is this array rotated, if not, call the binary search function and return the value.
- If the array is rotated, use the similar idea of question [159. Find Minimum in Rotated Sorted Array](https://www.lintcode.com/problem/find-minimum-in-rotated-sorted-array/description) to find the peak and valey location. Then perform binary search from the start to peak, and from valey to the end, to search the target.

```python
class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """
    def search(self, A, target):
        # write your code here
        ans = -1
        if not A:
            return ans
        start, end = 0, len(A) - 1
        
        # if the array is rotated 0 step, i.e. no ratation
        if A[start] < A[end]:
            return self.binary_search(A, target, start, end)
        else:
            while start + 1 < end:
                mid = (start + end) // 2
                if A[mid] > A[end]:
                    start = mid
                if A[mid] <= A[end]:
                    end = mid
            peak, valey = start, end
            
            search_left = self.binary_search(A, target, 0, peak)
            search_right = self.binary_search(A, target, valey, len(A) - 1)
            
            if search_left != -1:
                ans = search_left
            if search_right != -1:
                ans = search_right
            return ans
    
    def binary_search(self, A, target, start, end):
        while start + 1 < end:
            mid = (start + end) // 2
            if A[mid] == target:
                return mid
            if A[mid] < target:
                start = mid
            if A[mid] > target:
                end = mid
        if A[start] == target:
            return start
        if A[end] == target:
            return end
        return -1
```



### 900. Closest Binary Search Tree Value

[题目描述](https://www.lintcode.com/problem/closest-binary-search-tree-value/description)

##### 思路1：recursion

Find the lower and upper bound of the target, return the closest one.

How to find a lower bound:

- If `target <= root.val`, it means that the lower bound must be in the left sub-tree, just return `find_lower(root.left, target)`.
- If `target` is greater than root value, it means that current node is a lower bound candidate, we need to check if there is a node whose value is greater than current node value but is still smaller than the `target`, hence we let `lower = find_lower(root.right, target)` to see if there is a node that is closer to the lower bound.
  - If there is one, return it.
  - Otherwise, return current node.

##### 思路2：non-recursive

It uses the similar idea but in a non-recursive way.

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @return: the value in the BST that is closest to the target
    """
    def closestValue(self, root, target):
        # write your code here
        return self.recursion(root, target)
        #return self.non_recur(root, target)
        
    def non_recur(self, root, target):
        upper, lower = root, root
        while root:
            if target > root.val:
                lower = root
                root = root.right
            # use elif, instead of if. Since root might have
            # been changed above
            elif target < root.val: 
                upper = root
                root = root.left
            else:
                return root.val
        if abs(upper.val - target) <= abs(lower.val - target):
            return upper.val
        return lower.val
        
    def recursion(self, root, target):
        lower = self.find_lower(root, target)
        upper = self.find_upper(root, target)
        if not lower:
            return upper.val
        if not upper:
            return lower.val
        if abs(upper.val - target) <= abs(lower.val - target):
            return upper.val
        return lower.val
    
    def find_upper(self, root, target):
        if not root:
            return None
        
        if target >= root.val:
            return self.find_upper(root.right, target)
            
        upper = self.find_upper(root.left, target)
        
        return upper if upper else root
        
    def find_lower(self, root, target):
        if not root: 
            return None
        
        if target <= root.val:
            return self.find_lower(root.left, target)
        
        lower = self.find_lower(root.right, target)
        return lower if lower else root
```



### 596. Minimum Subtree

[题目描述](https://www.lintcode.com/problem/minimum-subtree/description)

##### 思路：

`node_min_sum` is the node with the minimum sum in the entire tree.

`min_sum` is the minimum value of sub-tree sum in the entire tree, i.e. it's the sum of `node_min_sum`.

`tree_sum` is the sum of current node.

The general idea is to compare the `min_sum` values of left sub-tree, right sub-tree and the current node, return the tuple with the mininum sum value.

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """
    def findSubtree(self, root):
        # write your code here
        node_min_sum, min_sum, tree_sum = self.find_min(root)
        return node_min_sum
        
    def find_min(self, root):
        if not root:
            return root, sys.maxsize, 0
            
        left, left_min, left_sum = self.find_min(root.left)
        right, right_min, right_sum = self.find_min(root.right)
        
        tree_sum = left_sum + right_sum + root.val
        
        if left_min == min(tree_sum, left_min, right_min):
            return left, left_min, tree_sum # output tree_sum not left_sum
        if right_min == min(tree_sum, left_min, right_min):
            return right, right_min, tree_sum
        return root, tree_sum, tree_sum
```



### 480. Binary Tree Paths

[题目描述](https://www.lintcode.com/problem/binary-tree-paths/description)

##### 思路1：

`path` is used to store current path, once the dfs traversal reaches a leaf, append this `path` to the `result` list, then pop an element from `path`.

For example:

```pseudocode
   1
 /   \
2     3
 \
  5
dfs(node 1, path = [1], result = []):
	go to left child - node 2 (not leaf):
		path.append(node 2) --> path = [1, 2]
		dfs(node 2, path, result):
            go to right child - node 5 (leaf):
			path.append(node 5) --> path = [1, 2, 5]
			dfs(node 5, path, result):
				result.append(serialized path '1->2->5')
			pop 5 from path --> path = [1, 2]
		pop 2 from path --> path = [1]
	go to right right - node 3 (leaf):
		path.append(node 3) --> path = [1, 3]
		dfs(node 3, path, result):
			result.append(serialized path '1->3')
		pop 3 from path --> path = [1]
result = ['1->2->5', '1->3']
```

```python
class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        if not root:
            return []
            
        ans = []
        self.dfs(root, [str(root.val)], ans)
        return ans
        
    def dfs(self, root, path, paths):
        if not root.left and not root.right:
            paths.append('->'.join(path))
            
        if root.left:
            path.append(str(root.left.val))
            self.dfs(root.left, path, paths)
            path.pop()
            
        if root.right:
            path.append(str(root.right.val))
            self.dfs(root.right, path, paths)
            path.pop()
```



##### 思路2：

Divide and conquer version DFS.

```python
class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        if not root:
            return []
            
        if not root.left and not root.right:
            return [str(root.val)] # use []
            
        paths = []
        #if root.left:
        for path in self.binaryTreePaths(root.left):
            paths.append(str(root.val) + '->' + path)
        
        #if root.right:
        for path in self.binaryTreePaths(root.right):
            paths.append(str(root.val) + '->' + path)
        
        return paths
```

Walk through the example above.

![](../img/q480_s2.jpg)



### 453. Flatten Binary Tree to Linked List

[题目描述](https://www.lintcode.com/problem/flatten-binary-tree-to-linked-list/description)

##### 思路：

divide: recursively flatten `root.left` and `root.right`, return the last node of `root.left` and last node of `root.right`. Let `root.right` be `left_last`'s right child, let `root`'s left child be its right child, finall reset `root.left` to `None`.

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        # write your code here
        if not root:
            return None
        
        left_last = self.flatten(root.left)
        right_last = self.flatten(root.right)
        
        if left_last:
            left_last.right = root.right
            root.right = root.left
            root.left = None
        
        if right_last:
            return right_last
            
        if left_last:
            return left_last
            
        return root
```



### 902. Kth Smallest Element in a BST

[LintCode](https://www.lintcode.com/problem/kth-smallest-element-in-a-bst/description), [LeetCode](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

##### Solution:

- Create a dummy tree node, whose right child is connected to the given root. The purpose of a dummy node here is to avoid a few lines of duplicate codes.
- Put `dummy` into a stack, then start our loop:
  - Pop a node from the stack, decrese the value of `k` by 1. If `k == 0`, then this node's value is the result.
  - Check if there is a right child of the popped node. If yes, push all it's right child's left descendances into the stack. 

```python
class Solution:
    """
    @param root: the given BST
    @param k: the given k
    @return: the kth smallest element in BST
    """
    def kthSmallest(self, root, k):
        dummy = TreeNode(-1)
        dummy.right = root
        stack = [dummy]
        k += 1
        
        while 1:
            node = stack.pop()
            k -= 1
            if k == 0:
                return node.val
            if node.right:
                root = node.right
                while root:
                    stack.append(root)
                    root = root.left
```



### 88. Lowest Common Ancestor of a Binary Tree

[LintCode](https://www.lintcode.com/problem/lowest-common-ancestor-of-a-binary-tree/description), [LeetCode](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

##### Solution:

Find Lowest Common Ancestor in left and right sub-trees. 

Cases:

- If the root is equal to `A`, `A` if the LCA.
- if `A` and `B` are in different sides, e.g. `A` is in left, `B` is in right, then the root is the LCA.
- If `A` and `B` are in the same sub-tree, find LCA in that sub-tree.

```python
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        # write your code here
        return self.helper(root, A, B)

    def helper(self, root, A, B):
        if not root:
            return None
            
        if root == A or root == B:
            return root
        
        left = self.helper(root.left, A, B)
        right = self.helper(root.right, A, B)
        
        if left and right:
            return root
        if left:
            return left
        if right:
            return right
        return None
```



### 578. Lowest Common Ancestor III

[LintCode](https://www.lintcode.com/problem/lowest-common-ancestor-iii/description)

In addition to previous LCA problem, there is one more condition: A or B may not exist in given tree.

##### Solution:

Add two more variables to the return values: `has_a` and `has_b`.

```python
class Solution:
    """
    @param: root: The root of the binary tree.
    @param: A: A TreeNode
    @param: B: A TreeNode
    @return: Return the LCA of the two nodes.
    """
    def lowestCommonAncestor3(self, root, A, B):
        # write your code here
        a, b, lca = self.helper(root, A, B)
        if a and b:
            return lca
        return None
        
    def helper(self, root, A, B):
        if not root:
            return False, False, None
            
        # left_node is lca if a and b are both in left sub-tree
        # if only a or b in left sub-tree, left_node is a or b
        # if neither, left_node is None
        left_has_a, left_has_b, left_node = self.helper(root.left, A, B)
        right_has_a, right_has_b, right_node = self.helper(root.right, A, B)
        
        a = left_has_a or right_has_a or root == A
        b = left_has_b or right_has_b or root == B
        
        if root == A or root == B:
            return a, b, root
            
        if left_node and right_node:
            return a, b, root
            
        if left_node:
            return a, b, left_node
        
        if right_node:
            return a, b, right_node
        return a, b, None
```



### 86. Binary Search Tree Iterator

[LintCode](https://www.lintcode.com/problem/binary-search-tree-iterator/description), [LeetCode](https://leetcode.com/problems/binary-search-tree-iterator/)

##### Solution:

Silimar to [902. Kth Smallest Element in a BST](https://www.lintcode.com/problem/kth-smallest-element-in-a-bst/description).

```python
class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        # do intialization if necessary
        
        self.dummy = TreeNode(None)
        self.dummy.right = root
        self.stack = [self.dummy]
        self.next()

    """
    @return: True if there has next node, or false
    """
    def hasNext(self, ):
        # write your code here
        return bool(self.stack)

    """
    @return: return next node
    """
    def next(self, ):
        # write your code here
        node = self.stack.pop()
        if node.right:
            root = node.right
            while root:
                self.stack.append(root)
                root = root.left
        return node
```





### 495. Implement Stack

[LintCode](https://www.lintcode.com/problem/implement-stack/description)

```python
class Stack:
    """
    @param: x: An integer
    @return: nothing
    """
    def __init__(self):
        self.data = []
        self.size = 0
    
    def push(self, x):
        if self.size == len(self.data) :
            self.data.append(x)
            self.size = len(self.data)
        else:
            self.data[self.size] = x
            self.size += 1
    """
    @return: nothing
    """
    def pop(self):
        if not self.isEmpty():
            self.size -= 1
            return self.data[self.size]
        
    """
    @return: An integer
    """
    def top(self):
        return self.data[self.size - 1]

    """
    @return: True if the stack is empty
    """
    def isEmpty(self):
        return self.size == 0
```

```python
class Stack:
    """
    @param: x: An integer
    @return: nothing
    """
    
    def __init__(self):
        self.stack = []
        
    def push(self, x):
        self.stack.append(x)

    """
    @return: nothing
    """
    def pop(self):
        self.stack.pop()

    """
    @return: An integer
    """
    def top(self):
        return self.stack[-1]

    """
    @return: True if the stack is empty
    """
    def isEmpty(self):
        return len(self.stack) == 0
```



### 494. Implement Stack by Two Queues

[LintCode](https://www.lintcode.com/problem/implement-stack-by-two-queues/description), [LeetCode](https://leetcode.com/problems/implement-stack-using-queues/)

[Approach 1](https://leetcode.com/problems/implement-stack-using-queues/solution/) here is a good illustration.

```python
from collections import deque
class Stack:
    """
    @param: x: An integer
    @return: nothing
    """
    def __init__(self):
        self.q1, self.q2 = deque(), deque()
    
    def push(self, x):
        # write your code here
        self.q1.append(x)

    """
    @return: nothing
    """
    def pop(self):
        # write your code here
        self.switch()
        self.q1, self.q2 = self.q2, self.q1

    """
    @return: An integer
    """
    def top(self):
        # write your code here
        self.switch()
        self.q2.append(self.q1[0])
        self.q1, self.q2 = self.q2, self.q1
        return self.q2.popleft()
        
            
    def switch(self):
        while self.q2:
            self.q2.popleft()
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        

    """
    @return: True if the stack is empty
    """
    def isEmpty(self):
        # write your code here
        return len(self.q1) == 0
        
```



### 224. Implement Three Stacks by Single Array

[LintCode](https://www.lintcode.com/problem/implement-three-stacks-by-single-array/description)

##### Solution: 

Use index 0, 1, 2 of the array to represent the top elements of three stacks. Use 3 linked list to store elements. The array stores the head node of each linked list.

```python
class Node:
    def __init__(self, value):
        self.val = value
        self.next = None
        #self.prev = None

class ThreeStacks:
    """
    @param: size: An integer
    """
    def __init__(self, size):
        # do intialization if necessary
        self.data = [None] * 3

    """
    @param: stackNum: An integer
    @param: value: An integer
    @return: nothing
    """
    def push(self, stackNum, value):
        # Push value into stackNum stack
        top = Node(value)
        top.next = self.data[stackNum]
        #top.next.prev = top
        self.data[stackNum] = top

    """
    @param: stackNum: An integer
    @return: the top element
    """
    def pop(self, stackNum):
        # Pop and return the top element from stackNum stack
        popped = self.data[stackNum]
        self.data[stackNum] = popped.next
        return popped.val

    """
    @param: stackNum: An integer
    @return: the top element
    """
    def peek(self, stackNum):
        # Return the top element
        return self.data[stackNum].val

    """
    @param: stackNum: An integer
    @return: true if the stack is empty else false
    """
    def isEmpty(self, stackNum):
        # write your code here
        return self.data[stackNum] == None
```



### 40. Implement Queue by Two Stacks

[LintCode](https://www.lintcode.com/problem/implement-queue-by-two-stacks/description), [LeetCode](https://leetcode.com/problems/implement-queue-using-stacks/)

##### Solution:

push: always push to stack `s1`.

pop: if `s2` is not empty, it means that the elements in `s1` has been moved to `s2`, and the `first-in` element is now at the `last-out` position in `s2`. Pop `s2` directly. 

top: same idea of pop, but return the last element in `s2`, instead of pop it.

```python
class MyQueue:
    
    def __init__(self):
        # do intialization if necessary
        self.s1, self.s2 = [], []

    """
    @param: element: An integer
    @return: nothing
    """
    def push(self, element):
        # write your code here
        self.s1.append(element)

    """
    @return: An integer
    """
    def pop(self):
        # write your code here
        if not self.s2:
            self.move_elements()
        return self.s2.pop()
        
    """
    @return: An integer
    """
    def top(self):
        # write your code here
        if not self.s2:
            self.move_elements()
        return self.s2[-1]

    def move_elements(self):
        while self.s1:
            self.s2.append(self.s1.pop())
            
    def switch(self):
        self.s1, self.s2 = self.s2, self.s1
```



### 955. Implement Queue by Circular Array

[LintCode](https://www.lintcode.com/problem/implement-queue-by-circular-array/description), [LeetCode](https://leetcode.com/problems/design-circular-queue/)

##### Solution:

Use `head` and `tail` to indicate the queue's front and end positions, respectively.

```python
class MyCircularQueue:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        """
        self.data = [0] * k
        self.size = 0
        self.head = 0
        self.tail = 0
        self.n = k
        

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        """
        if self.isFull(): 
            return False
        self.data[self.tail] = value
        self.tail = (self.tail + 1) % self.n
        self.size += 1
        return True

    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.n
        self.size -= 1
        return True

    def Front(self) -> int:
        """
        Get the front item from the queue.
        """
        if self.isEmpty():
            return -1
        return self.data[self.head]
        

    def Rear(self) -> int:
        """
        Get the last item from the queue.
        """
        if self.isEmpty():
            return -1
        return self.data[self.tail - 1]
        

    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return self.size == 0
        

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return self.size == self.n
        


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
```



### 128. Hash Function

[LintCode](https://www.lintcode.com/problem/hash-function/description)

##### Solution:

取模过程要使用同余定理：
(a * b ) % MOD = ((a % MOD) * (b % MOD)) % MOD

```python
class Solution:
    """
    @param key: A string you should hash
    @param HASH_SIZE: An integer
    @return: An integer
    """
    def hashCode(self, key, HASH_SIZE):
        ans = 0
        for char in key:
            ans = (ans*33 + ord(char)) % HASH_SIZE
        return ans
```



### 130. Heapify

[LintCode](https://www.lintcode.com/problem/heapify/description)

##### Solution:

- Perform `shiftdown` on all the father nodes (i.e. nodes with at least 1 child), from the last father node to the root.
- `shiftdown`:
  - Find the child (called `son`) with the minimum value between children.
  - If `son` is already greater than `father`, the tree rooted at `father` is already a heap.
  - If `son` is smaller than `father`, swap `son` and `father` and keep `shiftdown` until all nodes meet requirements.

```python
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # write your code here
        for i in reversed(range(len(A) // 2 )):
            self.shiftdown(A, i)
            
    def shiftdown(self, A, father):
        while father * 2 + 1 < len(A):
            son = father * 2 + 1
            if son + 1 < len(A) and A[son] > A[son + 1]:
                son += 1
            if A[son] > A[father]:
                break
            
            A[father], A[son] = A[son], A[father]
            father = son
```



### 657. Insert Delete GetRandom O(1)

[LintCode](https://www.lintcode.com/problem/insert-delete-getrandom-o1/description), [LeetCode](https://leetcode.com/problems/insert-delete-getrandom-o1/)

##### Solution:

- `val_to_index`: a dict stores value: index pairs.
- `data`: an array stores values.
- `insert`: append value to `data`, record its value and corresponding index in the array in the dict.
- `remove`: get the index of the target value from dict, overwrite this index by the last element in the array. Update the index of the element in dict. Finally delete the record of the target value in dict.
- `getRandom`: generate  a random number in the range of the array, return corresponding value.

```python
import random
class RandomizedSet:
    
    def __init__(self):
        # do intialization if necessary
        self.val_to_index = {}
        self.data = []

    """
    @param: val: a value to the set
    @return: true if the set did not already contain the specified element or false
    """
    def insert(self, val):
        # write your code here
        if self.has_value(val):
            return False
        self.data.append(val)
        self.val_to_index[val] = len(self.data) - 1
        return True

    """
    @param: val: a value from the set
    @return: true if the set contained the specified element or false
    """
    def remove(self, val):
        # write your code here
        if not self.has_value(val):
            return False
        index = self.val_to_index[val]
        last = self.data[-1]
        self.data[index] = last
        self.val_to_index[last] = index
        self.data.pop()
        del self.val_to_index[val]
        return True

    """
    @return: Get a random element from the set
    """
    def getRandom(self):
        # write your code here
        index = random.randint(0, len(self.data) - 1)
        return self.data[index]
        
    def has_value(self, val):
        return val in self.val_to_index


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param = obj.insert(val)
# param = obj.remove(val)
# param = obj.getRandom()
```



### 209. First Unique Character in a String

[LintCode](https://www.lintcode.com/problem/first-unique-character-in-a-string/description), [LeetCode](https://leetcode.com/problems/first-unique-character-in-a-string/)

##### Solution O(N^2):

Traverse the string twice. In the first traverse, use `dict` to record appearance of each character in the string. In the second traverse, return the first character whose appearance equals to 1.

##### Solution O(N):

Traverse the string only once. Treat the string as a datastream. Use a linked list to store characters that only appear once. If a character appears mote than once, delete it from the linked list. At the end, the first node in the linked list has the solution.

- Define a class `Node`, in which character and its previous node are stored.
- Define a class `Data_stream`, it reads in characters and construct the linked list. Global variables:
  - `dummyhead` points to the real head node of the linked list.
  - `tail` indicates the tail of the linked list.
  - `char_prev` stores the unique character and its previous node.
  - `duplicates` is a hash set, it stores the characters that have been read in.
  - `index` is the index of each character in the given string.



```python
class Node:
    def __init__(self, val = None, index = None, next_node = None):
        self.val = val
        self.index = index
        self.next = next_node

class Data_stream:
    def __init__(self):
        self.dummyhead = Node()
        self.tail = self.dummyhead
        self.char_prev = {}
        self.duplicates = set()
        self.index = -1
        
    def readin(self, ch):
        self.index += 1
        if ch in self.duplicates:
            if ch in self.char_prev:
                self.delete(ch)
        else:
            self.duplicates.add(ch)
            node = Node(ch, self.index)
            self.tail.next = node
            self.char_prev[ch] = self.tail
            self.tail = node
            
    def delete(self, ch):
        prev = self.char_prev[ch]
        node = prev.next
        next_node = node.next
        if not next_node: # node is tail
            self.tail = prev
        else: # node is not tail
            self.char_prev[next_node.val] = prev
        prev.next = next_node
        node.next = None
        del node
        del self.char_prev[ch]
        
class Solution:
    def firstUniqChar(self, s: str) -> int:
        data = Data_stream()
        for ch in s:
            data.readin(ch)
        return data.dummyhead.next.index if data.dummyhead.next else -1
```



### 685. First Unique Number in Data Stream

[LintCode](https://www.lintcode.com/problem/first-unique-number-in-data-stream/description)

##### Solution:

Same idea as [209. First Unique Character in a String](https://www.lintcode.com/problem/first-unique-character-in-a-string/description).

```python
class Node:
    def __init__(self, val = None, next_node = None):
        self.val = val
        self.next = next_node

class Solution:
    def __init__(self):
        self.dummyhead = Node()
        self.tail = self.dummyhead
        self.num_prev = {}
    
    def firstUniqueNumber(self, nums, number):
        found = False
        for num in nums:
            self.readin(num)
            if num == number:
                found = True
                break
        return self.dummyhead.next.val if self.dummyhead.next and found else -1
        
    def readin(self, num):
        if num in self.num_prev: # not unique
            if not self.num_prev[num]: # num appears > twice
                return
            self.delete(num)
            return
        # appears 1st time
        node = Node(num)
        self.tail.next = node
        self.num_prev[num] = self.tail
        self.tail = node
        
    def delete(self, num):
        prev = self.num_prev[num]
        node = prev.next
        next_node = node.next
        if next_node: # node is not tail
            self.num_prev[next_node.val] = prev
        else: # node is tail
            self.tail = prev
        prev.next = next_node
        node.next = None
        self.num_prev[num] = None
        del node
```



### 612. K Closest Points

[LintCode](https://www.lintcode.com/problem/k-closest-points/description)

##### Solution:

- Calculate distances.
- Push the tuple (distance, x, y) into a heap.
- Pop the first K tuples, the corresponding x and y values will be the solution

```python
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""
import heapq
class Solution:
    """
    @param points: a list of points
    @param origin: a point
    @param k: An integer
    @return: the k closest points
    """
    def kClosest(self, points, origin, k):
        # write your code here
        heap = []
        for point in points:
            dist = self.get_dist(point, origin)
            heapq.heappush(heap, (dist, point.x, point.y))
        ans = []
        for _ in range(k):
            _, x, y = heapq.heappop(heap)
            ans.append([x, y])
        return ans
        
    def get_dist(self, point, origin):
        return ((point.x - origin.x) ** 2 + (point.y - origin.y) ** 2) ** (0.5)
```



### 544. Top k Largest Numbers

[LintCode](https://www.lintcode.com/problem/top-k-largest-numbers/description)

##### Solution:

Use min-heap. Push numbers into the heap until the heap has k numbers. For the remaining numbers, compare them with the heap top (smallest number), if the number is greater than heap top, pop heap and push number; otherwise the number is of course not one of the top k largest numbers.

```python
import heapq
class Solution:
    """
    @param nums: an integer array
    @param k: An integer
    @return: the top k largest numbers in array
    """
    def topk(self, nums, k):
        # write your code here
        heap = []
        ans = []
        for num in nums:
            if len(heap) == k:
                heapq.heappushpop(heap, num)
            else:
                heapq.heappush(heap, num)
        while heap:
            ans.append(heapq.heappop(heap))
        ans.reverse()
        return ans
```



### 104. Merge K Sorted Lists

[LintCode](https://www.lintcode.com/problem/merge-k-sorted-lists/description)

##### Solution:

- Push all nodes into a min-heap.
- Reconstruct all nodes into a single linked list ordered by their values.
- Note: need to overwrite the comparison function of class `ListNode`, making a `ListNode` comparable by the heap.

```python
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
import heapq
ListNode.__lt__ = lambda x, y: (x.val < y.val)
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        # write your code here
        heap = []
        for node in lists:
            while node:
                heapq.heappush(heap, node)
                node = node.next
        dummyhead = ListNode(None)
        tail = dummyhead
        while heap:
            node = heapq.heappop(heap)
            tail.next = node
            tail = node
        return dummyhead.next
```





### 134. LRU Cache

[LintCode](https://www.lintcode.com/problem/lru-cache/description), [LeetCode](https://leetcode.com/problems/lru-cache/)

##### Solution:

- Define a class called `Node`, in which `key` and `value` pair is stored.
- Global variables:
  - `dummyhead` points to the real head node of the linked list.
  - `last_node` indicates the tail of the linked list.
  - `capacity`.
  - `size` is the current occupied capacity.
  - `key_to_prev`: key and previous node.
- `get(key)`. 
  - If `key` is in cache, return its value, and also move the corresponding node to the end. **DON'T** forget to update current node and its next node info in `key_to_prev`.
  - If it is not in cache, return -1.
- `put(key, value)`.
  - If `key` is in cache, update is value, and also move the corresponding node to the end. **DON'T** forget to update current node and its next node info in `key_to_prev`.
  - if `key` is not in cache, create new node, append it to the end.
    - If cache is full, remove the LRU.

```python
class Node():
    def __init__(self, key = None, val = None, next = None):
        self.key = key
        self.val = val
        self.next = next

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.dummyhead = Node()
        self.key_to_prev = {}
        self.last_node = self.dummyhead
        

    def get(self, key: int) -> int:
        # if key is not in cache
        if key not in self.key_to_prev:
            return -1
        # if key is in cache
        node = self.key_to_prev[key].next
        # if node is the last node
        if node == self.last_node:
            return node.val
        # if node is not the last node
        self.move_to_end(node)
        return node.val
    
    def put(self, key: int, value: int) -> None:
        # if node is in cache
        if key in self.key_to_prev:
            node = self.key_to_prev[key].next
            # if node is the last node
            if node == self.last_node:
                node.val = value
                return
            # if node is not the last node
            self.move_to_end(node)
            node.val = value
            return
        
        # if node is not in cache
        node = Node(key, value, None)
        #   if cache is not full
        if self.size < self.capacity:
            self.push_back(node)
            self.size += 1
            return
        #   if cache is full
        LRU_node = self.dummyhead.next
        self.remove_node(LRU_node)
        # delete LRU_node info in dict and delete the node
        del self.key_to_prev[LRU_node.key]
        del LRU_node
        self.push_back(node)
        
    def remove_node(self, node):
        # get prev and next nodes
        prev_node = self.key_to_prev[node.key]
        next_node = node.next
        prev_node.next = next_node
        # update dict info for node and next_node
        if node != self.last_node:
            self.key_to_prev[next_node.key] = prev_node
    
    def move_to_end(self, node):
        self.remove_node(node)
        self.push_back(node)
        
    def push_back(self, node):
        self.last_node.next = node
        #node.next = None
        self.key_to_prev[node.key] = self.last_node
        self.last_node = node # update last_node

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

Useful function to debug:

```python
def print_node(self):
    for key, node in self.key_to_prev.items():
        print(key, ':', '(', node.key, node.val , ')')
    linkedlist = '\n dummy ->'
    node = self.dummyhead.next
    while  node:
        linkedlist += '(' + str(node.key) + ', ' + str(node.val) + ')' + ' -> '
        node = node.next
    print(linkedlist)
    print('last_node:', '(', self.last_node.key, self.last_node.val , ')')
```

Solution 九章算法：

```python
class LinkedNode:
    
    def __init__(self, key=None, value=None, next=None):
        self.key = key
        self.value = value
        self.next = next

class LRUCache:

    # @param capacity, an integer
    def __init__(self, capacity):
        self.key_to_prev = {}
        self.dummy = LinkedNode()
        self.tail = self.dummy
        self.capacity = capacity
    
    def push_back(self, node):
        self.key_to_prev[node.key] = self.tail
        self.tail.next = node
        self.tail = node
    
    def pop_front(self):
        # 删除头部
        head = self.dummy.next
        del self.key_to_prev[head.key]
        self.dummy.next = head.next
        self.key_to_prev[head.next.key] = self.dummy
        
    # change "prev->node->next...->tail"
    # to "prev->next->...->tail->node"
    def kick(self, prev):    #将数据移动至尾部
        node = prev.next
        if node == self.tail:
            return
        prev.next = node.next
        if node.next is not None:
            self.key_to_prev[node.next.key] = prev
            node.next = None
        self.push_back(node)

    # @return an integer
    def get(self, key):		#获取数据
        if key not in self.key_to_prev:
            return -1
        self.kick(self.key_to_prev[key])
        return self.key_to_prev[key].next.value

    # @param key, an integer
    # @param value, an integer
    # @return nothing
    def set(self, key, value):     #数据放入缓存
        if key in self.key_to_prev:	   
            self.kick(self.key_to_prev[key])
            self.key_to_prev[key].next.value = value
        else:
            self.push_back(LinkedNode(key, value))  #如果key不存在，则存入新节点
            if len(self.key_to_prev) > self.capacity:		#如果缓存超出上限
                self.pop_front()					#删除头部
```



### 680. Split String

[LintCode](https://www.lintcode.com/problem/split-string/description)

##### Solution:

Use DFS, Each recursion delete 1 or 2 char(s).

```python
class Solution:
    """
    @param: : a string to be split
    @return: all possible split string array
    """

    def splitString(self, s):
        ans = []
        self.dfs(s, [], ans)
        return ans
        
    def dfs(self, s, path, ans):
        if s == '':
            ans.append(path[:]) # important: path[:] for deep copy
            return
        for i in range(2):
            if i + 1 <= len(s):
                path.append(s[:i + 1])
                self.dfs(s[i + 1 :], path, ans)
                path.pop()
```



### 425. Letter Combinations of a Phone Number

[LintCode](https://www.lintcode.com/problem/letter-combinations-of-a-phone-number/description)

##### Solution:

`index` is used to traverse a number's corresponding letters. For example,

```python
for letter in KEYBOARD[digits[index]]:
```

Add this letter into current path, then go to the next recursion call, in which the one of the next digit's corresponding letters will be added to the path. Repeat this process until the program reaches the base condition that `index` is out of the range.

```python
KEYBOARD = {
    '2': 'abc',
    '3': 'def',
    '4': 'ghi',
    '5': 'jkl',
    '6': 'mno',
    '7': 'pqrs',
    '8': 'tuv',
    '9': 'wxyz',
}

class Solution:
    """
    @param digits: A digital string
    @return: all posible letter combinations
    """
    def letterCombinations(self, digits):
        if not digits:
            return []
        ans = []
        self.dfs(digits, 0, '', ans)
        return ans
    
    def dfs(self, digits, index, path, ans):
        if index == len(digits):
            ans.append(path[:])
            return
        
        for letter in KEYBOARD[digits[index]]:
            self.dfs(digits, index + 1, path + letter, ans)
```



### 153. Combination Sum II

[LintCode](https://www.lintcode.com/problem/combination-sum-ii/description), [LeetCode](https://leetcode.com/problems/combination-sum-ii/)

##### Solution:

This solution [post](https://leetcode-cn.com/problems/combination-sum-ii/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-3/) provides an excellent explanation.

```python
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, num, target):
        if not num:
            return []
        ans = []
        num.sort()
        self.dfs(num, target, [], ans, 0, 0)
        return ans
        
    def dfs(self, num, target, path, ans, cumulated, index):
        if cumulated == target:
            ans.append(path[:])
            return
        for i in range(index, len(num)):
            if cumulated > target:
                break
            if i > index and num[i] == num[i - 1]:
                continue
            path.append(num[i])
            self.dfs(num, target, path, ans, cumulated + num[i], i + 1)
            path.pop()
```



### 135. Combination Sum

[LintCode](https://www.lintcode.com/problem/combination-sum/description), [LeetCode](https://leetcode.com/problems/combination-sum)

##### Solution:

This solution [post](https://leetcode-cn.com/problems/combination-sum/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-2/) provides an excellent explanation.

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        if not candidates:
            return []
        ans = []
        candidates.sort()
        self.dfs(candidates, target, [], ans, 0, 0)
        return ans
        
    def dfs(self, candidates, target, path, ans, cumulated, start):
        if cumulated == target:
            ans.append(path[:])
            return
        for i in range(start, len(candidates)):
            if cumulated > target:
                break
            path.append(candidates[i])
            self.dfs(candidates, target, path, ans, cumulated + candidates[i], i)
            path.pop()
```


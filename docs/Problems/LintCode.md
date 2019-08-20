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



### 137. 克隆图

[题目描述](https://www.lintcode.com/problem/clone-graph/description)

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

```python
class Solution:
    def maxDepth(self, root):
        if root is None:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```



### 93. 平衡二叉树

[题目描述](https://www.lintcode.com/problem/balanced-binary-tree/description)

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

利用BST的性质：对BST进行in-order traversal，将得到一个sorted array。中序遍历，如果当前节点值大于上一个节点值，则True；否侧False。

```

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


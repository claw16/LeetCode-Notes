# LintCode Problems

### 415 有效回文串

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
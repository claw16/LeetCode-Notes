# Time Complexity

Aka Time efficiency.

常见的时间复杂度：

| 复杂度      | 可能对应的算法                                               | 备注                           |
| ----------- | ------------------------------------------------------------ | ------------------------------ |
| O(1)        | 位运算                                                       | 常数级复杂度，一般面试中不会有 |
| O(logn)     | 二分法，倍增法，快速幂算法，辗转相除法                       |                                |
| O(n)        | 枚举法，双指针算法，单调栈算法，KMP算法，Rabin Karp，Manacher's Algorithm | 又称作线性时间复杂度           |
| O(nlogn)    | 快速排序，归并排序，堆排序                                   |                                |
| O(n^2)O(n2) | 枚举法，动态规划，Dijkstra                                   |                                |
| O(n^3)      | 枚举法，动态规划，Floyd                                      |                                |
| O(2^n)      | 与组合有关的搜索问题                                         |                                |
| O(n!)       | 与排列有关的搜索问题                                         |                                |



### Time Efficiency for Recursive Functions

参考:

- CSC 3110 Textbook ch 2.4, page 70
- This [post](https://yourbasic.org/algorithms/time-complexity-recursive-functions/)

#### Master Theorem

$a \ge 1$, $b > 1$ are constants, let f(n) be a function, and let T(n) be a function over the positive numbers defined by the recurrence:

> T(n) = aT(n/b) + f(n)

If f(n) = O ($n^d$), where $d \ge 0$, then

* $T(n) = O(n^d)$, if $a<b^d$,
* $T(n) = O(n^d \log n)$, if $a=b^d$,
* $T(n) = O(n^{\log_ba})$, if $a>b^d$,



#### Example 

```python
T(n) = T(n/2) + O(1)
     = T(n/4) + O(1) * 2
     = T(n/8) + O(1) * 3
     = T(n/16) + O(1) * 4
     ...
     = T(1) + O(1) * logn # when n=1, there are logn times expansion
     = O(logn)
```

#### Ex 1

有一个算法大致结构如下：

```
while (n > 1) {
   这里执行一个使用 O(n) 的算法，将 n 的规模缩小一半
   n = n / 2
}
```

问该算法的时间复杂度

根据算法，我们可以用`T函数推导法`写出公式：T(n) = T(n/2) + O(n)T(n)=T(n/2)+O(n)

推导过程如下：

```
T(n) = T(n/2) + O(n)
     = T(n/4) + O(n/2) + O(n)
     = T(n/8) + O(n/4) + O(n/2) + O(n)
     = ...
     = O(1) + O(2) + ... O(n/2) + O(n)
     = O(1 + 2 + 4 .. + n/2 + n)
     = O(2n) = O(n)
```

许多同学会拍脑袋认为这个式子的结果是 O(nlogn)，这是错误的。主要错在，当 T(n/2) 往下继续展开的时候，很多同学直接写成 T(n/4) + O(n)，这是不对的。应该是 T(n/4) + O(n/2)。这里我们暂时不能约掉 O(n/2) 里的 `/2`。因为会导致误差累积。

另外一个需要记住的结论就是：`O(1 + 2 + 4 ... + n/2 + n) = O(n)`。这个结论可以通过简单的将 n = 1024 带入计算可以得到：

```
1 + 2 + 4 + ... + 1024 = 2047 ~ 2 * 1024 = O(2n) = O(n)
```

这是一道小学数学题，给整个式子 + 1，然后两个1就变成了一个2，两个2就变成了一个4，以此类推。

#### Ex 2

Find time efficiency of merge sort.

参考： 3110 textbook ch 5.1, page 172


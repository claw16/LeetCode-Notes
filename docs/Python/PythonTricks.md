# Python Tricks

## String

### join()

将序列中的元素以指定的字符连接生成一个新的字符串。

> `str.join(sequence)`

##### Example

```python
str = "-";
seq = ("a", "b", "c"); # 字符串序列
print str.join( seq );
```

output:

```
a-b-c
```



### isalnum()

检测字符串是否由字母和数字组成。如果 string 至少有一个字符并且所有字符都是字母或数字则返回 True,否则返回 False。

结合`join()`举例：

##### Example

```python
string = "A man, a plan, a canal: Panama"
string = ''.join(char for char in string if char.isalnum()).lower()
print(string)
```

output:

```
amanaplanacanalpanama
```

解释：如果`char`通过`isalnum()`检测结果是字符，则用空字符连接；否则，跳过当前的`char`。上面的方法可以简单的去处字符串中所有的非数字和字母的字符，并且将所有字符转换为小写。



### String to Char Array: list()

For example:

```
Input : Word
Output : ['W', 'o', 'r', 'd']
```

```python
list('word')
# ['w', 'o', 'r', 'd']
```



## Simplified Coding

### bool()

Convert a given variable to a boolean value.

For example

```python
stack = []
if stack:
	return True
```

Can be simplified as:

```python
return bool(stack)
```



### dict.get(key, default = None)

Return value for `key`, if `key` is not in `dict`, return `default` value.

For example:

```python
for char in string:
	if char in dict:
        dict[char] += 1
    else:
        dict[char] = 1
```

Can be simplified as:

```python
for char in string:
    dict[char] = dict.get(char, 0) + 1
```



## List

### Reverse a list

This [post](https://dbader.org/blog/python-reverse-list) introduces 3 methods to reverse a list.


# Dynamic Programming

## Backpack Problems

Given a backpack with a fixed capacity and a set of items with corresponding volumes and values. Put as many items into the backpack as possible based on different scenarios (usually it is required to put in items with maximized value).

### 01 Backpack

In this case, items are either selected or not, each item can be selected only once. 

##### Solution idea:

Use array to record the maximum value for capacity `j` of first `i` items, i.e. reduce the problem size into a smaller sub-problem with capacity `j` and fewer items.

##### Example:

capacity `m = 10`, item size: `A = [2, 3, 5, 7]`, item value: `V = [1, 5, 2, 4]`.

| i, j | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 1    | 0    | 0    | 1    | 1    | 1    | 1    | 1    | 1    | 1    | 1    | 1    |
| 2    | 0    | 0    | 1    | 5    | 5    | 6    | 6    | 6    | 6    | 6    | 6    |
| 3    | 0    | 0    | 1    | 5    | 5    | 6    | 6    | 6    | 7    | 7    | 8    |
| 4    | 0    | 0    | 1    | 5    | 5    | 6    | 6    | 6    | 7    | 7    | 9    |

Use `dp[i][j]` to represent the maximum value that the backpack has `j` capacity and there are only the first `i` items available.

If we DO select the `ith` item, `dp[i][j] = dp[i-1][j-A[i]] + V[i]`, where `j > A[i]`.

If we DO NOT select the `ith` item, `dp[i][j] = dp[i-1][j]`.

As a result, `dp[i][j] = max(dp[i-1][j], dp[i-1][j-A[i]] + V[i])`.


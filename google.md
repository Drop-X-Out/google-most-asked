# DSA Problem Solutions

## Google Interview 2024

1. [Find All Triplets with Zero Sum](#1-find-all-triplets-with-zero-sum)
2. [Generate All Binary Strings from Given Pattern](#2-generate-all-binary-strings-from-given-pattern)
3. [Count of Strings that Can be Formed Using a, b and c Under Given Constraints](#3-count-of-strings-that-can-be-formed-using-a-b-and-c-under-given-constraints)
4. [Find Largest Word in Dictionary by Deleting Some Characters of Given String](#4-find-largest-word-in-dictionary-by-deleting-some-characters-of-given-string)
5. [Find Subarray with Given Sum (Nonnegative Numbers)](#5-find-subarray-with-given-sum-nonnegative-numbers)
6. [Find the Longest Substring with K Unique Characters](#6-find-the-longest-substring-with-k-unique-characters)
7. [Find Two Non-Repeating Elements in an Array](#7-find-two-non-repeating-elements-in-an-array)
8. [Flood Fill Algorithm](#8-flood-fill-algorithm)
9. [Meta Strings](#9-meta-strings)
10. [Print All Jumping Numbers Smaller Than or Equal to a Given Value](#10-print-all-jumping-numbers-smaller-than-or-equal-to-a-given-value)
11. [Sum of All Numbers Formed from Root to Leaf Paths](#11-sum-of-all-numbers-formed-from-root-to-leaf-paths)
12. [The Celebrity Problem](#12-the-celebrity-problem)
13. [Unbounded Knapsack (Repetition of Items Allowed)](#13-unbounded-knapsack-repetition-of-items-allowed)
14. [Sudoku Solver](#14-sudoku-solver)
15. [Boggle Using Trie](#15-boggle-using-trie)
16. [Check if a Binary Tree Contains Duplicate Subtrees](#16-check-if-a-binary-tree-contains-duplicate-subtrees)
17. [Egg Dropping Puzzle](#17-egg-dropping-puzzle)

---

## 1. Find All Triplets with Zero Sum
- **Explanation**: Find all unique triplets in an array that sum to zero. Uses sorting and two-pointer technique for efficiency.
- **Approach**: Sorting and Two-pointer technique.
- **Solution**:
```python
def find_triplets(arr):
    arr.sort()
    result = []
    for i in range(len(arr) - 2):
        left, right = i + 1, len(arr) - 1
        while left < right:
            total = arr[i] + arr[left] + arr[right]
            if total == 0:
                result.append((arr[i], arr[left], arr[right]))
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result
```

## 2. Generate All Binary Strings from Given Pattern
- **Explanation**: Generate all possible binary strings by replacing '?' in a binary pattern with 0 and 1.
- **Approach**: Backtracking to replace all '?'.
- **Solution**:
```python
def generate_binary_strings(pattern):
    def backtrack(index, current):
        if index == len(pattern):
            results.append(current)
            return
        if pattern[index] == '?':
            backtrack(index + 1, current + '0')
            backtrack(index + 1, current + '1')
        else:
            backtrack(index + 1, current + pattern[index])
    
    results = []
    backtrack(0, '')
    return results
```

## 3. Count of Strings that Can be Formed Using a, b and c Under Given Constraints
- **Explanation**: Count valid strings formed with characters a, b, and c, with constraints on b and c's appearances.
- **Approach**: Dynamic Programming.
- **Solution**:
```python
def count_strings(n):
    dp = [[0] * 3 for _ in range(n + 1)]
    dp[0] = [1, 1, 1]  # base case

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][2]  # 'a'
        dp[i][1] = dp[i - 1][0]  # 'b'
        dp[i][2] = dp[i - 1][0] + dp[i - 1][1]  # 'c'

    return dp[n][0] + dp[n][1] + dp[n][2]
```

## 4. Find Largest Word in Dictionary by Deleting Some Characters of Given String
- **Explanation**: From a string and a dictionary, find the largest word that can be formed by deleting characters from the string.
- **Approach**: Two-pointer technique to check subsequences.
- **Solution**:
```python
def largest_word(string, dictionary):
    def is_subsequence(word):
        it = iter(string)
        return all(char in it for char in word)

    largest = ''
    for word in dictionary:
        if is_subsequence(word) and len(word) > len(largest):
            largest = word
    return largest
```

## 5. Find Subarray with Given Sum (Nonnegative Numbers)
- **Explanation**: Find a subarray with a specific sum in an array of nonnegative numbers.
- **Approach**: Sliding window technique.
- **Solution**:
```python
def find_subarray_with_sum(arr, target):
    current_sum = 0
    start = 0

    for end in range(len(arr)):
        current_sum += arr[end]

        while current_sum > target:
            current_sum -= arr[start]
            start += 1

        if current_sum == target:
            return arr[start:end + 1]
    return []
```

## 6. Find the Longest Substring with K Unique Characters
- **Explanation**: Identify the longest substring that contains exactly K unique characters.
- **Approach**: Sliding window with hash map for character frequencies.
- **Solution**:
```python
def longest_substring_with_k_unique(s, k):
    char_count = {}
    start, max_length = 0, 0

    for end in range(len(s)):
        char_count[s[end]] = char_count.get(s[end], 0) + 1
        
        while len(char_count) > k:
            char_count[s[start]] -= 1
            if char_count[s[start]] == 0:
                del char_count[s[start]]
            start += 1

        max_length = max(max_length, end - start + 1)
    return max_length
```

## 7. Find Two Non-Repeating Elements in an Array
- **Explanation**: In an array where all elements appear twice except for two, identify the unique elements.
- **Approach**: XOR-based technique.
- **Solution**:
```python
def find_two_non_repeating(arr):
    xor_sum = 0
    for num in arr:
        xor_sum ^= num
    
    # Find rightmost set bit
    set_bit = xor_sum & -xor_sum
    first, second = 0, 0

    for num in arr:
        if num & set_bit:
            first ^= num
        else:
            second ^= num

    return first, second
```

## 8. Flood Fill Algorithm
- **Explanation**: Implement the flood fill algorithm, commonly used in paint applications.
- **Approach**: DFS or BFS to fill connected components.
- **Solution**:
```python
def flood_fill(image, sr, sc, new_color):
    original_color = image[sr][sc]
    if original_color == new_color:
        return image
    
    def fill(r, c):
        if image[r][c] == original_color:
            image[r][c] = new_color
            if r >= 1: fill(r - 1, c)  # up
            if r + 1 < len(image): fill(r + 1, c)  # down
            if c >= 1: fill(r, c - 1)  # left
            if c + 1 < len(image[0]): fill(r, c + 1)  # right
    
    fill(sr, sc)
    return image
```

## 9. Meta Strings
- **Explanation**: Check if two strings can be made identical by swapping exactly one pair of characters.
- **Approach**: Count mismatches and evaluate swap feasibility.
- **Solution**:
```python
def are_meta_strings(s1, s2):
    if len(s1) != len(s2):
        return False
    
    mismatches = []
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            mismatches.append((s1[i], s2[i]))
    
    return len(mismatches) == 2 and mismatches[0] == mismatches[1][::-1]
```

## 10. Print All Jumping Numbers Smaller Than or Equal to a Given Value
- **Explanation**: Print all jumping numbers up to a given limit, where adjacent digits differ by 1.
- **Approach**: BFS or DFS to generate valid numbers.
- **Solution**:
```python
def print_jumping_numbers(n):
    for num in range(10):
        dfs(num, n)

def dfs(num,

 n):
    if num > n:
        return
    print(num)
    last_digit = num % 10
    if last_digit > 0:
        dfs(num * 10 + last_digit - 1, n)
    if last_digit < 9:
        dfs(num * 10 + last_digit + 1, n)
```

## 11. Sum of All Numbers Formed from Root to Leaf Paths
- **Explanation**: Calculate the sum of all numbers formed by root-to-leaf paths in a binary tree.
- **Approach**: DFS to compute path sums.
- **Solution**:
```python
def sum_numbers(root):
    def dfs(node, current_sum):
        if not node:
            return 0
        current_sum = current_sum * 10 + node.val
        if not node.left and not node.right:
            return current_sum
        return dfs(node.left, current_sum) + dfs(node.right, current_sum)

    return dfs(root, 0)
```

## 12. The Celebrity Problem
- **Explanation**: In a matrix where M[i][j] = 1 means person i knows person j, find the celebrity (known by everyone but knows no one).
- **Approach**: Stack-based method to identify non-celebrities.
- **Solution**:
```python
def find_celebrity(M):
    n = len(M)
    stack = list(range(n))

    while len(stack) > 1:
        a = stack.pop()
        b = stack.pop()
        if M[a][b] == 1:
            stack.append(b)
        else:
            stack.append(a)

    candidate = stack[0]
    for i in range(n):
        if i != candidate and (M[candidate][i] == 1 or M[i][candidate] == 0):
            return -1
    return candidate
```

## 13. Unbounded Knapsack (Repetition of Items Allowed)
- **Explanation**: Maximize value in a knapsack with unlimited quantities of items without exceeding weight.
- **Approach**: Dynamic Programming.
- **Solution**:
```python
def unbounded_knapsack(wt, val, W):
    dp = [0] * (W + 1)
    for i in range(1, W + 1):
        for j in range(len(wt)):
            if wt[j] <= i:
                dp[i] = max(dp[i], dp[i - wt[j]] + val[j])
    return dp[W]
```

## 14. Sudoku Solver
- **Explanation**: Solve a 9x9 Sudoku puzzle using backtracking.
- **Approach**: Backtracking to fill cells while adhering to Sudoku rules.
- **Solution**:
```python
def solve_sudoku(board):
    def is_valid(board, r, c, num):
        for i in range(9):
            if board[i][c] == num or board[r][i] == num:
                return False
        box_row, box_col = 3 * (r // 3), 3 * (c // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        return True

    def solve(board):
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    for num in '123456789':
                        if is_valid(board, r, c, num):
                            board[r][c] = num
                            if solve(board):
                                return True
                            board[r][c] = '.'
                    return False
        return True

    solve(board)
```

## 15. Boggle Using Trie
- **Explanation**: Find all possible words in a Boggle board using a Trie for efficient dictionary storage.
- **Approach**: Trie and DFS on the board.
- **Solution**:
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

def build_trie(words):
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    return root

def find_words(board, trie):
    def dfs(r, c, node, path, result):
        if node.is_end_of_word:
            result.add(path)
            node.is_end_of_word = False  # Avoid duplicates
        
        if not (0 <= r < len(board) and 0 <= c < len(board[0])):
            return
        
        temp = board[r][c]
        if temp not in node.children:
            return
        
        board[r][c] = '#'  # Mark as visited
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            dfs(r + dr, c + dc, node.children[temp], path + temp, result)
        board[r][c] = temp  # Restore original value

    result = set()
    for r in range(len(board)):
        for c in range(len(board[0])):
            dfs(r, c, trie, '', result)
    return result
```

## 16. Check if a Binary Tree Contains Duplicate Subtrees
- **Explanation**: Determine if a binary tree has duplicate subtrees of size 2 or more.
- **Approach**: Post-order traversal with a hash set to identify duplicates.
- **Solution**:
```python
def check_duplicate_subtrees(root):
    def serialize(node):
        if not node:
            return '#'
        serial = f"{node.val},{serialize(node.left)},{serialize(node.right)}"
        if serial in seen:
            duplicates.add(serial)
        seen.add(serial)
        return serial

    seen = set()
    duplicates = set()
    serialize(root)
    return duplicates
```

## 17. Egg Dropping Puzzle
- **Explanation**: Calculate the minimum attempts needed to find the critical floor from which eggs break.
- **Approach**: Dynamic Programming.
- **Solution**:
```python
def egg_drop(eggs, floors):
    dp = [[0] * (floors + 1) for _ in range(eggs + 1)]

    for i in range(1, eggs + 1):
        dp[i][0] = 0
        dp[i][1] = 1

    for j in range(1, floors + 1):
        dp[1][j] = j  # Only one egg

    for i in range(2, eggs + 1):
        for j in range(2, floors + 1):
            dp[i][j] = float('inf')
            for x in range(1, j + 1):
                res = 1 + max(dp[i - 1][x - 1], dp[i][j - x])
                dp[i][j] = min(dp[i][j], res)

    return dp[eggs][floors]
```



########################################
# Two Sum
########################################

"""
Two Sum: Given an array of integers, can you find the two numbers such that
they add up to a specific target?
"""


def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(result)

########################################
# Reverse Integer
########################################

"""
Reverse Integer: How would you reverse the digits of a given integer?

"""


def reverse_integer(num):
    # Convert the integer to a string
    num_str = str(num)

    # Reverse the string using slicing with a step of -1
    reversed_str = num_str[::-1]

    # Convert the reversed string back to an integer
    reversed_num = int(reversed_str)

    return reversed_num


num = 12345
reversed_num = reverse_integer(num)
print(reversed_num)

########################################
# Palindrome Number
########################################

"""
Can you determine if a given integer is a palindrome without converting the 
integer to a string?
"""


def is_palindrome(x):
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    revertedNumber = 0
    while x > revertedNumber:
        revertedNumber = revertedNumber * 10 + x % 10
        x //= 10
    return x == revertedNumber or x == revertedNumber // 10


# Call the function
print(is_palindrome(121))  # Prints True
print(is_palindrome(-121))  # Prints False
print(is_palindrome(10))  # Prints False

########################################
# Roman to Integer
########################################
"""
Roman to Integer: How can you convert a Roman numeral to an integer?

"""


def roman_to_int(s: str) -> int:
    # Define the dictionary to map Roman numerals to integers.
    roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500,
        'M': 1000}
    total = 0
    prev_value = 0

    # Traverse the Roman numeral string from right to left.
    for roman_digit in reversed(s):
        current_value = roman_dict[roman_digit]

        # If the current value is less than the previous value,
        # subtract it from the total. Otherwise, add it.
        if current_value < prev_value:
            total -= current_value
        else:
            total += current_value

        prev_value = current_value

    return total


assert roman_to_int('XVII') == 17
assert roman_to_int('IX') == 9
assert roman_to_int('MCMIV') == 1904

########################################
# Valid Parentheses
########################################

"""
Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
 can you determine if the input string is valid?
"""


def valid_parentheses(s: str) -> bool:
    bracket_map = {"(": ")", "[": "]", "{": "}"}
    open_par = set(["(", "[", "{"])
    stack = []
    for i in s:
        if i in open_par:
            stack.append(i)
        elif stack and i == bracket_map[stack[-1]]:
            stack.pop()
        else:
            return False
    return stack == []


########################################
# Merge Two Sorted Lists
########################################

"""
How would you merge two sorted linked lists and return it as a new sorted list?
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def merge_two_Lists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = cur = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next


########################################
# Remove Duplicates from Sorted Array
########################################
"""
Given a sorted array, can you remove the duplicates in-place and return
 the new length?
"""


def remove_duplicates(nums) -> int:
    if len(nums) == 0:
        return 0
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1


########################################
# Longest Common Prefix
########################################
"""
How would you find the longest common prefix string amongst an array of strings?
"""


def longest_commonPrefix(strs) -> str:
    prefix = ""
    for z in zip(*strs):
        if len(set(z)) == 1:
            prefix += z[0]
        else:
            break
    return prefix


########################################
# Valid Sudoku
########################################

"""
Can you determine if a given 9x9 Sudoku board is valid?
"""


def is_valid_sudoku(board):
    for i in range(9):
        row = board[i]
        if not is_valid(row):
            return False
        col = [board[j][i] for j in range(9)]
        if not is_valid(col):
            return False
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            square = [board[x][y] for x in range(i, i + 3) for y in
                      range(j, j + 3)]
            if not is_valid(square):
                return False
    return True


def is_valid(nums):
    nums = [i for i in nums if i != '.']
    return len(nums) == len(set(nums))


########################################
# Container With Most Water
########################################
"""
 How would you find two lines, which together with the x-axis forms a 
 container, such that the container contains the most water?
"""


def max_area(height):
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        h = min(height[left], height[right])
        w = right - left
        max_area = max(max_area, h * w)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area


########################################
# Remove Nth Node From End of List
########################################
"""
Can you remove the nth node from the end of the list and return its head?
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def removeNthFromEnd(head, n):
    fast = slow = head
    for _ in range(n):
        fast = fast.next
    if not fast:
        return head.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head


########################################
# Generate Parentheses
########################################

"""
Can you generate all combinations of well-formed parentheses for a given number?
"""


def generate_parenthesis(n):
    def generate(p, left, right, parens=[]):
        if left: generate(p + '(', left - 1, right)
        if right > left: generate(p + ')', left, right - 1)
        if not right: parens.append(p)
        return parens

    return generate('', n, n)


########################################
# Merge k Sorted Lists
########################################

"""
How would you merge k sorted linked lists and return it as one sorted list?
"""
import heapq


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def mergeKLists(lists):
    dummy = ListNode(0)
    current = dummy
    heap = []

    for index in range(len(lists)):
        if lists[index]:
            heapq.heappush(heap, (lists[index].val, index))
            lists[index] = lists[index].next

    while heap:
        val, index = heapq.heappop(heap)
        current.next = ListNode(val)
        current = current.next

        if lists[index]:
            heapq.heappush(heap, (lists[index].val, index))
            lists[index] = lists[index].next

    return dummy.next


########################################
# Pow(x, n)
########################################
"""
 Can you implement pow(x, n), which calculates x raised to the power n?
"""


def my_pow(x, n):
    if n < 0:
        x = 1 / x
        n = -n
    result = 1
    while n:
        if n & 1:
            result *= x
        x *= x
        n >>= 1
    return result


########################################
# Subsets
########################################
"""
 Can you find all possible subsets of a given set of distinct integers?
"""


def subsets(nums):
    result = []
    nums.sort()

    def backtrack(start, path):
        result.append(path)
        for i in range(start, len(nums)):
            backtrack(i + 1, path + [nums[i]])

    backtrack(0, [])
    return result


########################################
# Longest Increasing Subsequence
########################################

"""
Can you find the length of the longest subsequence of a given sequence such
 that all elements of the subsequence are sorted in increasing order?
"""


def lengthOfLIS(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


########################################
# Word Search
########################################

"""
Given a 2D board and a word, can you find if the word exists in the grid?
"""


def exist(board, word):
    if not board:
        return False
    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(board, i, j, word):
                return True
    return False


def dfs(board, i, j, word):
    if len(word) == 0:
        return True
    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or word[0] != \
            board[i][j]:
        return False
    tmp = board[i][j]
    board[i][j] = "#"
    res = dfs(board, i + 1, j, word[1:]) or dfs(board, i - 1, j,
                                                word[1:]) or dfs(board, i,
                                                                 j + 1, word[
                                                                        1:]) or dfs(
        board, i, j - 1, word[1:])
    board[i][j] = tmp
    return res


########################################
# Combination Sum
########################################

"""
Can you find all unique combinations in a set of candidates where the 
candidate numbers sums to a target?
"""


def combinationSum(candidates, target):
    result = []
    candidates.sort()

    def backtrack(remain, stack, start):
        if remain == 0:
            result.append(stack)
            return
        elif remain < 0:
            return
        for i in range(start, len(candidates)):
            backtrack(remain - candidates[i], stack + [candidates[i]], i)

    backtrack(target, [], 0)
    return result


# Call the function
print(combinationSum([2, 3, 6, 7], 7))

########################################
# Number of Islands
########################################

"""
How would you count the number of islands in a given 2D grid map?
"""


def numIslands(grid):
    if not grid:
        return 0
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(grid, i, j)
                count += 1
    return count


def dfs(grid, i, j):
    if (i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][
        j] != '1'):
        return
    grid[i][j] = '#'
    dfs(grid, i + 1, j)
    dfs(grid, i - 1, j)
    dfs(grid, i, j + 1)
    dfs(grid, i, j - 1)


# Call the function
grid = [['1', '1', '1', '1', '0'], ['1', '1', '0', '1', '0'],
        ['1', '1', '0', '0', '0'], ['0', '0', '0', '0', '0']]
print(numIslands(grid))

########################################
# Regular Expression Matching
########################################

"""
Given an input string (s) and a pattern (p), implement regular expression
 matching with support for '.' and '*'?
"""


def isMatch(s, p):
    dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
    dp[-1][-1] = True
    for i in range(len(s), -1, -1):
        for j in range(len(p) - 1, -1, -1):
            match = i < len(s) and (p[j] == s[i] or p[j] == '.')
            if j + 1 < len(p) and p[j + 1] == '*':
                dp[i][j] = dp[i][j + 2] or match and dp[i + 1][j]
            else:
                dp[i][j] = match and dp[i + 1][j + 1]
    return dp[0][0]


# Call the function
print(isMatch('aa', 'a*'))

########################################
# Maximum Subarray
########################################
"""
Can you find the contiguous subarray within an array (containing at least one 
number) which has the largest sum?
"""


def maxSubArray(nums):
    for i in range(1, len(nums)):
        if nums[i - 1] > 0:
            nums[i] += nums[i - 1]
    return max(nums)


# Call the function
print(maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))

########################################
# Binary Tree Inorder Traversal
########################################

"""
Can you retrieve all the values of a binary tree's nodes in the order of 
Inorder Traversal?
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def inorder_traversal(root):
    def helper(node, res):
        if node is not None:
            helper(node.left, res)
            res.append(node.val)
            helper(node.right, res)

    res = []
    helper(root, res)
    return res


# Call the function
root = TreeNode(1)
root.right = TreeNode(2)
root.right.left = TreeNode(3)
print(inorder_traversal(root))  # Prints [1, 3, 2]

########################################
# Validate Binary Search Tree
########################################
"""
Given a binary tree, can you determine if it is a valid binary search tree?
"""


def isValidBST(root):
    stack, inorder = [], float('-inf')

    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        if root.val <= inorder:
            return False
        inorder = root.val
        root = root.right
    return True


########################################
# Serialize and Deserialize Binary Tree
########################################

"""
How would you design an algorithm to serialize and deserialize a binary tree?
"""


class Codec:
    def serialize(self, root):
        def helper(node):
            if node:
                vals.append(str(node.val))
                helper(node.left)
                helper(node.right)
            else:
                vals.append('#')

        vals = []
        helper(root)
        return ' '.join(vals)

    def deserialize(self, data):
        def helper():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = helper()
            node.right = helper()
            return node

        vals = iter(data.split())
        return helper()


# Call the function
codec = Codec()
serialized = codec.serialize(root)
deserialized = codec.deserialize(serialized)
print(inorder_traversal(deserialized))  # Prints [1, 2, 3]

########################################
# Course Schedule
########################################

"""
Given the total number of courses and a list of prerequisite pairs, can you 
determine if it is possible for you to finish all courses
"""


def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    visited = [0 for _ in range(numCourses)]
    # create graph
    for pair in prerequisites:
        x, y = pair
        graph[x].append(y)
    # visit each node
    for i in range(numCourses):
        if not dfs(graph, visited, i):
            return False
    return True


def dfs(graph, visited, i):
    # if ith node is marked as being visited, then a cycle is found
    if visited[i] == -1:
        return False
    # if it is done visited, then do not visit again
    if visited[i] == 1:
        return True
    # mark as being visited
    visited[i] = -1
    # visit all the neighbours
    for j in graph[i]:
        if not dfs(graph, visited, j):
            return False
    # after visit all the neighbours, mark it as done visited
    visited[i] = 1
    return True


# Call the function
print(canFinish(2, [[1, 0]]))  # Prints True

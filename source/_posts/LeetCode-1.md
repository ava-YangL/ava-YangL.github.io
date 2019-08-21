---
title: LeetCode_1
date: 2019-08-21 19:58:41
tags:
---



#### 1 Find First and Last Position of Element in Sorted Array
Description：
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.
Your algorithm's runtime complexity must be in the order of O(log n).
If the target is not found in the array, return [-1, -1].
<!--more-->
E.g.:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
分析：
- 二分法，先找左边开始的地方，再找右边开始的地方。
- 时间复杂度O(log10(n)) :Because binary search cuts the search space roughly in half on each iteration, there can be at most log_{10}(n)iterations. Binary search is invoked twice, so the overall complexity is logarithmic.  ???为啥是10不是2呢
- 空间 O(1)
Code
```c
class Solution {
public:
    
    vector<int> searchRange(vector<int>& nums, int target) {
        //基本是抄的 我觉得还是有点乱。。。。
        vector <int> res(2,-1);
        
        if(nums.size()==0)
            return res;
        int i=0;
        int j=nums.size()-1;
        while(i<j)
        {
            //mid小的话 mid+1 因为是找左边的端点。所以j不变
            //这里不是i往前走 就是j不变，但mid会变的 所以j也在变
            int mid=i+(j-i)/2;
            if(nums[mid]<target)
                i=mid+1;
            else
                j=mid;
        }
        if(nums[i]!=target) //找不到 就是找不到的情况的单独的判断
            return res;
        res[0]=i;
    
        j=nums.size()-1;
         while(i<j)
        {
            //int mid=i+(j-i)/2;
             //j变小或者i变大 但mid不能不变
             int mid=i+(j-i)/2+1; //因为i=mid，所以要防止死循环（步长小于1）的情况而且左边的数定了
            if(nums[mid]>target)
                j=mid-1;
            else
                i=mid;
             
        }
       res[1]=j;
       return res;
         
    }
};
```

#### 2 Maximal Square
Description：
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area. 注意是方的
E.g.:
Input: 
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Output: 4
分析：
```c
 dp[i][j]=min(min(dp[i-1][j-1],dp[i-1][j]),dp[i][j-1])+1;     

```
- 我这个做法的时间复杂度和空间复杂度都是O(mn) 但空间复杂度似乎可以优化为O(n)
Code
```c
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.size()==0)
            return 0;
        int maxnum=0;
        vector<vector<int>> dp(matrix.size(),vector<int>(matrix[0].size()));
        // 这惊人的初始化方法
        for(int i=0;i<matrix.size();i++)
            for(int j=0;j<matrix[0].size();j++)
            {
                dp[i][j]=matrix[i][j]-'0';
                if(dp[i][j]==1)
                    maxnum=1;
            }
        for(int i=1;i<matrix.size();i++)
        {
            for(int j=1;j<matrix[0].size();j++)
            {
                if(matrix[i][j]=='1')
                {
                    //cout<<i<<" "<<j;
                    dp[i][j]=min(min(dp[i-1][j-1],dp[i-1][j]),dp[i][j-1])+1;                 
                    maxnum=max(maxnum,dp[i][j]);
                }
            }
        }
        return maxnum*maxnum;
    }
};
```

#### 3 Longest Valid Parentheses
Description：
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.
E.g.:
Input: ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()"
分析：
两种方法：
- 不用额外空间的方法 左右扫描： 太强了，判断左右符号数 , 时间O(n),空间O(1)
- 动态规划的方法： 当前字符是） 后面分为前一个字符是（ 或者）的情况, 时间空间复杂度都是O(n)
Code:
```c



// 不用额外空间的方法 左右扫描 太强了
class Solution {
public:
    int longestValidParentheses(string s) {
        int left=0;
        int right=0;
        int max_num=0;
        for(int i=0;i<s.size();i++)
        {
            if(s[i]=='(')
                left++;
            if(s[i]==')')
                right++;
            //cout<<left<<" "<<right<<endl;
            if(right==left)
            {
                int temp=right*2;
                max_num=(max_num>temp)?max_num:temp;
            }
            if(right>left)
            {
                left=0;
                right=0;
            }
        }
        
        left=0;right=0;
        for(int i=s.size()-1;i>=0;i--)
        {
            if(s[i]=='(')
                left++;
            if(s[i]==')')
                right++;
            //cout<<left<<" "<<right<<endl;
            if(right==left)
            {
                int temp=right*2;
                max_num=(max_num>temp)?max_num:temp;
            }
            if(left>right)
            {
                left=0;
                right=0;
            }
        }
        
        
        return max_num;

    }

};





// //动态规划的方法 当前字符是） 后面分为前一个字符是（ 或者）的情况
// class Solution {
// public:
//     int longestValidParentheses(string s) {
//         int n=s.size();
//         if(n==0)
//             return 0;
//         vector<int> res(n,0);
//         int maxans=0;
//         for(int i=0;i<n;i++)
//         {
//             if(s[i]==')' && ((i-1)>=0) && (s[i-1]=='('))
//                 res[i]=(i-2>=0)?res[i-2]+2:2;
//             if(s[i]==')' && ((i-1)>=0) && (s[i-1]==')'))
//             {
//                 //算这一套（（））的
//                 res[i]=((i-res[i-1]-1)>=0 && s[i-res[i-1]-1]=='('  )?res[i-1]+2:0;
//                 //加前面的！！！
//                 res[i]+=((i-res[i-1]-2)>=0 && (s[i-res[i-1]-1]=='(')?res[i-res[i-1]-2]:0);
                   
//                 // ( ) ) ) ) ( ( ( ) )  )  )
//                 // 0 1 2 3 4 5 6 7 8 9 10 11
//             }
//             //cout<<i<<" "<<res[i]<<endl;
//             if(res[i]>maxans) maxans=res[i];     
//         }
//         return maxans;
//     }

// };
```

#### 4 Word Search
Description：
Given a 2D board and a word, find if the word exists in the grid.
The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.
E.g.:
```c
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```
分析：
- 本函数用于循环双重vector，对每一种情况进行判断
- dfs函数，有个满足word的情况，返回true， 然后在这个word的上下左右搜索，注意状态的回滚。
Code
```c
class Solution {
public:
    
    bool dfs(vector<vector<char>>& board, string word, int i,int j,int k)
    {
        //cout<<"%%"<<endl;
        //cout<<i<<" "<<j<<" "<<k<<" "<<endl;
        if(i<0||j<0||i>(board.size()-1)||j>(board[i].size()-1)||k>=word.size()||board[i][j]!=word[k])
            return false;
        //k用来判断word种植
        if(k==word.length()-1)//这里是-1
            return true;
        char now=word[k];
        char mark=board[i][j];
        board[i][j]='*';//光这样不行 要注意状态的回退
        
        bool ans=dfs(board,word,i,j+1,k+1)
            ||dfs(board,word,i,j-1,k+1)
            ||dfs(board,word,i-1,j,k+1)
            ||dfs(board,word,i+1,j,k+1);
        board[i][j]=mark;
        return ans;
    }  
    bool exist(vector<vector<char>>& board, string word) {
        if(word.empty()) return true;
        for(int i=0;i<board.size();i++)
        {
            for(int j=0;j<board[i].size();j++)
            {
                if(board[i][j]==word[0])
                    if(dfs(board,word,i,j,0))
                        return true;
            }
        }
        return false;
        
    }
};
```


#### 5 Combination Sum
Description：
E.g.:
```c
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]

Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```
分析： 也是递归的方法来做，另外那个递归函数里，满足条件就push，注意状态的回滚
Code
```c
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        vector<vector<int>> ans;
        vector<int> temp;
        com(candidates,target,ans,temp,0);
        return ans;      
    }
    
    void com(vector<int> candidates, int target, vector<vector<int>>& ans ,vector<int> temp, int index)
    {
        if(target<0)
            return;
        if(target==0)
        { 
            ans.push_back(temp);
            return;
        }
        
        for(int i=index;i<candidates.size();i++)
        {
            if(target<candidates[i])
                break;
            temp.push_back(candidates[i]);
            com(candidates,target-candidates[i],ans,temp,i);
            temp.pop_back();
        }
        
    } 
};
```

#### 6 Merge Intervals --2019/8/21
Introduction:
Given a collection of intervals, merge all overlapping intervals.
E.g.:
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
分析：
- 重点是sort函数的模板函数的写法，注意要是static的
- 利用sort函数对第一个数字进行排序
-复杂度分析：时间复杂度 nlogn,因为sort嘛，然后还有简单的线性扫描；空间复杂度O(1)或者O(n)主要看sort的时候需不需要额外的空间。

```c
//这个必须放在外面诶  或者放在里面 写个static
// const bool comp(const vector<int> &a ,const vector<int> &b)
// {
//     return a[0]<b[0];
// }
class Solution {
public:
    
    static const bool comp(const vector<int> &a ,const vector<int> &b)
    {
     return a[0]<b[0];
    }

    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> ans;
        if(intervals.size()==0)
            return ans;
        sort(intervals.begin(),intervals.end(),comp);

        //把第一个数push进去
        ans.push_back(intervals[0]);
        
        for(int i=1;i<intervals.size();i++)
        {
            if(intervals[i][0]> ans[ans.size()-1][1]) //我总是和intervals去比。。。。其实是不对的
                ans.push_back(intervals[i]);
            else{
              ans[ans.size()-1][1]=max(ans[ans.size()-1][1],intervals[i][1]);
            } 
        }       
        return ans;
    }
};
```


Description：
E.g.:
分析：
Code

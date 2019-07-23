---
title: Code
date: 2019-03-22 14:31:02
categories: 
- Code
tags:
- 二分法
- 牛顿法
- 堆排序
---
希望能每天刷一道题？？？
<!--more-->

#### 1 LeetCode 132. Palindrome Partitioning II （20190709）

Given a string s, partition s such that every substring of the partition is a palindrome.
Return the minimum cuts needed for a palindrome partitioning of s.
```c
class Solution {
public:
    int minCut(string s) {   
        /*
        思路：找一个一维数组mark从后往前
        保存最小割数；找一个二维数组保存p[i][j]
        是不是回文
        就是：mark[i]=s(i:len)的割数
        针对每个可能的割点j:
        (1)p[i][j]是回文（就是找到了是的mark[i]更小的方法）
        那么，mark[i]=1+mark[j+1](这里就是在j这里切了一刀)
        (2)p[i][j]不是回文：那没有办法了，就等于原来mark[i]
        */      
        int len=s.size();
        int mark[len+1];
        bool p[len][len];
        //最坏情况
        for(int i=0;i<=len;i++)//多了一个
        {
            mark[i]=len-i-1;
        }      
        for(int i=0;i<len;i++)
            for(int j=0;j<len;j++)
                p[i][j]=false;
        for(int i=len-1;i>=0;i--)//从最后一个开始
        {
            for(int j=i;j<len;j++)//针对每个可能的割点j
            {
                if(s[i]==s[j] && (j-i<2||p[i+1][j-1]))//如果割点j起作用了，也就是ij是回文
                {
                    p[i][j]=true;
                    mark[i]=min(mark[i],mark[j+1]+1);//这里是因为可能之前的割点更好啊
                }
                    
                   
            }
        }        
        return mark[0];
        
    }
};
```

注意vector的初始化和定义方法
```c
vector<int> p(len+1,0);
        for(int i=0;i<len+1;i++)
            p[i]=len-i-1;
        vector<bool> tmp(len,false);
        vector<vector<bool>> b(len,tmp);
```
#### 2 Leetcode135 Candy （20190710）
There are N children standing in a line. Each child is assigned a rating value.
You are giving candies to these children subjected to the following requirements:
Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
What is the minimum candies you must give?

本题采用从左扫到右，再从右扫到左的策略（好像还有其他的解法，但我有点懒得看了。。。。）

```c
class Solution {
public:
    int candy(vector<int> &ratings) {
        int len=ratings.size();
        vector<int> res(len,1);
        for(int i=1;i<len;i++)
        {
            if(ratings[i]>ratings[i-1])
                res[i]=res[i-1]+1;  
        }
        for(int i=len-2;i>=0;i--)
        {
            if(ratings[i]>ratings[i+1] && res[i]<=res[i+1]) //注意多了一个情况
                res[i]=res[i+1]+1;  
        }
        return accumulate(res.begin(),res.end(),0);
        
    }
};
```

### 3 Single Number Leetcode136 (20190711)
Given a non-empty array of integers, every element appears twice except for one. Find that single one.
Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
这里做的是异或运算（转换为2进制，每一位相同的是0，不同的是1） 也就是10进制的异或，和0异或是他本身，两个相等的10进制数异或为0，这样就能把那个单独的数找出来啦。
2 ^ 3= 0000 0010 ^ 0000 0011 = 0000 0001   1
1 ^ 2= 0000 0001 ^ 0000 0010 = 0000 0011   3
3 ^ 3= 0000 0011 ^ 0000 0011 =  0000 0000  0
0 ^ 4= 0000 0000 ^ 0000 0100 = 0000 0100   4
4 ^ 1= 0000 0100 ^ 0000 0001 = 0000 0101   5
5 ^ 4= 0000 0101 ^ 0000 0100 =0000 0001    1
```c
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int num=0;
        int n=nums.size();
        for(int i=0;i<n;i++)
            num^=nums[i];
        return num;
        
    }
};
```
#### 4 142. Linked List Cycle II  （20190713）
判断有没有环，没有返回NULL，有就返回起点的位置~~

```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        /*
        快慢指针，这个题之前做过了，我有在偷懒
        就是快慢指针先看看相不相遇，相遇就是有环，记相遇时：
        慢指针走了a+x ,(x肯定在环里，即从入口到相遇点距离为x)
        从相遇点再到入口，距离为y
        则快指针走了a+x+a+x=a+x+l（快比慢多走了l）=a+x+(x+y)
        所以，a=y,所以慢指针再走y，快指针再从0走到a就是相遇点啦。
        */
        if(head==NULL || head->next==NULL)
            return NULL;
        ListNode *fast=head;
        ListNode *slow=head;
        while( slow && fast && fast->next){
            fast=fast->next->next;
            slow=slow->next;
            if(fast==slow)
            {
                fast=head;
            while(fast!=slow)
            {
                fast=fast->next;
                slow=slow->next;
            }
                return fast;
            }
        }
        return NULL;
    }
};
```
#### 5 Word break (20190714)
Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

For example, given
s ="leetcode",
dict =["leet", "code"].

Return true because"leetcode"can be segmented as"leet code".

注意s.substr的参数是起始位置，字符个数。注意unordered_set是find不等于end
动态规划的思想

```c
class Solution {
public:
    bool wordBreak(string s, unordered_set<string> &dict) {
        
        //针对每一个字符做一下判断？？
        int len=s.length();
        vector<bool> mark(len+1,false);//len+1是因为0是标志位啊
        mark[0]=true; 
        /*
        mark[pos]=mark[i] && mark[i+1,pos]
        就是0-pos可分词 是 0-i在字典里 并且i+1-pos在字典里 都是闭括号
        */
        for(int pos=1;pos<=len;pos++) //判断每个字符
        { //以<=i做判断
            for(int i=0;i<=(pos-1);i++)
            {
                //if(mark[pos] && dict.find(s[pos+1:i])!=dict.end())
                if(mark[i] && dict.find(s.substr(i,pos-i))!=dict.end())
                    mark[pos]=true;
            }
        }
        return mark[len];
    }
};
```
然后同样的题在Leetcode 139,注意代码里判断vector里面存不存在某元素的方法鸭。
```c
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int len=s.length();
        vector<bool> mark(len+1,false);
        mark[0]=true;
        
        for(int i=1;i<=len;i++)
        {
            for(int j=0;j<=i;j++)
            {
                //mark{i]=mark[0:j] && mark[j+1,i]
                if(mark[j] && find(wordDict.begin(),wordDict.end(),s.substr(j,i-j))!=wordDict.end())
                    mark[i]=true;
            }
        }
        return mark[len];
        
    }
};



```
#### 6  LEETCODE 121. Best Time to Buy and Sell Stock (20190715)


Say you have an array for which the ith element is the price of a given stock on day i.
If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.
Note that you cannot sell a stock before you buy one.
Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
竟然不是一道动态规划题目，可以画图来做！！ 先找最低点 （然后找差值最大的）
```c
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minn=INT_MAX;
        int maxx=0;
        for(int i=0;i<prices.size();i++)
        {
            if(prices[i]<minn)
                minn=prices[i];
            if((prices[i]-minn) > maxx)
                maxx=(prices[i]-minn);
            //cout<<minn<<" "<<maxx<<" "<<endl;
        }
        return maxx;
        
    }
};
```


#### 7 104. Maximum Depth of Binary Tree  （20190716）
二叉树的深度 ，这是递归的方法，别人竟然只写了一行代码？？？？？
```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        
        if(root==NULL)
            return 0;
        // int a=1;
        // int b=1;
        // if(root->left) 
        //     a= 1+maxDepth(root->left);
        // if(root->right) 
        //     b=1+maxDepth(root->right);
        // return max(a,b);
        return 1+max(maxDepth(root->left),maxDepth(root->right));
        
    }
};
```
非递归的方法呢 就是层序遍历 我太强了！！！
```C
class Solution {
public:
    int maxDepth(TreeNode* root) {
        int num=0;
        //非递归的方法就是层序遍历？？？
        queue<TreeNode* > q;
        if(root==NULL) return 0;
        q.push(root);
        while(!q.empty())
        {
            num++;
            int start=0;
            int end=q.size();
            while(start++<end)
            {
                TreeNode * front=q.front();
                q.pop();
                if(front->left) q.push(front->left);
                if(front->right) q.push(front->right);
            }
        }
        return num;
        
    }
};
```
#### 8  771. Jewels and Stones (20190717)
You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

思路：防止N乘M的遍历，就用字典来做鸭！！
```c
class Solution {
public:
    int numJewelsInStones(string J, string S) {
        int num=0;
        unordered_map<char,int> dic;  // unordered map
        for(int i=0;i<J.length();i++)
        {
            dic[J[i]]=i;
        }
        for(int i=0;i<S.length();i++)
        {
            if(dic.find(S[i])!=dic.end()) //  注意这里
                num++;
                
        }
        return num;
        
    }
};

```
#### 9 198. House Robber  (20190717)
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

其实我思路是对的 但是写的不太对，其实是永远更新最大值，但我0 1 元素一开始都等于自己了 -.-
```c
class Solution {
public:
    int rob(vector<int>& nums) {
        if(nums.size()==0)
            return 0;
        if(nums.size()==1)
            return nums[0];
        //我写的有些麻烦 其实可以多两个元素的
        // vector<int> mark(nums.size(),0);
        // for(int i=0;i<nums.size();i++)
        // {
        //     if(i==0)
        //     {
        //         mark[i]=nums[i];
        //         continue;
        //     }   
        //     if(i==1)
        //     {
        //         mark[i]=max(nums[0],nums[1]);
        //         continue;
        //     }
        //     mark[i]=max(mark[i-1],mark[i-2]+nums[i]);
        // }
        // return max(mark[nums.size()-1],mark[nums.size()-2]);
        vector<int> mark(nums.size()+2,0);
        for(int i=2;i<nums.size()+2;i++)
        {
            mark[i]=max(mark[i-1],mark[i-2]+nums[i-2]);
        }
        return max(mark[nums.size()],mark[nums.size()+1]);
    }
};
```
#### 10 437. Path Sum III (20190718)
https://leetcode.com/problems/path-sum-iii/
找有几条节点和为sum的路径在这个树里
```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int pathSum(TreeNode* root, int sum) {
        if(root==NULL)
            return 0;
        //返回以我为起点的 以我孩子为起点的，所以调用的是本来这个函数
        return get(root,sum,0)+pathSum(root->left,sum)+pathSum(root->right,sum);
    }
    int get(TreeNode* root ,int sum ,int now)
    {
        
        if(root==NULL)
            return 0;
        //这个是用来去遍历这一分支的，但都是以root为起点的哦
        return ((sum-now)==root->val)+get(root->left,sum,now+root->val)+get(root->right,sum,root->val+now);
    }
};

```
#### 11 621 task scheduler（20180721）
就是输入任务列表，然后输入n，要求相同的任务之间必须隔有n个任务，要你输出总共执行了多少个任务呢。
-.- 似乎有更好的做法？？？？或者更简洁的代码？？再看看呢。
```c
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        unordered_map <char,int> dict;
        //放入字典
        for(int i=0;i<tasks.size();i++)
        {
            if(dict.find(tasks[i])!=dict.end())
                dict[tasks[i]]++;
            else
                dict[tasks[i]]=1;
        }
        //找出最大值
        int maxnum=0;
        auto iter=dict.begin();
        while(iter!=dict.end())
        {         
            maxnum=max(maxnum,iter->second);
            iter++;
        }
        //有没有多的最大值
        int k=0;
        auto iter2=dict.begin();
        while(iter2!=dict.end())
        {   
            if(iter2->second==maxnum)
                k++;
            iter2++;
        }
        //插空 按照最大值来算
        int ans=maxnum+(maxnum-1)*(n)+k-1;
        //特殊处理 万一没有需要插孔的之类的呢
        int x=tasks.size();
        int res=max(ans,x);
        return res;
        
    }
};
```

#### 12 543. Diameter of Binary Tree （饶了好久 5555555,20190722）
求最长的路径，再看看呢

```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int diameterOfBinaryTree(TreeNode* root) {
        if(root==NULL)
            return 0;
        int a=0;
        return getH(root,a);
        
    }
    int getH(TreeNode* root, int &h) //h代表高度 妈呀我觉得这个题有点难lv
    {
        if(root==NULL)
        {
            //h=0;//之前总是忘了这一句 这个似乎也不重要？？
            return h;
        }
        int lv=0;
        int rv=0;

        int left=getH(root->left,lv); //LV代表高度， left表示left的直径
        int right=getH(root->right,rv);
        h=max(lv,rv)+1; //更新自己高度 所以适合lv rv比 而不是left right比
        return max(max(left,right),lv+rv);
    }
};
```

#### 13 617. Merge Two Binary Trees (20190723)
Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.
递归的方法来做 是不是有些简单呢。

```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        if(t1==NULL && t2==NULL)
            return t1;
        if(t1==NULL)
            return t2;
        if(t2==NULL)
            return t1;
        t1->val=t1->val+t2->val;
        t1->left=mergeTrees(t1->left,t2->left);
        t1->right=mergeTrees(t1->right,t2->right);
        return t1;
        
    }
    
    
};
```
-----------------------------------------------------------------------
这里是7月和3月的分界线
------------------------------------------------------------------------
### 1 求平方根
#### 牛顿法

泰勒公式：
$$f(x)\approx g(x)=g(x_0)+\frac{f^1(x_0)}{1!}(x-x_0)+\frac{f^2(x_0)}{2!}(x-x_0)^2+...+\frac{f^n(x_0)}{n!}(x-x_0)^n$$
一阶展开：
$$f(x)\approx g(x)=g(x_0)+\frac{f^1(x_0)}{1!}(x-x_0)$$
牛顿法让这个等于0：$$0=g(x_0)+\frac{f^1(x_0)}{1!}(x-x_0)$$
也就是：$$f(x_n)+f^1(x_n)(x_{n+1}-x_n)=0$$
也就是：$$x_{n+1}=x_n-\frac{f(x_n)}{f^1(x_n)}$$
带入：$y=x^2$ 得到$ f(x)=x^2-a$ ，注意这里求平方根就说明a是确定得了
得到：$x_{n+1}=x_n-\frac{x_n^2-a}{2x_n}$
```c
class Solution {
public:
    int mySqrt(int x) {
        double ep=1e-2;
        double res=x;
        while((res*res-x)>ep){
            res=res-(res*res-x)/(2*res);
        }
        cout<<res;
        return int(res);
    }
};
```

#### 二分法
```c
class Solution {
public:
    int mySqrt(int x) {
        int l=1;
        int r=x;
        if(x<=1) return x;  //注意这里
        while(l<=r){
            //int mid=(l+r)/2;  //这样会越界
            int mid=l+(r-l)/2;
            //if(mid*mid<=x && (mid+1)*(mid+1)>x ) //也会越界
            if(mid==(x/mid))
                return mid;
            else{
                if(mid>(x/mid))
                    r=mid-1;
                else
                    l=mid+1;
            }
        }
        return r;  //跳出后r偏小，返回r
        
        
    }
};
```
### 2 n个数找k大
堆排序
关于堆排序的一些新的认识，一整个堆升序确实可以用小根堆；降序可以用大根堆；
但是要做到nlogk的时间复杂度，堆里面只有K个数的话，升序就要用到大根堆，选出小的数来，降序用小根堆，选出大的数来。

#### 全部排序再找前k个

```c
class Solution {
public:
    void adjust_heap(vector<int>&a,int node,int n){
        int left=node*2+1;//假设left就是较大孩子的下标 
        while(left<n){
        	if(left+1<n && a[left]<a[left+1]){
        		left++;//更新大孩子下标 
			}
			if(a[node]<a[left]){
				swap(a[node],a[left]);
				node=left;
				left=2*node+1;
			}
			else
			  break; 
		}
        
    }
void heap_sort(vector<int> &a,int n){
        for(int i=(n-1)/2;i>=0;i--){
            adjust_heap(a,i,n);
        }
        for(int i=n-1;i>=0;i--){
            swap(a[i],a[0]);
            adjust_heap(a,0,i);
        }  
    }
      
vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        int n=input.size();
        heap_sort(input,n);
        vector<int> res;
        if(k>n)
            return res;
        for(int i=0;i<k;i++)
            res.push_back(input[i]);
        return res;
    }
};
```
#### 这个不用全部排序
```c
class Solution {
public:
    void adjust_heap(vector<int>&a,int node,int n){
        int child=2*node+1;
        while(child<n){
            if(child+1<n && a[child]<a[child+1]){
                child++;
            }
            if(a[node]<a[child]){
                swap(a[node],a[child]);
                node=child;
                child=2*node+1;
            }
            else
                break;
        }
   }
    
    void heapsort(vector<int>& a,int n){
        for(int i=(n-1)/2;i>=0;i--){
            adjust_heap(a,i,n);
        }
        ** 替换掉最大的
        for(int i=n;i<a.size();i++){
            if(a[0]>a[i])
            {
                a[0]=a[i];
                adjust_heap(a,0,n);
            }
            else
               continue;
        }
        **
        for(int i=n-1;i>=0;i--){
            swap(a[i],a[0]);
            adjust_heap(a,0,i);
        }
    } 
  
      
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) { 
        heapsort(input,k);
        vector<int>ans;
        if(k>input.size())
            return ans;
        for(int i=0;i<k;i++){
           ans.push_back(input[i]);
         }
       return ans;    
    }
};
```
```c
priority_queue<int> xxx 大根堆
priority_queue<int, vector<int>, greater<int>> xxxx 小根堆
```

### 3 重复元素
一个大小为n的数组，里面的数都属于范围[0, n-1]，有不确定的重复元素，找到至少一个重复元素
https://blog.csdn.net/SunnyYoona/article/details/43883519

有n+1个属于1～n的数，只有一个重复的数，求出来：所有数相加-（1+...n)

### 4 全排列，LeetCode 46. Permutations

```c
class Solution {
public:
    vector<vector<int>> ans;
    vector<vector<int>> permute(vector<int>& nums) {
        dfs(nums,0);
        return ans;   
    }
    void dfs(vector<int> nums,int pos){
        if(pos==nums.size()-1)
        {
            ans.push_back(nums);
            return;
        }
        for(int i=pos;i<nums.size();i++){ //这里写错了，是i=pos
            swap(nums[i],nums[pos]);
            dfs(nums,pos+1);
            swap(nums[i],nums[pos]);
         }  
        
    }
   
};

```
### 5 LeetCode98 验证二叉搜索树
```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
    bool isValidBST(TreeNode* root) {
        bool res;
        long long max=LLONG_MAX;
        long long min=LLONG_MIN;
        res=dfs(root,max,min);
        if(root==NULL)
            return true;
        if(root->left==NULL && root->right==NULL)
            return true;
        return res;
        
    }
    bool dfs(TreeNode* root,long long max,long long min){
        if(root==NULL)
            return true;
        
        bool a=true;
        bool b=true;
        
        if(root->left) a =dfs(root->left,root->val,min);
        if(root->right) b=dfs(root->right,max,root->val);
        bool c=root->val>min && root->val<max;
        return a&&b&&c;
        
        
    }
};

```
### Softmax

--------------------------------------------------------------------------------------------
2019年4月编码准备

##### 1 二叉树最小深度
###### 1 递归方法，左树不存在，右树高度加1；右树不存在，左树高度加1；左右树都存在，最小高度+1。
```c
 int run(TreeNode *root) {
        if (root==NULL)
            return 0;
        if(root->left==NULL && root->right==NULL)
        {
            return 1;
        }
        if(root->left==NULL) return 1+run(root->right);
        if(root->right==NULL) return 1+run(root->left);
        int depthl=0;
        int depthr=0;
        if(root->left) depthl=run(root->left);
        if(root->right) depthr=run(root->right);
        return 1+min(depthl,depthr);

    }
```
###### 2 广度优先，层序遍历，遇到第一个叶子节点的时候
```c
 int run(TreeNode *root) {
         if(root == NULL) return 0;
         queue<TreeNode *>q;
         q.push(root);
         int depth=1;
         while(!q.empty()){
             int i=0;
             int len=q.size();
             for(i=0;i<len;i++){  //这里要写len 不能写q.size()因为q一直在变？
                 TreeNode *temp=q.front();
                 q.pop();
                 if(temp->left==NULL && temp->right==NULL) return depth;
                 if(temp->left) q.push(temp->left);
                 if(temp->right) q.push(temp->right);
             }
             if(i==len) //跳出来 不是等于len-1啊
                 depth++;
         }
         
         return depth;
    }
```
##### 2 链表排序Onlogn
###### 1 归并排序，找到中间节点（一个两步，一个一步），然后断开链表，然后合并两个断开的链表。用递归的方法。

```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *sortList(ListNode *head) {
        if(head==NULL || head->next==NULL) return head;
        ListNode * p=head;
        ListNode * q=head->next;
        while(p!=NULL && q!=NULL && q->next!=NULL) //不加q->next!=NULL的判断就会内存超限
        {
            p=p->next;
            if(q->next)
                q=q->next->next;
            else
                q=q->next;
        }
        //返回合并后的链表的右边
        ListNode *right=sortList(p->next);
        //返回链表的左边
        p->next=NULL;//断开
        ListNode *left=sortList(head);
        //合并
        return merge(left,right);
        
        
    }
    ListNode * merge(ListNode * left,ListNode * right){
        if(left==NULL && right==NULL)
            return NULL;
        if(left==NULL)
            return right;
        if(right==NULL)
            return left;
        ListNode dummy(-1); //这样写，以前都是直接new的没有释放掉
        ListNode *p= &dummy;
        while(left!=NULL && right!=NULL)
        {
            if(left->val > right->val)
            {
                p->next=right;
                p=p->next;
                right=right->next;
            }else{
                p->next=left;
                p=p->next;
                left=left->next;
            }
            
        }
        if(left!=NULL)  p->next=left;
        if(right!=NULL) p->next=right;
        return dummy.next; //注意这里是next哦
    }
};
```
###### 2 快排的改进版本（注意本身的快排自己最好动手写一下）
因为链表没办法往前（j--）
所以这里是，两个指针往后挪，来找到key应该在的partition的位置的。
p=head;q=head->next; 遇到q比key小，p=p->next, 交换pq。
最后需要记得把p的位置的值和key位置的值进行交换。
```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getParti(ListNode * head,ListNode * end){
        if(head==NULL || head->next==NULL) 
            return head;     
        int key=head->val; 
        ListNode *p=head;
        ListNode *q=head->next;
        //q的值比p小，p往后挪一下，交换。
        while(q!=end){
            if(q->val < key){
                p=p->next;
                swap(p->val,q->val);
            }
            q=q->next;
        }
        swap(head->val,p->val);//p还是p->next,是p因为最后p的位置是比key小的最后一个位置。
        //再把这个小数换到前面就好了
        return p;
    }
    
   void quickSort(ListNode *head,ListNode *end){
       //if(head==NULL || end==NULL || head==end) //一开始这里写了这句话，这样就直接跳过了
           //return;
        if(head!=end){
            ListNode *mid=getParti(head,end);
            quickSort(head,mid);
            quickSort(mid->next,end);
        }
    }
    
    ListNode *sortList(ListNode *head) {
        //能通过指针的操作直接改变值？ 所以quicksort没有返回值可行
        quickSort(head,NULL);
        return head;
    }
};
```
###### 冒泡排序
```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        
        ListNode *temp=head;
        ListNode *tail=NULL;
        
        while(tail!=head)
        {
            ListNode *max=head;
            while(temp->next!=tail){
                if(max->val <temp->val)
                    max=temp;
                temp=temp->next;
            }
            tail=temp;    
            if(max->val > temp->val)
              swap(max->val,temp->val);       
            temp=head;
            //cout<<tail->val<<" "<<temp->val<<" "<<max->val<<endl;
        }
        //cout<<head->val<<endl;
        //cout<<tail->val<<endl;
        //cout<<temp->val<<endl;
        return head;
        
    }
};


```

##### 3 Two Sum Leetcode1
###### 基本知识
1 地址：https://leetcode.com/problems/two-sum/
- list支持快速的插入和删除，查找费时
- vector支持快速的查找，插入费时
- STL的map底层是用红黑树（平衡二叉树？）存储的，查找时间复杂度是log(n)级别；
- 2、STL的hash_map底层是用hash表存储的，查询时间复杂度是常数级别；
- 3、什么时候用map，什么时候用hash_map?
  这个要看具体的应用，不一定常数级别的hash_ map一定比log(n)级别的map要好，hash_ map的hash函数以及解决地址冲突等都要耗时，而且众所周知hash表是以空间效率来换时间效率（时间效率一般是常数级别？）的，因而hash_ map的内存消耗肯定要大。一般情况下，如果记录数非常大时，考虑hash_ map，查找效率会高很多，如果要考虑内存消耗，则要谨慎使用hash_map。
```C
#include <map>
map<string, string> namemap;
namemap["东方不败"]="第一高手，葵花宝典";
//查找。。
if(namemap.find("岳不群") != namemap.end()){
        ...
}

#include <hash_map>

hash_map<int, string> mymap;
mymap[9527]="唐伯虎点秋香";

 ...
if(mymap.find(10000) != mymap.end()){
    ...
}
```
C++ STL中，哈希表对应的容器是 unordered_ map（since C++ 11）。根据 C++ 11 标准的推荐，用 unordered_ map 代替 hash_ map。
```c
#include <unordered_map>
```

###### 两个循环
```c
这种方法其实相同的也能处理。两种循环
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> ans;
        unordered_map<int,int> data;
        for(int i=0;i<nums.size();i++){
            //data[i]=nums[i]; //注意这里是把数的值当作key
            data[nums[i]]=i;
       }
        for(int i=0;i<nums.size();i++){
            int cha=target-nums[i];
            //if(data.find(cha)!=data.end() && data[cha]!=data[nums[i]]){ 错
            //if(data.find(cha)!=data.end() && data[cha]!=i){ 对
            if(data.count(cha) && data[cha]!=i)
            {
                ans.push_back(i);
                ans.push_back(data[cha]);
                return ans;
            }
            
        }
     return ans;

    }
};

```
###### 优化为一个循环
```c
一个循环的方法，这样就很简单了
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> ans;
        unordered_map<int ,int> data;
        for(int i=0;i<nums.size();i++){
            int cha=target-nums[i];
            if(data.count(cha))
            {
                ans.push_back(i);
                ans.push_back(data[cha]);
                break;
            }
            data[nums[i]]=i;
        }
        return ans;
    }
};

```
##### 4 层次遍历二叉树 Leetcode 102
https://leetcode.com/problems/binary-tree-level-order-traversal/

```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int> > ans;
        queue<TreeNode*> q;
        if(root==NULL) return ans;
        q.push(root);
        while(!q.empty()){
            int len=q.size();
            vector<int> aa;
            while(len--){
                TreeNode * temp=q.front();
                q.pop();
                aa.push_back(temp->val);
                if(temp->left) q.push(temp->left);
                if(temp->right) q.push(temp->right); 
            }
           ans.push_back(aa);
        }
        return ans;
    }
};
```

##### 5 DFS括号匹配全排列 leetcode22
字符串删除最后一个字符知识点。
```c
str = str.substr(0, str.length() - 1);
str.erase(str.end() - 1);
str.pop_back();
```
这是我自己写的非常乱的一个代码
```c
class Solution {
public:
    vector<string> ans;
    vector<string> generateParenthesis(int n) {
        string temp;
        dfs(0,n,temp,ans,0,0);
        return ans;      
    }
    void dfs(int now,int n,string temp,vector<string>& ans,int num1,int num2){
        if(now==2*n)
        {
            ans.push_back(temp);
            return;
        }
        if(num1>num2 && num1<n)
        {
            temp.push_back('(');
            dfs(now+1,n,temp,ans,num1+1,num2);
            temp.pop_back();
            temp.push_back(')');
            dfs(now+1,n,temp,ans,num1,num2+1);
            temp.pop_back();
            return;
        }
        if(num1>num2 && num1==n)
        {
            temp.push_back(')');
            dfs(now+1,n,temp,ans,num1,num2+1);
            temp.pop_back();
            return;
        }
        {
            temp.push_back('(');
            dfs(now+1,n,temp,ans,num1+1,num2);
            temp.pop_back();
            
        }
        
    }
};
```
代码精简之后,其实这种全排列，dfs函数（我这个名字起的似乎很不准确）（1）：最开始有一个控制循环停止的条件，我们这里在最前面加上判断，使得)比(多的情况直接被PASS掉了。（2）还有一个到了某种情况把temp结果push到最终结果的。（3）通过不断地迭代走到最后一步 （4）而DFS函数本身是对各种情况的控制。
```c
class Solution {
public:
    vector<string> ans;
    vector<string> generateParenthesis(int n) {
        string temp;
        dfs(n,temp,ans,0,0);
        return ans;      
    }
    void dfs(int n,string temp,vector<string>& ans,int num1,int num2){
        //开始加个跳出的
        if(num1<num2) //这个直接就防止了)比（多的情况 所以下面的num1==num2那个可以省去了
            return; //return了就不会进入死循环了
        if(num2==n)
        {
            ans.push_back(temp);
            return;
        }
        /*
        if(num1==num2)
        {
             dfs(n,temp+'(',ans,num1+1,num2);
             return;
        }*/
        if(num1<n) 
            dfs(n,temp+'(',ans,num1+1,num2); //注意加上if判断
        dfs(n,temp+')',ans,num1,num2+1);    
    }
};
```
##### 6 旋转数组找某值 Leetcode33
一开始，right或者left处的边界总是出错， 所以在 while 循环里面加了对于边界的判断才过了。

```c
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left=0;
        int right=nums.size()-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]==target) 
                return mid;
            if(nums[right]==target) 
                return right;
            if(nums[left]==target) 
                return left;
            if(nums[mid]>nums[left]){ //左边有序
                if(target<nums[mid] && target>=nums[left])
                {
                    right=mid-1;
                    //right=mid;
                }else{
                    left=mid+1;
                }
                    
            }
            else{ //右边有序
                if(nums[mid]<target && nums[right]>=target)
                {
                    left=mid+1;
                }else{
                    right=mid-1;
                }
            }
        }
        return -1;
        
    }
};
```
这样就没有边界问题了，可是为什么呢？？
```c
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left=0;
        int right=nums.size()-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]==target) 
                return mid;     
            if(nums[mid]<nums[right]){
                 if(nums[mid]<target && nums[right]>=target)
                {
                    left=mid+1;
                }else{
                    right=mid-1;
                }          
            }
            else{
                if(target<nums[mid] && target>=nums[left])
                {
                    right=mid-1;
                }else{
                    left=mid+1;
                }
                
            }
        }
        return -1;
        
    }
};
```
##### 7 Leetcode 146. LRU Cache Hard

```c
class LRUCache {
private:
    int cap;
    list<int> recent; //存的key
    unordered_map<int,int> cache; //key value
    unordered_map<int,list<int>::iterator> pos; //key pos
public:
    LRUCache(int capacity) {
        cap=capacity;        
    }
    
    int get(int key) 
    {
        if(cache.find(key)!=cache.end()){
            
            //list<int>::position=pos[key];
            recent.erase(pos[key]);
            recent.push_front(key);
            //cache是不用变得
            pos[key]=recent.begin();
            
            
            return cache[key];
        }else
            return -1;
    }
        
    void put(int key, int value) {
        if(recent.size()>=cap && pos.find(key)==pos.end()){ //这里的put可能是替换 所以这一步不一定会执行
            int old=recent.back();
            recent.pop_back();
            cache.erase(old);
            pos.erase(old);//似乎map可以直接erase 但list不行 list要用迭代器 可以erasekey吗
        }
        //反正是插入，所以cache和pos是不用变了,pos位置还是要变得
        if(pos.find(key)!=pos.end()){
            recent.erase(pos[key]);
        }
        recent.push_front(key);
        pos[key]=recent.begin();
        //不管原先存不存在 都可以赋值，就是保证三个都变
        cache[key]=value;     
    }
};


```
##### 8 Leetcode 91
1、O(2^n)，DFS更适合求全部解码的结果，而不是计算有多少种解码的方法。
2、O(n)
DFS 显示超时。（自己写的，也不知道对不对）
```c
class Solution {
public:
    int numDecodings(string s) {
        int ans=0;
        dfs(0,s,ans);
        return ans;    
    }
    void dfs(int pos,string s,int & ans){
        if(pos>s.length()) 
            return;
        if(pos==(s.length()))
        {
            ans+=1;
            return;
        }
         if( s[pos]>='1' && s[pos]<='9')
        {           
            dfs(pos+1,s,ans);
        }
                
         if((s[pos]=='1' && s[pos+1] <='9') || (s[pos]=='2' && s[pos+1]>='0' && s[pos+1]<='6' ))
        {
            dfs(pos+2,s,ans);
        }
        
    }
};

```
动态规化
这个破题0好难处理啊，处理到这个数，如果这个数==0，就只能等于i-2了，就是这个只能是>10的了。如果这个数不是0，那么判断一下是i-1还是i-2啊。
```c
class Solution {
public:
    int numDecodings(string s) {
        int num=s.length();
        vector<int> mark;
        if(num==0) return 0;
        //对第一个数的处理
        if(s[0]>='1' && s[0]<='9')
            mark.push_back(1);
        else
            mark.push_back(0);
        //从第2个开始处理
        for(int i=1;i<num;i++){
            int tempans=0;
            if(s[i]=='0')
            {//这个数是0 前面的必须<=2 且 他的值就只等于-2的值了
                if(s[i-1]>='1' && s[i-1]<='2')
                if((i-2)<0)
                    tempans=1;
                else
                    tempans=mark[i-2];          
            }
            else{//这个数不是0，等于-1 和 -2的和
             if(s[i]>='1' && s[i]<='9')
                    tempans+=mark[i-1];
            if(s[i-1]=='1' || (s[i-1]=='2' && s[i]>'0' && s[i]<='6'))
            {
                if((i-2)<0)
                    tempans+=1;
                else
                    tempans+=(mark[i-2]);
            }              
            }
            mark.push_back(tempans);
        }
        return mark[num-1];
    }

};

```
别人写的动态规划就很简单啊！
```c
public:
    int numDecodings(string s) 
    {
        if (s.length() <= 0)
            return s.length();
        vector<int> dp(s.length() + 1, 0);
        dp[0] = 1;
        dp[1] = s[0]=='0'?0:1;
        for (int i = 2; i <= s.length(); i++)
        {
            if (s[i-1] != '0')//前一个不是0 i-1就加上去 前1个是0 就只能是i-2le 
                dp[i] = dp[i - 1];
            if (s[i - 2] != '0' && stoi(s.substr(i-2,2))<=26)
                dp[i] += dp[i - 2];
        }
        return dp[s.length()];
    }
```
##### 9 Leetcode 993 看是不是表兄弟节点，也就是层序遍历，万一是亲兄弟节点就错了
```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isCousins(TreeNode* root, int x, int y) {
        if(root==NULL) 
            return false;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            int mark1=0;
            int mark2=0;     
            int start=0;
            int len=q.size();
            //while(len--){
            while(start++<len){
                TreeNode * temp=q.front();
                cout<<temp->val<<" ";
                q.pop();
                ////////////////////////////
                //这里是我看别人的代码加的 然后就好了 别人可真厉害
                
                if(temp->left && temp->left->val==x && temp->right && temp->right->val==y)
                    return false;
                if(temp->left && temp->left->val==y && temp->right && temp->right->val==x)
                    return false;
                ///////////////////////////////
                if(temp->val==x )
                    mark1=1;
                if(temp->val==y)
                    mark2=1;
                if(temp->left) 
                {
                    q.push(temp->left);
                    
                }
                if(temp->right) 
                {
                    q.push(temp->right);
                }

                
            }
            cout<<endl;
                if(mark1==1 && mark2==1)
                    return true;
        }
        return false;
        
    }
};
```

##### 10 Leetcode 547 直接朋友间接朋友的朋友圈个数
my_batch_num22
```c
class Solution {
public:
    int findCircleNum(vector<vector<int>>& M) {
        int count=0;
        vector<bool > mark (M.size(),false);
        for(int i=0;i<M.size();i++){
            if(mark[i]==false)
            {  //如果i 没被访问过，就是新的一个朋友圈了，就++，其实主要的功能还是还是把一个朋友圈内的置为true
                dfs(M,mark,count,i);
                count++;
            }
        }
        return count;
    }
    
    void dfs(vector<vector<int>>& M,vector<bool> & mark,int & count,int k)
    {
        mark[k]=true;
        for(int i=0;i<M.size();i++){
            if(M[i][k]==1 && mark[i]==false)
                dfs(M,mark,count,i);
        }
    }
};
```

##### 11 Leetcode200 岛屿个数
DFS,我自己写的
```c
class Solution {
public:
    void dfs(vector<vector<char>>& grid, vector<vector<bool> > &mark, int m, int n){
        mark[m][n]=true;
        if((m-1) >=0  && mark[m-1][n]==false && grid[m-1][n]=='1')
            dfs(grid,mark,m-1,n);
        if((m+1) <grid.size()&& mark[m+1][n]==false && grid[m+1][n]=='1')
            dfs(grid,mark,m+1,n);
        if((n-1) >=0 && mark[m][n-1]==false && grid[m][n-1]=='1')
            dfs(grid,mark,m,n-1);
        if((n+1) <grid[0].size() && mark[m][n+1]==false && grid[m][n+1]=='1')
            dfs(grid,mark,m,n+1);
    
    }
    int numIslands(vector<vector<char>>& grid) {
        
        if(grid.size()==0) return 0;
        //vector<vector<bool> > mark(false,sizeof(grid));
        vector<vector<bool> > mark (grid.size());
        for(int i=0;i<grid.size();i++)
            mark[i].resize(grid[0].size());
        int count=0;
        
        for(int i=0;i<mark.size();i++)
            for(int j=0;j<mark[0].size();j++)
                mark[i][j]=false;
        
        int m=grid.size();
        int n=grid[0].size();
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(mark[i][j]==false && grid[i][j]=='1')
                {
                    dfs(grid,mark,i,j);
                    count++;
                }
                    
            }
        }
        return count;
    }
    
    
    
    
};
```
DFS 别人的代码 dfs就是一条路走到黑，当某个节点的临接节点都访问过了，回退到上一个节点。
```c
class Solution {
// 抹掉就好了，不用mark来记录了
public:
    void dfs(vector<vector<char>>& grid, int m, int n){

        grid[m][n]='0';
        if((m-1) >=0   && grid[m-1][n]=='1')
            dfs(grid,m-1,n);
        if((m+1) <grid.size() && grid[m+1][n]=='1')
            dfs(grid,m+1,n);
        if((n-1) >=0 &&  grid[m][n-1]=='1')
            dfs(grid,m,n-1);
        if((n+1) <grid[0].size() && grid[m][n+1]=='1')
            dfs(grid,m,n+1);
    
    }
    int numIslands(vector<vector<char>>& grid) {       
        if(grid.size()==0) return 0;
        int count=0; 
        int m=grid.size();
        int n=grid[0].size();
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]=='1')
                {
                    dfs(grid,i,j);
                    count++;
                }
                    
            }
        }
        return count;
    }     
};
```
BFS
节点入队，相邻节点再入队。其实BFS和DFS都是访问过的置为0 就行了。
```c
class Solution {

public:
    int numIslands(vector<vector<char>>& grid) {    
        if(grid.size()==0) return 0;
        int count=0;
        for(int i=0;i<grid.size();i++)
            for(int j=0;j<grid[0].size();j++)
            if(grid[i][j]=='1'){
                count++;
                bfs(grid,i,j);
            }
        return count;

    }
    void bfs(vector<vector<char>>& grid,int m,int n)
    {
        grid[m][n]='0';
        queue<vector<int>>  q;
        q.push({m,n});
        while(!q.empty()){
            int x=q.front()[0];
            int y=q.front()[1];
            q.pop();
            if( (x-1)>=0 && grid[x-1][y]=='1')
            {
                //bfs(grid,x-1,y);
                q.push({x-1,y});
                grid[x-1][y]='0';
            }
            if( (x+1)<grid.size() && grid[x+1][y]=='1')
            {
                //bfs(grid,x+1,y);
                grid[x+1][y]='0';
                q.push({x+1,y});
            }
            if( (y-1)>=0 && grid[x][y-1]=='1')
            {
                //bfs(grid,x,y-1);
                grid[x][y-1]='0';
                q.push({x,y-1});
            }
            if( (y+1)<grid[0].size() && grid[x][y+1]=='1')
            {
                //bfs(grid,x,y+1);
                grid[x][y+1]='0';
                q.push({x,y+1});
            }
        }
    }
};
```

##### 12 求这个人所有的直接/简介粉丝数，比如（1，2）（2，3）（3，4）（2，4）（5，6）（1，5），1的粉丝数为5，2为2，3为1，4为0，5为1，6为0.
```c
# include <iostream>
#include <vector>
#include <algorithm> 
# include <unordered_map>
using namespace std;

void dfs(int key,int num[7][7],vector<int> & mark,int &ans)
{
    mark[key]=1;
    for(int i=1;i<7;i++)
    {
        if(num[key][i]==1 && mark[i]==0)
        {
            ans++;
            dfs(i,num,mark,ans);
        }
    }   
    
}

int main()
{
    vector<vector<int>> input;
    input.push_back({1,2});
    input.push_back({2,3});
    input.push_back({3,4});
    input.push_back({5,6});
    input.push_back({2,4});
    input.push_back({1,5});
    //input.push_back({4,5});
    unordered_map <int,int> m;
    int num[7][7];
    for(int i=1;i<7;i++)
     for(int j=1;j<7;j++)
        num[i][j]=0;
    
    for(int i=0;i<input.size();i++)
    {   
        int x=input[i][0];
        int y=input[i][1];
        num[x][y]=1;

    }
    // 正式开始 
    for(int i=1;i<7;i++)
    {
        int ans=0;
        vector<int> mark(7,0);
        dfs(i,num,mark,ans);
        cout<<ans<<endl;
    }
    return  0;
 } 
```

##### Leetcode207 判断课程有没有环
(1) 入度为0的入栈 判断ans的个数和input的个数一不一样
```c
class Solution {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        vector<int> degree(numCourses,0);//各个节点的入度
        stack<int> temp;
        //vector<int> res;
        int res=0;
        //计算入度
        for(int i=0;i<prerequisites.size();i++)
        {
            degree[prerequisites[i].first]++;
        }
        //入度为0入栈
        for(int i=0;i<numCourses;i++)
        {
            if(degree[i]==0)
                temp.push(i);
        }
        //计算
        while(!temp.empty())
        {
            int now=temp.top();
            temp.pop();
            res++;
            //res.push_back(now);
            for(int i=0;i<prerequisites.size();i++)
            {
                if(prerequisites[i].second==now)
                {
                    degree[prerequisites[i].first]--;
                    if(degree[prerequisites[i].first]==0)
                        temp.push(prerequisites[i].first);
                }
            }    
       }
      return numCourses==res;//.size();
    }
};
```
（2） DFS的方法
```c

```
##### 两个栈实现队列
```c
class MyQueue {
public:
    stack<int> a;
    stack<int> b;
    /** Initialize your data structure here. */
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        a.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        while(!a.empty())
        {
            int temp=a.top();
            a.pop();
            b.push(temp);
        }
        int temp=b.top();
        b.pop();
        while(!b.empty())
        {
            int temp=b.top();
            b.pop();
            a.push(temp);
        }
        return temp;
    }
    
    /** Get the front element. */
    int peek() {
        while(!a.empty())
        {
            int temp=a.top();
            a.pop();
            b.push(temp);
        }
        int peek=b.top();
        while(!b.empty())
        {
            int temp=b.top();
            b.pop();
            a.push(temp);
        }
        return peek;
        
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return a.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
/*
MyQueue queue = new MyQueue();

queue.push(1);
queue.push(2);  
queue.peek();  // returns 1
queue.pop();   // returns 1
queue.empty(); // returns false
*/
```

##### 两个队列实现栈
```c
class MyStack {
public:
    queue<int> a;
    queue<int> b;
    /** Initialize your data structure here. */
    MyStack() {
        
    }
    
    /** Push element x onto stack. */
    void push(int x) {
        a.push(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        while(a.size()!=1)
        {
            int temp=a.front();
            a.pop();
            b.push(temp);
        }
        int temp=a.front();
        a.pop();
        while(!b.empty())
        {
            int temp=b.front();
            b.pop();
            a.push(temp);
        }
        
        return temp;
        
    }
    
    /** Get the top element. */
    int top() {
        while(a.size()!=1)
        {
            int temp=a.front();
            a.pop();
            b.push(temp);
        }
        int temp=a.front();
        a.pop();
        b.push(temp);
        while(!b.empty())
        {
            int temp=b.front();
            b.pop();
            a.push(temp);
        }
        
        return temp;
        
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return a.empty();
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 MyStack stack = new MyStack();

stack.push(1);
stack.push(2);  
stack.top();   // returns 2
stack.pop();   // returns 2
stack.empty(); // returns false
 */
```
##### 链表找环
```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(head==NULL || head->next==NULL)
            return false;
        if(head==head->next) return true;
        ListNode* slow=head;
        ListNode* fast=head->next;
        while(slow!=fast && fast!=NULL && fast->next!=NULL)
        {
            slow=slow->next;
            fast=fast->next->next;
            if(slow==fast)
                return true;
        }
        //if(slow==fast)
          //  return true;
        //else
            return false;
        
    }
};
```
##### 找环的入口位置
```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(head==NULL) return NULL;
        
        ListNode * slow=head;
        ListNode * fast=head;
        while(fast!=NULL && fast->next!=NULL  )
        {
            slow=slow->next;
            fast=fast->next->next;
            if(slow==fast)
            {
                //cout<<"%%"<<endl;
                slow=head;
                while(slow!=fast)
                {
                    slow=slow->next;
                    fast=fast->next;
                }
                return slow;
            }
        }
        return NULL;
        
    }
};
```

##### 二叉搜索树最小公共父亲
```c


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        int minn=p->val;
        int maxx=q->val;
        
        if(p->val > q->val)
        {
            swap(minn,maxx);
        }
        while(root->val<minn || root->val>maxx)
        {
            cout<<root->val<<" "<<minn<<" "<<maxx<<endl;
            if(root->val==minn || root->val==maxx)
               return root;
            if(root->val>maxx)
                if(root->left) root=root->left;
            if(root->val<minn)
                if(root->right) root=root->right;
        }
        return root;
        
    }
};
```
##### 二叉搜索树 中序遍历，自己写错了好多哦
```c
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
         stack<TreeNode*> s;
         int ans=0;
         if(pRoot==NULL) return NULL;
         //s.push(pRoot);
         while(pRoot!=NULL || !s.empty()) //我这里写的proot-》left
         {
             while(pRoot!=NULL)
             {
                 s.push(pRoot);
                 pRoot=pRoot->left;
             }
             if(!s.empty())
             {
                 pRoot=s.top(); //我写的temp。。。。。
                 s.pop();
                 ans++;
                 if(ans==k)
                     return pRoot;
                 pRoot=pRoot->right;   
             }
         }
        return NULL;
    }

    
};
```

##### 之字形打印二叉树
```c
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    //奇数从左到右 偶数从右到左
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int> > ans;
            queue<TreeNode*> q;
            if(pRoot==NULL) return ans;
            q.push(pRoot);
            int index=0;
            while(!q.empty())
            {
                int len=q.size();
                index++;
                vector<int> tempp;
                for(int i=0;i<len;i++)
                {
                    TreeNode* temp=q.front();
                    q.pop();
                    tempp.push_back(temp->val);
                       if(temp->left) q.push(temp->left);
                       if(temp->right) q.push(temp->right);
                }
                if(index%2==0)
                    reverse(tempp.begin(),tempp.end());
                ans.push_back(tempp);
            }
     return ans;
        }
 
};
```


##### 哈希表去重复
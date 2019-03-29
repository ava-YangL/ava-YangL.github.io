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
 ##### 1 求平方根
 ##### 2 n个数找k大
<!--more-->
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




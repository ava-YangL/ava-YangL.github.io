---
title: DFS相关问题整理
date: 2019-03-28 18:55:11
categories: 
- Code
tags:
- DFS
- Leetcode
- 基础知识 
---
感觉遇到这种题自己总是不会，整理下来希望能长长脑子。
<!--more-->

### 1 电话号码问题,Leetcode17 （这个题我大概是会了）
![avater](1.jpg)

```c
class Solution {
    vector<string> phone={"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};//直接这样根据index进行定位就好啦，不用傻fufu的去写个map
    vector<string> ans;
    
public:
    //重点就是这个dfs函数能不能写对，参数有res temp pos digits
    //pos很重要，要来判断处理到那个按键了
    //所以递归的时候pos=pos+1，而且pos=size了，说明这个情况处理完了，可以把temp push到ans里了。
    void dfs(vector<string> res,string digits,int pos,string temp){ //处理按键的数字
        string cur;
        if(pos==digits.size())
            ans.push_back(temp);

       if(pos<digits.size()) //这个似乎也没啥用？？？
           cur=phone[digits[pos]-'0'];
           for(int i=0;i<cur.length();i++) //里面的这个循环是用来处理各种情况的，外面的参数里的东西pos是用来判断ans什么时候pushtemp的
           //所以里面的循环，是处理比如说2对应“abc”三种不同情况的
           {
               temp.push_back(cur[i]);
               dfs(res,digits,pos+1,temp);
               temp.erase(temp.begin()+temp.length()-1);//处理下一种情况的时候，肯定是要把上一种情况清空啊
           }
               
        }
        
    
    vector<string> letterCombinations(string digits) {
        if(digits.length()==0) return ans;
        string temp="";
        dfs(ans,digits,0,temp);
        return ans;
    }
};

```
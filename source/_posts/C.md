---
title: C++
date: 2019-04-12 16:32:52
tags:
---
## 1 关于多态
### 大定义
多态：一种形式多种状态
面向对象编程的两个主要特征：继承、多态
C++ 中是通过间接利用“指向父类”的指针或引用来操作其子类对象，从而达到多态的目的；如果直接操作某个实例，那么多态将无从谈起
具体来说，多态是通过动态绑定机制，达到在运行时确定实际被调用的是哪个子类类型，进而调用对应的 override 方法
<!--more-->
### 体现多态
 1 方法的重载：编译时的多态。
 2 父类引用接受子类对象：运行时多态。

### 分类
- 静态多态（早绑定）: 传参的不同，编译时的多态。
- 多态多态（晚绑定）：必须以封装和继承为基础。
 ### 一开始

#### 虚函数
```
Shape  calcArea() 构造函数Shape() 析构函数~Shape()
Circle：public Shape calcArea() 构造函数Circle() 析构函数~Circle()
Rect ：public Shape  calcArea() 构造函数Rect() 析构函数~Rect()
```
```

Shape *shape1=new Rect(3,0,5,0);
Shape *shape2=new Circle(4.0);
shape1->calcArea();
shape2->calcArea();
delete shape1；
shape1=null;
delete shape2；
shape2=null;
```
- 都不加virtual
Shape() Rect() Shape() Circle() Shape->calc Shape->calc ~Shape()  ~Shape() 

- 父类加上virtual（当然写在子类上更规范）
Shape() Rect() Shape() Circle() Rect->calc Circle->calc ~Shape()  ~Shape() 


#### 动态多态中内存泄漏的问题
-  父类指针想去销毁子类对象的时候，父类指针执行父类析构函数，子类析构函数中定义的需要delete的东西就delete不聊了
-  虚析构函数是为了避免使用父类指针释放子类对象时造成内存泄露。

#### 虚析构函数
- virtual ~Shape()，父类中写，子类中可写可不写。这样父类指针指向的是啥对象，就释放子对象
- 因为你不知道子类会new啥东西，所以析构函数最好还是加上virtual吧，保证子类被delete掉。

#### virtual不能用的情况
- 不能修饰普通函数，必须是类的成员函数
- 不能修饰静态成员函数，因为静态成员函数不属于对象，和类共生死
- 不能修饰内联函数，会忽略inline
- 不能修饰构造函数
 
#### 纯虚函数
 virtual double calc（）=0；
#### 抽象类
- 含有纯虚函数的类叫做抽象类
- 在面向对象的概念中，所有的对象都是通过类来描绘的，但是反过来，并不是所有的类都是用来描绘对象的，如果一个类中没有包含足够的信息来描绘一个具体的对象，这样的类就是抽象类
- 不能被实例化
- 派生类必须实现未实现的方法

#### 接口类

## 2 关于new
1 new创建对象，对象保存在堆还是栈？
- 堆内存是用来存放由new创建的对象和数组，即动态申请的内存都存放在堆内存
- 栈内存是用来存放在函数中定义的一些基本类型的变量和对象的引用变量

2 例子
- 局部变量存放在栈；new函数和malloc函数申请的内存在堆；函数调用参数，函数返回值，函数返回地址存放在栈

3 堆和栈的区别
- 栈区（stack）—   由编译器自动分配释放，存放函数的参数值，局部变量的值等。其操作方式类似于数据结构中的栈。  
- 堆区（heap） —   一般由程序员分配释放，若程序员不释放，程序结束时可能由OS回收   。注意它与数据结构中的堆是两回事，分配方式倒是类似于链表 ，呵呵。


4 Object may be visible invisible
Object =Attributes(Data)+ Services(Operations/functions)


5 VS对齐代码关键字， Ctrl+K Ctrl+F
6 关于头文件
```c
以下文件在"os_cpu.h"中。
#ifndef __OS_CPU_H__
#define __OS_CPU_H__ 
/*

中间有许多定义啦声明啦！；；

*/
#endif /*__OS_CPU_H__*/

这样，在编译阶段（ifndef是在编译阶段起作用滴！）假设有两个文件同时include了这个文件（os_cpu.h），这两个文件如果一个先编译了，那么__OS_CPU_H__就被定义了。当编译到第二个文件的时候，那么在开始的判断（ifnef）就会自动跳出os_cpu.h这个文件的重复编译。这样就避免了重复编译文件。。
--------------------- 
版权声明：本文为CSDN博主「thimin」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/thimin/article/details/1539960
```


--------------------- 
作者：WX_Chen 
来源：CSDN 
原文：https://blog.csdn.net/kl1411/article/details/65959992 
版权声明：本文为博主原创文章，转载请附上博文链接！
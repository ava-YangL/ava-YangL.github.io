---
layout: forward
title: Proxy
date: 2020-12-07 14:01:24
tags: Develop
---

Some notes about Forward Proxy, Reverse Proxy, Load Balance.
<!--more-->
#### Reference
[反向代理（Reverse Proxy）](https://www.jianshu.com/p/37dc1699489a)

### 1 Forward Proxy
允许一个网络中端通过Forward Proxy与另一个网路终端进行非直接的连接，e.g.科学上网时就是Forward Proxy.
<img src="4.PNG" width=500/>

服务边界：Forward Proxy Server 属于Client端, 简而言之，代理客户端
<img src="1.PNG" width=500/>


### 2 Reverse Proxy
服务边界：Reverse Proxy Server 属于Service端，简而言之，代理服务器
<img src="2.PNG" width=500/>
Reverse Proxy 是一个可以集中调用内部服务，并提供统一接口给公共客户的web服务器。来自客户端的请求先被反向代理服务器转发到可响应请求的服务器，然后代理再把服务器的响应结果返回给客户端。

反向代理根据客户端的请求，从其关联的一组或多组后端服务器上获取资源，然后再将这些资源返回给客户端；**客户端只知道反向代理的IP地址**，而不需知道代理服务器身后服务器集群的存在。

##### 反向代理主要有以下好处：
- 增加安全性：隐藏后端服务的信息，设置黑白名单
- 提高可扩展性和灵活性：客户端只能看到反向代理服务器的IP，运维只需通过配置增减服务
- 终结SSL会话：解密传入请求和加密服务器响应；后端服务器就不必完成这些额外的操作
- 压缩、缓存、添加请求头等一些附加功能。

##### 不利之处：
- 增加**复杂度**
- 延迟
- 单点故障 ??

### 3 Compare
正向代理隐藏客户端，反向代理隐藏服务器


### 4 Load Balance
通常，LB将流量路由给一组**功能相同**的服务器。它可以通过硬件或HAProxy等软件来实现流量分发；一般基于取模、轮询、随机等算法路由流量，高级点还会根据加权、负载等等进行调度。Reverse Proxy一般根据不同的API分发到**不同功能**的服务上。
<img src="3.PNG" width=500/>





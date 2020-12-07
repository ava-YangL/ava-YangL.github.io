---
title: stateless-vs-stateful
date: 2020-12-07 14:42:06
tags: Develop
---

Some notes about stateless and stateful.
<!--more-->


##### Reference
[stateful, stateless, cookie and session](https://sethuramanmurali.wordpress.com/2013/07/07/stateful-stateless-cookie-and-session/)
[Statefulness in a Stateless Web](https://medium.com/@t0ri/being-stateful-in-a-stateless-web-8185ed777048)

#### Introduction

**stateful** – keep track of the previously stored information which is used for current transaction.

- means the computer or program keeps track of the state of interaction, **usually by setting values** in a storage field designated for that purpose.

**stateless** – every transaction is performed **as** if it were being done for **the very first time**. There is **no previously stored information** used for the current transaction.

- means there is **no record** of previous interactions and each interaction request has to be handled based **entirely on information that comes with it**.


#### Example

The Internet’s basic protocol, the Internet Protocol ( **IP** )

- is an example of a **stateless** interaction
- Each packet travels entirely on its own without reference to any other packet

The Web’s Hypertext Transfer Protocol ( **HTTP** ), an application layer above TCP/IP, is also **stateless**

#### Cookie

In order to have stateful communication, a site developer must furnish  **a special program** that the **server can call** that can **record and retrieve state information**. Web browsers such as Netscape Explorer and *Microsoft Internet Explorer provide an area in their sub-directories where state information* can be stored and accessed. The area and the information that Web browsers and server applications put in this area is called a **cookie**.

Cookies are small, **string-filled** text files. Session cookies expire when you close your browser, and are useful for things like remembering who’s logged in.

We can **use cookies to remember our application’s session state!** However, because cookies are stored **locally**, they still aren’t the best option for storing data in your web apps. Users can view and manipulate cookie data, which could be bad for your site. So, how can we safely store state client-side?

The preferred solution is to use cookies to store a Session Id 🏷️ locally, but save the session data on server-side.(To read the second article in reference.)

#### Session
Imagine a request over the web where you have a client browser communicating to a server process. To **maintain state over the stateless http protocol the browser typically send a session identifier to the server on each request.** For each request the server will be like “ah, its this guy”. State information can then be looked up in server side memory or in a database based on this session id.


#### Request-Response Cycle
When you opened this article, your **browser (the client)** sent a request to Medium’s servers that loosely said, “Hey, GET me this article please!”. Then Medium responded with the files required for your browser to render what you’re reading right now. This communication between the client and server is called the Request-Response Cycle.



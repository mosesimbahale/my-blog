---
layout: post
title:  "React: MERN JWT Aunthentication & Authorization"
date:   2022-01-30 
category: tech
---

### React.js + Node.js Express + MongoDB


In this tutorial, we will learn how to build a full stack MERN JWT authentication with example.
The back-end server uses Node.js Express with jsonwebtoken for JWT authentication & authorizatio, Mongoose for interacting with MongoDB database. The front-end will be created with react, react router & axios. We will also use Bootstrap and perform form validation.


##### JWT(JSON Web Token)
Comparing JWT with Session based authentication that need to storee session cookie, the big advantage of the token-based authentication is that we store the Jsoon web token (JWT) on client side: Local storage for browser, keychain for IOS and shared preferances for android so we dont need to build another backend project that supports native appps or an additional authentication module for Native app users.

There are three important parts of a JWT:
- Header 
- Payload
- Signature

All togather they are combined to a standard structure: `Header.payload.signaure`

The client typically attaches JWT in x-access-token header:

`x-access-token: [header].[payload].[signature]`







##### 01 Login page

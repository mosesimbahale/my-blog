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




#### Example application

It will be a fullstack MERN authentication, with Node.js Express for back-end and React.js for front-end. The access is verified by JWT Authentication

- Users can signup new account, login with username & password.
- Authorization by the role of the user(Admin, Moderator, User)
- Anyone can access the public page before logging in
- After logging in, app directs the user to profile page
- UI for authorization login(the navigator bar will change by authorities)
- If a user who doesnt have admin roles tries to access Admin/Moderator Board page:




#### Flow for user registration and user login
The diagram shows flow of  the user registration, user login and authorization process 

There are two endpoints for authentication process.

- api/auth/signup for user registration
- api/auth/signin for user login

If client wants to send request to protected data/endpounts, it add legal JWT to HTTP ***x-access-token*** header


#### Back-end with Node.js & MongoDB
##### Overview

Our Node.js Express application can be summarized in the diagram below:














Via Express routes, **HTTP request** that matches a route will be checked by **CORS Middleware** before coming to security layer.

Security layer includes:

- JWT authentication Middleware: Verify signup, verify token
- Authorization middleware: check user's roes with record in MongoDB database 

An error message will besent as HTTP response to client when the middlewares throw any error.

**Controllers** interact with MongoDB Dtabase via mongoose library and send HTTP response(token, user information, data based on roles...) to client.


#### Technology

- Express 
- Bcryptjs
- jsonwebtoken
- Mongoose
- MongoDB

#### Project structure
This is the directory structure for our Node.js Express & MongoDB application:

- **config**
 - Configure MongoDB database connection
 - Configure Auth Key
- **routes**    
 - aut.routes.js: POST signup & signin
 - user.routes.js: GET public and protected resources
- **middlewares**
 - verifySignup.js: Check duplicate username or Email
 - authJWT.js:verify token, check user roles in MongoDB database
- **Controllers**
 - auth.controller.js: handle signup & signin actions
 - user.controller.js: return public & protected content
- **models** for mongoose models
 - user.model.js
 - role.model.js   

- Server.js: Import and initialize necessary modules and routes, listen for connections.



#### Implementation





















### Front-end with React, React Router
##### Overview

Let's look at the diagram below












- The `app` components is a container with react router(`browserRouter`). Basing on the state, the navbar can display its items.

- Login & Register components have form for data submission(with support of `react-validation` library). They call methods form auth.services to make login/register request.


- `auth.services` methods use axios to make HTTP requests. Its also store or get JWT form browser local storage inside these methods.

- `Home component` is public for all visitors.

- `profile` component displays user information after the login action is successful.

- `BoadUser`, `BoardModerator`, `BoardAdmin` components will be displayed by state `user.roles`. In these components,we use `user.service` to access protected resources from web API.

- `User.service` uses `auth-header()` helper function to add JWT to HTTP header. `auth-header()` returns an object containing the JWT of the currently logged in user from local storage.


#### Technology
- React
- React-router-dom 5
- Axios
- react-validation
- Bootstrap 4
- validator 12.2.0

#### Project structure
This is the folders for this react application:











With the explanation in the diagram above you can understand the project structure easily.

#### Implementation































##### 
Now we have an overview of MERN authentication with JWT example by building registration & login page using react.js, MongoDB, Node.js Express.

We also take a look at Node.js Express server archiitecture using jsonwebtoken & Mongoose, as well as react.js project structure for building a front-end app working with JWT.













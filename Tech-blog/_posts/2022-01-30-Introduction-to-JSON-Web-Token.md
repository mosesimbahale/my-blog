---
layout: post
title:  "React: in-depth introduction to JSON Web Token"
date:   2022-01-30 
category: tech
---

### Session-based Authentication VS Token-based Authentication

For using any website, mobile app or desktop app.. you almost always need an account, then use it to login for accessing features of the application. We call that authentication.

So how do we authenticate an account?

<<<<<<< Updated upstream
First, we are goint to take a look at a simple method that is popularly used in many websites: ***Session-based authentication***

![image](https://user-images.githubusercontent.com/42868535/151693443-e583a43f-f93b-4182-a486-db883063d836.png)


![image](https://user-images.githubusercontent.com/42868535/151693460-0ed7c836-ed29-4a44-b32b-09229a0b4265.png)
=======
First, we are going to take a look at a simple method that is popularly used in many websites: ***Session-based authentication***
>>>>>>> Stashed changes


In the image above,when a user logs into a website, the server will generate a `session` for that user and store it(in memory or database). Server also returns a `sessionId` for the client to save it in browser cookie.

The session on server has an expiration time. After that time, this session has expired and the user must re-login to create another session.

If the user has logged in and the sesssion has not expired yet, the cookie(Including sessionId) always goes with all HTTP requests to server. Server will compare this `sessionId` with stored `session` to authenticate and return response.

I'ts ok but why do we need Token-based authentication?

The answer is we don't have only website, there are many platforms over there.

Assume that we have a website which works well with session. One day, we want to implement system for mobile (Native apps) and use the same database with the current web app. What should we do? We cannot authenticate users who use native app using session based authentication because these kind don't have Cookie.

Should we build another backend project that support native apps? Or should we write an authentication module for native app users>

That's why token based authentication was born.

With this method, the user login state is encoded into a JSON web token by the server and send to the client. Nowadays many REESTful APIs use it. Let's go to the next section, we are going to know how it works.

You can see that it's simple to understand. Instead of creating a session, the server generated a JWT form user login data and send it to the client. The client saves the JWT and from now, every request form client should be attached that JWT (commonly at header) The server will validate the JWT and return the response.

For strong JWT on client side, it depends on the platform you use:

- Browser: Local storage.
- IOS: Keychain
- Android: sharedPreferences

That is overview of a token based authentication flow. You will understand it more deeply with the next section.


#### How to create a JWT

First, you should know three important parts of a JWT:
- Header
- Payload
- Signature

##### Header
The header answers the question: How will we calculate the JWT?
Now lookk at the exaple of `header`, it's a JSON object like this:

```
{
    "typ" : "JWT"
    "alg" : "HS256"
}

```

- `typ` is 'type', indicates that the token type here is JWT.
- `alg` stands for  'algorithm' which is a hash algorithm for generating Token `signature` In the code above, `HS256` is HMAC-SHA256 - the algorithm which uses secret key.

##### Payload
The payload helps us to answer: What do we want to store in JWT?

This is a payload sample:

```
  "userId": "abcd12345ghijk",
  "username": "moses",
  "email": "contact@bezkoder.com",
  // standard fields
  "iss": "zKoder, author of bezkoder.com",
  "iat": 1570238918,
  "exp": 1570238992

```

In the JSON object above, we store 3 user fields: `userId` , `username`, `email`. You can save anyfield you want.

We also have some `standaed fields` They are optional.

- iss (Issuer): who issues the JWT
- iat (Issued at): time the JWT was issued at
- exp (Expiration Time): JWT expiration time

##### Signature
This part is where we use the Hash algorithm.
Look at the code for getting the signature below:

```
const data = Base64UrlEncode(header) + '.' + Base64UrlEncode(payload);
const hashedData = Hash(data, secret);
const signature = Base64UrlEncode(hashedData);

```

Let's explain it.
- First, we encode Header and Payload, join them with a dot .

`data = '[encodedHeader].[encodedPayload]'`

- Next, we make a hash of the data using Hash algorithm (defined at header) with a `secret` string.

- Finally, we encode the hashing result to get `Signature`

##### Combine all things
After having header, payload, signature, we're going to combine them into JWT standard structure: `header.payload.signature`

Following code will illustrate how we do it.

```
const encodedHeader = base64urlEncode(header);
/* Result */
"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"

const encodedPayload = base64urlEncode(payload);
/* Result */
"eyJ1c2VySWQiOiJhYmNkMTIzNDVnaGlqayIsInVzZXJuYW1lIjoiYmV6a29kZXIiLCJlbWFpbCI6ImNvbnRhY3RAYmV6a29kZXIuY29tIn0"

const data = encodedHeader + "." + encodedPayload;
const hashedData = Hash(data, secret);
const signature = base64urlEncode(hashedData);
/* Result */
"crrCKWNGay10ZYbzNG3e0hfLKbL7ktolT7GqjUMwi3k"

// header.payload.signature
const JWT = encodedHeader + "." + encodedPayload + "." + signature;
/* Result */
"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VySWQiOiJhYmNkMTIzNDVnaGlqayIsInVzZXJuYW1lIjoiYmV6a29kZXIiLCJlbWFpbCI6ImNvbnRhY3RAYmV6a29kZXIuY29tIn0.5IN4qmZTS3LEaXCisfJQhrSyhSPXEgM1ux-qXsGKacQ"

```


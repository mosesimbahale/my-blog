---
layout: post
title:  "How to connect react with express backend"
date:   2022-01-27 
category: tech
---


#### Inroduction

In your terminal, navigate to a directory where you would like to save your project. Now create a new directory for your project and navigate into it:

```
mkdir my_awesome_project
cd my_awesome_project

```

**create a react app**
This process is really straightfoward.

```
npx create-react-app frontend
cd frontend
npm start

```

In your browser navigate to `http://localhost:3000/`

If its done correctly, you will see the react welcome page. That means you have a basic react react application runnng on a local machine.

To stop your react app, just press `Ctrl + c` in your terminal.

**Create an express app**
- Navigate to your project top folder
- We will be using the ExpressApplicationGenerato to quickly create an application skeleton and name it api:

```
npx express-generator api
cd api
npm install
npm start

```
If its done correctly you will see the express welcome page. That means now you have a basic express application running on your local machine.

To stop your react app, just press `Ctrl + c` in you terminal


**Configuring a new route in the express API**

### 01
Inside the api directory, go to `bin/www` and change the port number on line 15 from 3000 to 9000. We will be running both apps at the same time later on, so doing this will avoid issues.

### 02
On api/routes, create a testAPI.js file and paste this code:

```
var express = require(“express”);
var router = express.Router();

router.get(“/”, function(req, res, next) {
    res.send(“API is working properly”);
});

module.exports = router;

```

### 03
On the api/app.js file, insert a new route on line 24:

`app.use("/testAPI", testAPIRouter);`


### 04
Ok, you are "telling" express to use this route but, you still have to require it. Lets do that on line 9:

`var testAPIRouter = require("./routes/testAPI");`


### 05 
Congratulations! you have created a new route

If you start your API app (in your terminal, navigate to the API directory and “npm start”), and go to http://localhost:9000/testAPI in your browser, you will see the message: API is working properly.


Connecting the react frontend to the express api

1. On your code editor, let’s work in the frontend directory. Open app.js file located in frontend_project/client/app.js.

2. Here I will use the Fetch API to retrieve data from the API. Just paste this code after the Class declaration and before the render method:



```
constructor(props) {
    super(props);
    this.state = { apiResponse: "" };
}

callAPI() {
    fetch("http://localhost:9000/testAPI")
        .then(res => res.text())
        .then(res => this.setState({ apiResponse: res }));
}

componentWillMount() {
    this.callAPI();
}

```

3. Inside the render method, you will find a <;p> tag. Let’s change it so that it renders the apiResponse:


##### Recap:

- On lines 6 to 9, we inserted a constructor, that initializes the default state.

- On lines 11 to 16, we inserted the method callAPI() that will fetch the data from the API and store the response on this.state.apiResponse.

- On lines 18 to 20, we inserted a react lifecycle method called componentDidMount(), that will execute the callAPI() method after the component mounts.

- Last, on line 29, I used the <;p> tag to display a paragraph on our client page, with the text that we retrieved from the API.

#### CORS

We are almost done. But if we start both our apps (client and API) and navigate to http://localhost:3000/, you still won't find the expected result displayed on the page. If you open chrome developer tools, you will find why. In the console, you will see this error:


> Failed to load http://localhost:9000/testAPI: No
> ‘Access-Control-Allow-Origin’ header is present on
> the requested resource. Origin
> ‘http://localhost:3000' is therefore not allowed
> access. If an opaque response serves your needs, set
> the request’s mode to ‘no-cors’ to fetch the resource
> with CORS disabled.

This is simple to solve. We just have to add CORS to our API to allow cross-origin requests. Let’s do just that. You should check here to find out more about CORS.

- In your terminal navigate to the API directory and install the CORS package:

`npm install --save cors`

- On your code editor go to the API directory and open the my_awesome_project/api/app.js file.

- On line 6 require CORS:

`var cors = require("cors");`

- Now on line 18 “tell” express to use CORS:

`app.use(cors());`


Great Work. It’s all done!!
Ok! We are all set

Now start both your apps (client and API), in two different terminals, using the npm start command.


If you navigate to http://localhost:3000/ in your browser you should find something like this:
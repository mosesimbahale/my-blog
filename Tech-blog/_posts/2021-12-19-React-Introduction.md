---
layout: post
title:  "React: Getting started"
date:   2021-12-19 
category: tech
---



Hello and welcome!

## **What's in our React App**

Let's take a look of what our React App looks like right now. We will go through our file structure which is a [standard React setup.](https://create-react-app.dev/docs/getting-started/) You will not be editing any files in this step, but the structure is important for future code references.

`package.json`

The `package.json` file is our roadmap of the app. It tells us the name of the app, the version, the different dependencies we need to run our app, and more.

`public/`

The `public/` directory contains our `index.html` file. The index.html file directs us to the rest of the web application that requires additional processing.

`src`

This is where most of your code will go. You'll notice we have `App.jsx` along with other `jsx` files. But, what is `jsx`? Think of `jsx` as a mix between html and javascript. We can write tags, functions, and classes. Take a look at the App.jsx file. Some of that contents might look familiar from html, like `<div/>` tags.

## **Step 01: Setup react application Locally.**

**⌨️ Activity: Clone [this](https://github.com/mosesimbahale/intro-react) repository and install Node**

1. Open your terminal
- If you're using a Windows operating system, we recommend downloading and using git bash as your terminal
2. Change directories into a location where you want to keep this project
3. Clone this repository: git clone `link`
4. Move into the repository's directory: `cd intro-react`
5. Checkout to the `changes` branch: `git checkout changes`
6. Open the `intro-react` folder in your favorite text editor
7. Check for Node installation: `node -v`
8. Install Node if it is not installed
9 In your repository folder, install the necessary packages: `npm install`
10. Start the React Application: `npm start`

Your browser should automatically open `http://localhost:3000/`, and you should see our barebones gradebook application.

You'll see that our app can't really do anything! All we have is three buttons! We are going to add all the functionality.

**⌨️ Activity: Open a Pull Request**


## **Building Blocks of Life Apps - Components**

In the Pull Request you will:

Add components
Pass data to child components
Create and use state variables
Use callback functions to communicate data
First, we'll learn about components.

**Components**

Components are the building blocks of React apps. Think of components as different parts of the app. Each button is a component, each paragraph of text is a component, and so on. From html, you might recognize some of the built in components like `<div />` and `<li />`. In React, we can create our own components! How cool is that?

Components in `src/App.jsx`


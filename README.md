# SVM
### CS760 Project Site

This contains the Jekyll site for our cs760 project.

## What is this?
This is a jekyll based-blog. Jekyll is a ruby-based framework for generating blogs. You write posts in markdown and jekyll takes care of the html generation automaticall. Jekyll is endorsed by Github and many github pages sites are made with Jekyll. There's tons of tutorials around, so this README will only provide instructions for getting started with jekyll and the post-writing workflow.

## Install Jekyll
You'll need:
* ruby
* gem package manager (`gem -v`)

```bash
gem install bundler
# installs all the dependencies of this project (including jekyll)
bundler update
jekyll -v
```

## How to write a post

```bash
jekyll serve
# open your browser to localhost:4000, you should see the site
# any edits you make will immediately be picked up and updated 
# when you refresh your page
```

## How to publish

Once you have written and previewed your post, you will need to commit and push any changes you made to both your source `.md` files and compiled `.html` files.

```bash
jekyll build
git add -u

# if needed, add new content
git add content/* 

git commit -m 'my new post'
git push upstream
```

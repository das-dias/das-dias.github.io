This is a simple personal landing page containing my personal and professional information. This web-based *curriculum vitae* was built on top of [MkDocs](https://www.mkdocs.org/).

# Deployment:

Run `mkdocs build` to build the website from the source repo, and then push the built website source code into a new repo branch `gh-pages` using `mkdocs gh-deploy`. 

This new branch only contains the built static website source code, and it shares no similar history with the `main` branch. This cannot be merged together.

Now, it is required to configure GitHub Pages to deploy the website from the source of the new `gh-pages`. For this, go to `Settings > Pages > Deploy from branch` and select the new `gh-pages / (root)` branch.

Voi lá.

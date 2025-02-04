// For more info: https://stackoverflow.com/questions/27882261/mkdocs-and-mathjax

MathJax.Hub.Config({
  "tex2jax": { inlineMath: [ [ '$', '$' ] ] }
}); // inline $ $ code resolver
MathJax.Hub.Config({
  config: ["MMLorHTML.js"],
  jax: ["input/TeX", "output/HTML-CSS", "output/NativeMML"],
  extensions: ["MathMenu.js", "MathZoom.js"]
});
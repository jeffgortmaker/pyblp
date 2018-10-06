$(document).ready(function() {
    // modify MathJax settings so that the generated text looks more like the text surrounding it
    MathJax.Hub.Config({
        'HTML-CSS': {
            matchFontHeight: false,
            fonts: ['Latin-Modern', 'TeX']
        }
    });

    // wrap API links created in notebooks with missing code formatting
     $('.document a[href*="pyblp."]:has(span):not(:has(code))').wrapInner('<code class="xref py py-meth docutils literal notranslate">');
});

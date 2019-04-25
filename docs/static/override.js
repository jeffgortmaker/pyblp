// wrap API links created in notebooks with missing code formatting
$(document).ready(function() {
     $('.document a[href*="pyblp."]:has(span):not(:has(code))').wrapInner('<code class="xref py py-meth docutils literal notranslate">');
});

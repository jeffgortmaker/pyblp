// modify MathJax settings so that the generated text looks more like the text surrounding it

$(document).ready(function(){
    MathJax.Hub.Config({
        'HTML-CSS': {
            matchFontHeight: false,
            fonts: ['Latin-Modern', 'TeX']
        }
    });
});

/*!
 * Clean Blog v1.0.0 (http://startbootstrap.com)
 * Copyright 2015 Start Bootstrap
 * Licensed under Apache 2.0 (https://github.com/IronSummitMedia/startbootstrap/blob/gh-pages/LICENSE)
 */

 /*!
 * Hux Blog v1.6.0 (http://startbootstrap.com)
 * Copyright 2016 @huxpro
 * Licensed under Apache 2.0 
 */

// Tooltip Init
// Unuse by Hux since V1.6: Titles now display by default so there is no need for tooltip
// $(function() {
//     $("[data-toggle='tooltip']").tooltip();
// });


// make all images responsive
/* 
 * Unuse by Hux
 * actually only Portfolio-Pages can't use it and only post-img need it.
 * so I modify the _layout/post and CSS to make post-img responsive!
 */
// $(function() {
//  $("img").addClass("img-responsive");
// });

// responsive tables
$(document).ready(function() {
    $("table").wrap("<div class='table-responsive'></div>");
    $("table").addClass("table");
});

// responsive embed videos
$(document).ready(function() {
    $('iframe[src*="youtube.com"]').wrap('<div class="embed-responsive embed-responsive-16by9"></div>');
    $('iframe[src*="youtube.com"]').addClass('embed-responsive-item');
    $('iframe[src*="vimeo.com"]').wrap('<div class="embed-responsive embed-responsive-16by9"></div>');
    $('iframe[src*="vimeo.com"]').addClass('embed-responsive-item');
});

// MathJax block formula horizontal scrolling support
$(document).ready(function() {
    // Function to wrap MathJax display formulas with scrollable container
    function wrapMathJaxFormulas() {
        // Wait for MathJax to finish rendering
        if (typeof MathJax !== 'undefined') {
            MathJax.Hub.Queue(function() {
                // Wait a bit more to ensure MathJax is fully rendered
                setTimeout(function() {
                    // Find all MathJax display formulas
                    $('.MathJax_Display').each(function() {
                        var $mathDisplay = $(this);
                        
                        // Check if already wrapped
                        if (!$mathDisplay.parent().hasClass('math-display-container')) {
                            // Wrap the formula with scrollable container
                            $mathDisplay.wrap('<div class="math-display-container"></div>');
                            
                            // Add scroll indicator for long formulas
                            var container = $mathDisplay.parent();
                            
                            // Calculate widths after wrapping
                            setTimeout(function() {
                                var mathWidth = $mathDisplay.outerWidth(true);
                                var containerWidth = container.width();
                                
                                if (mathWidth > containerWidth) {
                                    container.addClass('has-scroll');
                                }
                            }, 100);
                        }
                    });
                }, 500);
            });
        }
    }
    
    // Initial wrap after page load
    setTimeout(function() {
        wrapMathJaxFormulas();
    }, 1000);
    
    // Re-wrap on window resize
    $(window).on('resize', function() {
        wrapMathJaxFormulas();
    });
    
    // Listen for MathJax typesetting complete
    if (typeof MathJax !== 'undefined') {
        MathJax.Hub.Register.MessageHook("End Process", function() {
            setTimeout(function() {
                wrapMathJaxFormulas();
            }, 300);
        });
    }
});

// Navigation Scripts to Show Header on Scroll-Up
jQuery(document).ready(function($) {
    var MQL = 1170;

    //primary navigation slide-in effect
    if ($(window).width() > MQL) {
        var headerHeight = $('.navbar-custom').height(),
            bannerHeight  = $('.intro-header .container').height();     
        $(window).on('scroll', {
                previousTop: 0
            },
            function() {
                var currentTop = $(window).scrollTop(),
                    $catalog = $('.side-catalog');

                //check if user is scrolling up by mouse or keyborad
                if (currentTop < this.previousTop) {
                    //if scrolling up...
                    if (currentTop > 0 && $('.navbar-custom').hasClass('is-fixed')) {
                        $('.navbar-custom').addClass('is-visible');
                    } else {
                        $('.navbar-custom').removeClass('is-visible is-fixed');
                    }
                } else {
                    //if scrolling down...
                    $('.navbar-custom').removeClass('is-visible');
                    if (currentTop > headerHeight && !$('.navbar-custom').hasClass('is-fixed')) $('.navbar-custom').addClass('is-fixed');
                }
                this.previousTop = currentTop;


                //adjust the appearance of side-catalog
                $catalog.show()
                if (currentTop > (bannerHeight + 41)) {
                    $catalog.addClass('fixed')
                } else {
                    $catalog.removeClass('fixed')
                }
            });
    }
});

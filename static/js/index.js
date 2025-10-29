window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "https://storage.googleapis.com/nerfies-public/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
var imagesLoaded = false;

// Disable preloading to prevent 403 errors from inaccessible Google Cloud Storage
function preloadInterpolationImages() {
  // Disabled: Images are no longer accessible from Google Cloud Storage
  // This function is kept for compatibility but does nothing
  imagesLoaded = false;
  console.log('Interpolation image preloading disabled - images not accessible');
}

function setInterpolationImage(i) {
  // Only load image if interpolation-image-wrapper exists and image is requested
  var wrapper = $('#interpolation-image-wrapper');
  if (wrapper.length === 0) {
    return; // Element doesn't exist, skip
  }
  
  // Check if we have the image cached, if not, don't try to load it
  if (!interp_images[i]) {
    // Don't attempt to load from inaccessible URL
    return;
  }
  
  var image = interp_images[i];
  if (!image || !image.complete) {
    return; // Image not loaded or failed to load
  }
  
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  wrapper.empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    
    // Only initialize interpolation if the slider exists
    var interpolationSlider = $('#interpolation-slider');
    if (interpolationSlider.length > 0) {
      // Preloading disabled due to 403 errors on Google Cloud Storage
      // preloadInterpolationImages();
      
      interpolationSlider.on('input', function(event) {
        // Only set image if wrapper exists and image is available
        var wrapper = $('#interpolation-image-wrapper');
        if (wrapper.length > 0) {
          setInterpolationImage(this.value);
        }
      });
      
      // Don't try to set initial image if images aren't loaded
      // setInterpolationImage(0);
      interpolationSlider.prop('max', NUM_INTERP_FRAMES - 1);
    }

    bulmaSlider.attach();

})

var imgs = document.getElementsByClassName("ad-img");
for (var i = 0; i < imgs.length; i++) {
    imgs[i].addEventListener("click", function() {
        var current = document.getElementsByClassName("active");
        if (current.length > 0) { 
            current[0].className = current[0].className.replace(" active", "");
        }
        this.className += " active";
    });
    }

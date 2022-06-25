function toggleFunc(x) {
    x.classList.toggle("fa-user-tie");
    var heading = document.getElementById("userProfile")
    var content = document.getElementById("profileDiv")
    if (heading.innerHTML==="Athlete"){
        heading.innerHTML = "Banker";
    }
    else{
        heading.innerHTML = "Athlete";
    }
  }
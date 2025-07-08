function showSending(event) {
    event.preventDefault();
    const btnText = document.getElementById("btnText");
    const dots = document.getElementById("dots");
    const arrow = document.getElementById("arrowIcon");

    btnText.textContent = "Verifying";
    dots.style.display = "inline-block";
    arrow.style.display = "none";

    setTimeout(() => {
        event.target.submit(); // Actually submits the form
    }, 2500);
}
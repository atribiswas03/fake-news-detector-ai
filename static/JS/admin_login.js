function showSending(event) {
    event.preventDefault();
    const btnText = document.getElementById("btnText");
    const dots = document.getElementById("dots");
    const arrow = document.getElementById("arrowIcon");

    btnText.textContent = "Sending";
    dots.style.display = "inline-block";
    arrow.style.display = "none";

    // Simulate real sending for demo (remove setTimeout in actual use)
    setTimeout(() => {
        event.target.submit(); // Actually submits the form
    }, 2500);
}
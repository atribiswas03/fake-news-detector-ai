function showLogoutPopup() {
    document.getElementById("logoutModal").style.display = "flex";
}

function closeLogoutPopup() {
    document.getElementById("logoutModal").style.display = "none";
}

function confirmLogout() {
    fetch("/logout", { method: "POST" })
        .then(() => {
            window.location.href = "/";
        });
}

let deleteFormRef = null;

function showDeletePopup(event) {
    event.preventDefault();
    deleteFormRef = event.target;
    document.getElementById("deleteModal").style.display = "flex";
    return false;
}

function closeDeletePopup() {
    document.getElementById("deleteModal").style.display = "none";
    deleteFormRef = null;
}

function confirmDelete() {
    if (deleteFormRef) {
        deleteFormRef.submit();
    }
}
function handleAction(event, id, action) {
    event.preventDefault();
    fetch(`/${action}/${id}`, {
        method: "POST",
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                const statusCell = document.getElementById("status-" + id);
                const color = data.status === "Approved" ? "green" : "red";
                statusCell.innerHTML = `<b style="color: ${color};">${data.status}</b>`;
            } else {
                alert("❌ Action failed: " + data.error);
            }
        })
        .catch(err => {
            alert("❌ Network error");
            console.error(err);
        });
    return false;
}

function openAdminEmailModal() {
    const modal = document.getElementById("adminEmailModal");
    modal.classList.add("show");
    modal.style.display = "flex";
}
function closeAdminEmailModal() {
    const modal = document.getElementById("adminEmailModal");
    modal.classList.remove("show");
    setTimeout(() => modal.style.display = "none", 300);
}

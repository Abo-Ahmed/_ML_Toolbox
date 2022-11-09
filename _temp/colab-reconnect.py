// run every 1 minute
// paste this code in your window console
setInterval(function () {
	// close max ram window
	document.getElementById("cancel") && document.getElementById("cancel").innerText == "IGNORE" && document.getElementById("cancel").click();
	// reconnect
	document.getElementById("ok") && document.getElementById("ok").innerText == "RECONNECT" && document.getElementById("ok").click();
	!document.getElementsByTagName('iron-overlay-backdrop').length && document.getElementById("connect") && document.getElementById("connect").innerText == "RECONNECT" && document.getElementById("connect").click();
}, 60000);
document.addEventListener('DOMContentLoaded', function () {
    // const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);
    const socket = io(); // Should automatically connect to the host that serves the page

    const videoFeed = document.getElementById('videoFeed');
    const overallAlert = document.getElementById('overallAlert');
    const blinks = document.getElementById('blinks');
    const microsleepsDuration = document.getElementById('microsleepsDuration');
    const yawns = document.getElementById('yawns');
    const yawnDuration = document.getElementById('yawnDuration');
    const leftEyeState = document.getElementById('leftEyeState');
    const rightEyeState = document.getElementById('rightEyeState');
    const yawnState = document.getElementById('yawnState');
    const eventLog = document.getElementById('eventLog');

    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');

    let lastAlert = "";

    function addLog(message) {
        const p = document.createElement('p');
        p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        eventLog.appendChild(p);
        eventLog.scrollTop = eventLog.scrollHeight; // Scroll to bottom
    }

    socket.on('connect', function() {
        addLog('Connected to server via WebSocket.');
        // videoFeed.src = "/video_feed?" + new Date().getTime(); // Add cache buster for some browsers
        // No need to change src if it is already /video_feed and server handles stream restart
    });

    socket.on('disconnect', function() {
        addLog('Disconnected from server.');
        overallAlert.textContent = 'Disconnected';
        overallAlert.className = 'alert-danger';
    });

    socket.on('connection_ack', function(data) {
        addLog(`Server: ${data.message}`);
    });

    socket.on('status_update', function (data) {
        if (data.error) {
            addLog(`Error from server: ${data.error}`);
            overallAlert.textContent = data.error;
            overallAlert.className = 'alert-danger';
            return;
        }

        blinks.textContent = data.blinks;
        microsleepsDuration.textContent = data.microsleeps_duration.toFixed(2) + 's';
        yawns.textContent = data.yawns;
        yawnDuration.textContent = data.yawn_duration.toFixed(2) + 's';
        leftEyeState.textContent = data.left_eye_state;
        rightEyeState.textContent = data.right_eye_state;
        yawnState.textContent = data.yawn_state;

        const currentAlert = data.overall_alert || "Awake";
        overallAlert.textContent = currentAlert;

        if (currentAlert.includes("Prolonged Microsleep")) {
            overallAlert.className = 'alert-danger';
        } else if (currentAlert.includes("Prolonged Yawn")) {
            overallAlert.className = 'alert-warning';
        } else {
            overallAlert.className = 'alert-safe';
        }

        if (currentAlert !== "Awake" && currentAlert !== lastAlert) {
            addLog(`Alert: ${currentAlert}`);
            // Potentially play a sound here if browser allows
            // var audio = new Audio('/path/to/alert_sound.mp3');
            // audio.play().catch(e => console.log("Audio play failed: ", e));
        }
        lastAlert = currentAlert;
    });

    startButton.addEventListener('click', function() {
        fetch('/start_detection', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                addLog(`Start detection: ${data.message || data.error}`);
                // Reload video feed if it might have stopped or to ensure it starts
                // Adding a cache buster to the URL can help force a reload.
                videoFeed.src = "/video_feed?" + new Date().getTime();
            })
            .catch(error => {
                addLog(`Error starting detection: ${error}`);
            });
    });

    stopButton.addEventListener('click', function() {
        fetch('/stop_detection', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                addLog(`Stop detection: ${data.message || data.error}`);
                // Optionally, clear the video feed or show a static image
                // videoFeed.src = ""; // This might show a broken image icon
                // For now, let the server stop sending frames. The <img> tag will show last frame or browser default.
            })
            .catch(error => {
                addLog(`Error stopping detection: ${error}`);
            });
    });

    addLog('Script initialized. Attempting to connect to WebSocket...');
}); 
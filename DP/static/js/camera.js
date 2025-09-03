const video = document.getElementById("videoElement");
const canvas = document.getElementById("canvas");
const resultText = document.getElementById("resultText");
const videoContainer = document.getElementById("videoContainer");
const startButton = document.getElementById("startButton");

let streaming = false;
let sendInterval = null;

// Hide video at the start
videoContainer.style.display = 'none';

startButton.addEventListener("click", () => {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            videoContainer.style.display = 'block';
            startButton.style.display = 'none';

            streaming = true;

            // Start sending frames every 500ms
            sendInterval = setInterval(() => {
                captureAndSend();
            }, 500);

        })
        .catch((err) => {
            console.error("Camera error:", err);
            alert("Could not access camera. Please allow permissions.");
        });
});

function captureAndSend() {
    if (!streaming) return;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL("image/jpeg");

    fetch('/start_capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => res.json())
    .then(data => {
        if (data.result) {
            resultText.textContent = "Result: " + data.result;
        } else {
            resultText.textContent = "No palm detected";
        }
    })
    .catch(err => {
        console.error("Detection error:", err);
    });
}

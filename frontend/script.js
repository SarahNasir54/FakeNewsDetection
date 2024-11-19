function checkFakeNews() {
    const textInput = document.getElementById("newsInput").value.trim();
    const imageInput = document.getElementById("imageInput").files[0];
    const resultDiv = document.getElementById("result");

    if (!textInput && !imageInput) {
        resultDiv.innerHTML = "Please enter some text, a URL, or upload an image.";
        return;
    }

    const formData = new FormData();
    if (textInput) formData.append("text", textInput);
    if (imageInput) formData.append("file", imageInput);

    fetch('http://127.0.0.1:8000/predict', { 
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `Error: ${data.error}`;
        } else {
            const finalLabel = data.final_label || "Unknown";
            const finalScore = data.final_score !== undefined ? `Score: ${data.final_score.toFixed(2)}` : "";
            resultDiv.innerHTML = `Prediction: ${finalLabel} (${finalScore})`;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultDiv.innerHTML = "An error occurred. Please try again.";
    });
}

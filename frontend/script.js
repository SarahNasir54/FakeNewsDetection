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
    if (imageInput) formData.append("image", imageInput);

    fetch('http://127.0.0.1:8000/predict', { 
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `Error: ${data.error}`;
        } else {
            const label = data.label || "Unknown";
            const score = data.score !== undefined ? `, Score: ${data.score.toFixed(2)}` : "";
            resultDiv.innerHTML = `Label: ${label}${score}`;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultDiv.innerHTML = "An error occurred. Please try again.";
    });
}

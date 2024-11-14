function checkFakeNews() {
    const input = document.getElementById("newsInput").value;
    const resultDiv = document.getElementById("result");

    if (input.trim() === "") {
        resultDiv.innerHTML = "Please enter some text or a URL.";
        return;
    }

    // Make a POST request to the FastAPI backend
    fetch('http://127.0.0.1:8000/fake-news-check', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: input }),
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.innerHTML = data.message;
    })
    .catch(error => {
        console.error("Error:", error);
        resultDiv.innerHTML = "An error occurred. Please try again.";
    });
}

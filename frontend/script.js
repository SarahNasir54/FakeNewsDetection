function fetchProfileAndCheckNews() {
    const profileUrl = document.getElementById('profileUrl').value;
    const postText = document.getElementById('postText').value;
    const imageUpload = document.getElementById('imageUpload').files[0];
    const formData = new FormData();

    if (profileUrl) formData.append('profile_id', profileUrl);
    if (postText) formData.append('text', postText);
    if (imageUpload) formData.append('file', imageUpload);

    fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerHTML = `Error: ${data.error}`;
        } else {
            const profile = data.user_profile;
            let resultText = `<strong>User Profile:</strong><br>
                              Name: ${profile.name}<br>
                              Email: ${profile.email}<br>
                              Joined: ${profile.joined}<br><br>`;
            resultText += `<strong>Fake News Check:</strong><br>
                           Final Label: ${data.final_label}<br>
                           Final Score: ${data.final_score.toFixed(2)}`;
            document.getElementById('result').innerHTML = resultText;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById('result').innerHTML = "An error occurred. Please try again.";
    });
}

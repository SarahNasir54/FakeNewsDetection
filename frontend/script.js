async function fetchProfileAndCheckNews() {
    // Get inputs from the form
    const profileUrl = document.getElementById("profileUrl").value;
    const postText = document.getElementById("postText").value;
    const fileInput = document.getElementById("fileInput").files;

    let resultDiv = document.getElementById("result");
    resultDiv.innerHTML = ''; // Reset the result area

    // Check if user entered a profile URL or a post text
    if (profileUrl || postText || fileInput.length > 0) {
        
        // Handle Profile URL or ID
        if (profileUrl) {
            const profileResponse = await fetch(`http://127.0.0.1:8000/profiles/${profileUrl}`);
            if (profileResponse.ok) {
                const profileData = await profileResponse.json();
                resultDiv.innerHTML += `
                    <p><strong>Username:</strong> ${profileData.username}</p>
                    <p><strong>Name:</strong> ${profileData.name}</p>
                    <p><strong>Email:</strong> ${profileData.email}</p>
                    <p><strong>Age:</strong> ${profileData.age}</p>
                    <p><strong>Bio:</strong> ${profileData.bio}</p>
                    <p><strong>Followers:</strong> ${profileData.followers}</p>
                    <p><strong>Following:</strong> ${profileData.following}</p>
                `;
            } else {
                resultDiv.innerHTML += `<p>Profile not found</p>`;
            }
        }

        // Handle Post Text for Fake News Detection
        if (postText) {
            const fakeNewsResponse = await fetch("http://127.0.0.1:8000/predict", {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: postText })
            });
            const fakeNewsData = await fakeNewsResponse.json();
            resultDiv.innerHTML += `
                <p><strong>Fake News Detection Result:</strong></p>
                <p><strong>Label:</strong> ${fakeNewsData.final_label}</p>
                <p><strong>Score:</strong> ${fakeNewsData.final_score}</p>
            `;
        }

        // Handle File Uploads (if any)
        if (fileInput.length > 0) {
            const formData = new FormData();
            for (let i = 0; i < fileInput.length; i++) {
                formData.append("files", fileInput[i]);
            }

            // For simplicity, we're sending the files as a multipart/form-data to backend
            const fileUploadResponse = await fetch("http://127.0.0.1:8000/upload", {
                method: 'POST',
                body: formData
            });

            const fileUploadData = await fileUploadResponse.json();
            resultDiv.innerHTML += `<p><strong>Files uploaded successfully:</strong></p>`;
            fileUploadData.files.forEach(file => {
                resultDiv.innerHTML += `<p>File: ${file}</p>`;
            });
        }
    } else {
        resultDiv.innerHTML = `<p>Please enter a Profile ID, Post Text, or upload a file.</p>`;
    }
}

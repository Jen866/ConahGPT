async function sendMessage() {
    const inputElement = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const userText = inputElement.value.trim();

    // Stop if the input is empty
    if (!userText) {
        return;
    }

    // Display the user's question immediately
    chatBox.innerHTML += `<div><strong>You:</strong> ${userText}</div>`;
    inputElement.value = ""; // Clear the input box

    try {
        // Send the question to the Python backend
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // This correctly formats the question in JSON
            body: JSON.stringify({ question: userText })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        // Display the answer or an error from the backend
        let botResponse = "";
        if (data.answer) {
            // This replaces newline characters from the AI with <br> tags for HTML
            const formattedAnswer = data.answer.replace(/\n/g, '<br>');
            botResponse = `<div><strong>Conah GPT:</strong> ${formattedAnswer}</div>`;
        } else {
            botResponse = `<div><strong style="color: red;">Error:</strong> ${data.error || 'No answer returned.'}</div>`;
        }
        chatBox.innerHTML += botResponse;

    } catch (error) {
        // Handle network errors
        console.error("Fetch Error:", error);
        chatBox.innerHTML += `<div><strong style="color: red;">Error:</strong> Could not connect to the server.</div>`;
    }

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;
}
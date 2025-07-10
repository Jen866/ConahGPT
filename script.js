document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("question-form");
  const input = document.getElementById("question");
  const messages = document.getElementById("messages");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    appendMessage("You", question);
    input.value = "";

    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      if (data.answer) {
        appendMessage("Conah GPT", marked.parse(data.answer));
      } else {
        appendMessage("Conah GPT", "Sorry, no response received.");
      }
    } catch (error) {
      appendMessage("Conah GPT", "Something went wrong. Please try again.");
    }
  });

  function appendMessage(sender, text) {
    const message = document.createElement("div");
    message.innerHTML = `<strong>${sender}:</strong> ${text}`;
    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;
  }
});

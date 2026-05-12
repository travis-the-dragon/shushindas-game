// Tracks an image the user has pasted into the chat input
let pastedImageData = null;  // full base64 data URL
let pastedImageMimeType = null;

function showImagePreview(dataUrl) {
    const preview = document.getElementById('imagePreview');
    const img = document.getElementById('previewImg');
    img.src = dataUrl;
    preview.style.display = 'block';
}

function clearImagePreview() {
    pastedImageData = null;
    pastedImageMimeType = null;
    const preview = document.getElementById('imagePreview');
    const img = document.getElementById('previewImg');
    img.src = '';
    preview.style.display = 'none';
}

// Capture image paste events anywhere on the page
document.addEventListener('paste', function (event) {
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
            event.preventDefault();
            const blob = items[i].getAsFile();
            pastedImageMimeType = items[i].type;
            const reader = new FileReader();
            reader.onload = function (e) {
                pastedImageData = e.target.result;
                showImagePreview(e.target.result);
            };
            reader.readAsDataURL(blob);
            break;
        }
    }
});

function updateQuickQuestions(sample_questions) {
    // Update the input values
    document.getElementById('quickQ1Input').value = sample_questions.question1;
    document.getElementById('quickQ2Input').value = sample_questions.question2;
    document.getElementById('quickQ3Input').value = sample_questions.question3;
    document.getElementById('quickQ4Input').value = sample_questions.question4;

    // Update the button texts
    document.getElementById('quickQ1Button').querySelector('span').textContent = sample_questions.question1;
    document.getElementById('quickQ2Button').querySelector('span').textContent = sample_questions.question2;
    document.getElementById('quickQ3Button').querySelector('span').textContent = sample_questions.question3;
    document.getElementById('quickQ4Button').querySelector('span').textContent = sample_questions.question4;
}



function renderHistory(history) {
    const chatHistory = document.getElementById('chatHistory');
    chatHistory.innerHTML = '';  // Clear the current chat history

    history.forEach(function (item, index) {  // Use index to create unique IDs for thumbs
        const div = document.createElement('div');
        div.classList.add('d-flex', 'mb-2');

        if (item.is_tool_step) {
            let label;
            if (item.tool === 'list_books') {
                label = '📚 Consulting the card catalog...';
            } else if (item.tool === 'get_random_fact') {
                label = '🎲 Rummaging through the shelves for something interesting...';
            } else if (item.tool === 'check_overdue_books') {
                label = '📋 Checking the overdue ledger...';
            } else {
                label = `🔍 Searching the stacks for "${item.args.query}"...`;
            }
            div.classList.add('justify-content-center');
            div.innerHTML = `<div class="tool-step">${label}</div>`;
        } else if (item.is_her) {
            div.classList.add('justify-content-end');
            div.innerHTML = `
                <div class="msg-bubble msg-received position-relative" data-call-id="${item.call_id}">
                    ${marked.parse(item.text)}
                    <div class="feedback-icons position-absolute" style="bottom: 5px; right: 5px;">
                        <button class="btn btn-sm p-0" id="thumbsUpBtn-${index}">
                            👍
                        </button>
                        <button class="btn btn-sm p-0 ms-1" id="thumbsDownBtn-${index}">
                            👎
                        </button>
                    </div>
                </div>
            `;
        } else {
            div.classList.add('justify-content-start');
            const imageTag = item.has_image
                ? `<div style="font-size:0.8em; margin-top:4px; opacity:0.75;">&#128247; Image attached</div>`
                : '';
            div.innerHTML = `
                <div class="msg-bubble msg-sent">
                    ${item.text}${imageTag}
                </div>
            `;
        }

        chatHistory.appendChild(div);
    });

    // After rendering the history, re-attach the event listeners
    attachFeedbackListeners();
}

function attachFeedbackListeners() {
    const thumbsUpButtons = document.querySelectorAll('[id^="thumbsUpBtn"]');
    const thumbsDownButtons = document.querySelectorAll('[id^="thumbsDownBtn"]');

    thumbsUpButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            event.preventDefault();

            // Retrieve the message text and call_id
            const messageBubble = this.closest('.msg-bubble');
            const messageText = messageBubble.textContent.trim();
            const callId = messageBubble.getAttribute('data-call-id');

            // Call fetch to send feedback
            sendFeedback(messageText, '👍', callId);
        });
    });

    thumbsDownButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            event.preventDefault();

            // Retrieve the message text and call_id
            const messageBubble = this.closest('.msg-bubble');
            const messageText = messageBubble.textContent.trim();
            const callId = messageBubble.getAttribute('data-call-id');

            // Call fetch to send feedback
            sendFeedback(messageText, '👎', callId);
        });
    });
}



function getSelectedModel() {
    const dropdownButton = document.querySelector('#modelDropdown .dropdown-toggle');
    return dropdownButton.getAttribute('data-selected-value');
}


document.addEventListener('DOMContentLoaded', function () {
    const dropdownItems = document.querySelectorAll('#modelDropdown .dropdown-item');
    const dropdownButton = document.querySelector('#modelDropdown .dropdown-toggle');

    dropdownButton.setAttribute('data-selected-value', defaultModel);

    document.getElementById('clearImageBtn').addEventListener('click', function () {
        clearImagePreview();
    });

    dropdownItems.forEach(item => {
        item.addEventListener('click', function (event) {
            event.preventDefault(); // Prevent default link behavior

            // Get the value of the selected item
            const selectedValue = this.getAttribute('value');

            // Update the dropdown button text with the selected value
            dropdownButton.textContent = selectedValue;

            // Store the selected value for later use
            dropdownButton.setAttribute('data-selected-value', selectedValue);

            console.log('Selected Value:', selectedValue);
        });
    });


    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('buttonAsk');

    function setInputBusy(busy) {
        questionInput.disabled = busy;
        questionInput.style.opacity = busy ? '0.45' : '';
        askButton.disabled = busy;
    }

    function submitButton(buttonElement) {
        const spanElement = buttonElement.querySelector('span');
        let originalText = '';
        if (spanElement) {
            originalText = spanElement.textContent;
            spanElement.textContent = '';
        } else {
            originalText = buttonElement.textContent;
            buttonElement.textContent = '';
        }

        const spinner = document.createElement('div');
        spinner.classList.add('spinner');
        spinner.style.width = '24px';
        spinner.style.height = '24px';
        spinner.style.backgroundImage = 'url("/static/images/wizard-hat.png")';
        spinner.style.backgroundSize = 'cover';
        spinner.style.animation = 'spin 2s linear infinite';
        if (spanElement) {
            spanElement.appendChild(spinner);
        } else {
            buttonElement.appendChild(spinner);
        }

        let question = '';
        let qNum = 0;
        if (buttonElement.id === "buttonAsk") {
            question = questionInput.value;
            qNum = 0;
        } else if (buttonElement.id === "quickQ1Button") {
            question = document.getElementById('quickQ1Input').value;
            qNum = 1;
        } else if (buttonElement.id === "quickQ2Button") {
            question = document.getElementById('quickQ2Input').value;
            qNum = 2;
        } else if (buttonElement.id === "quickQ3Button") {
            question = document.getElementById('quickQ3Input').value;
            qNum = 3;
        } else if (buttonElement.id === "quickQ4Button") {
            question = document.getElementById('quickQ4Input').value;
            qNum = 4;
        }

        if (!question.trim()) {
            if (spanElement) spanElement.textContent = originalText;
            else buttonElement.textContent = originalText;
            return;
        }

        const model = getSelectedModel();
        const imageDataToSend = pastedImageData;
        if (imageDataToSend) clearImagePreview();

        // Clear and lock the text input immediately so it's obvious the submit landed
        if (buttonElement.id === 'buttonAsk') {
            questionInput.value = '';
        }
        setInputBusy(true);

        fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question, model: model, q_num: qNum, image_data: imageDataToSend })
        })
            .then(response => response.json())
            .then(data => {
                if (spanElement) spanElement.textContent = originalText;
                else buttonElement.textContent = originalText;
                setInputBusy(false);
                renderHistory(data.history);
                updateQuickQuestions(data.sample_questions);
            })
            .catch(error => {
                console.error('Error:', error);
                if (spanElement) spanElement.textContent = originalText;
                else buttonElement.textContent = originalText;
                setInputBusy(false);
            });
    }

    // Submit on Enter key in the text input
    questionInput.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            submitButton(askButton);
        }
    });

    // Handling clicks for all quick question buttons
    const buttons = document.querySelectorAll('button[id^="quickQ"], #buttonAsk');
    buttons.forEach(button => {
        button.addEventListener('click', function (event) {
            event.preventDefault();
            submitButton(this);
        });
    });
    

});

function sendFeedback(message, feedbackType, callId) {
    fetch('/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            feedback: feedbackType,
            call_id: callId
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Feedback response:', data);
        // Handle response if needed
    })
    .catch(error => {
        console.error('Error sending feedback:', error);
    });
}

// Handling the clear button click
document.getElementById('clearBtn').addEventListener('click', function (event) {
    event.preventDefault();  // Prevent the default form submission

    const data = {};

    fetch('/clear-history', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(data => {
            renderHistory(data.history);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
});


// CSS for spinner animation
const style = document.createElement('style');
style.innerHTML = `
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
`;
document.head.appendChild(style);

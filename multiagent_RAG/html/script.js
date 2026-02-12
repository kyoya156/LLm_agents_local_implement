const API_URL = 'http://localhost:8000';

async function askQuestion() {
    const query = document.getElementById('query').value.trim();
    
    if (!query) {
        showError('Please enter a question');
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');
    document.getElementById('submitBtn').disabled = true;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query, top_k: 3 })
        });

        if (!response.ok) {
            throw new Error('Failed to get response from server');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError('Error: ' + error.message + '. Make sure the API server is running.');
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('submitBtn').disabled = false;
    }
}

function displayResults(data) {
    // Show results section
    document.getElementById('results').classList.remove('hidden');

    // Display answer
    document.getElementById('answer').textContent = data.final_answer;


    
    data.retrieved_docs.forEach((doc, index) => {
        const docDiv = document.createElement('div');
        docDiv.className = 'document';
        docDiv.innerHTML = `<strong>Document ${index + 1}:</strong><br>${doc.document}`;
    });
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

// Allow Enter to submit (Shift+Enter for new line)
document.getElementById('query').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});
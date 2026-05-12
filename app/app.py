import base64
import datetime
import io
import random
from flask import Flask, jsonify, render_template, request, session
import os
from PIL import Image as PILImage
from utils import *

app = Flask(__name__)
app.secret_key = 'So long and thanks for all the fish!'

WEAVE_PROJECT = "shushindas-game"
weave_client = weave.init(WEAVE_PROJECT)

VECTOR_DB = "chroma"  # Switch to "bigquery" to use BigQuery instead
LLM = "openai"  # Set language model to OpenAI

history = []

@app.route("/feedback", methods=["POST"])
def feedback():
    """Endpoint to receive feedback for a specific call.

    This function handles feedback submitted by users for a specific call ID. 
    It adds the feedback as a reaction, a note, and stores user info associated 
    with the feedback in Weave.

    Returns:
        dict: An empty JSON response.
    """
    data = request.json
    call_id = data.get("call_id")
    feedback = data.get("feedback")
    print(f"{call_id}: {feedback}")

    call = weave_client.get_call(call_id)
    call.feedback.add_reaction(feedback)
    call.feedback.add_note("this is a note")
    call.feedback.add("UserInfo", {"name": "clyde", "user_id": "42231"})

    return jsonify({})

@app.route("/clear-history", methods=['POST'])
def clear_hist():
    """Endpoint to clear the chat history stored in session.

    Resets the chat history session variable to an empty list.

    Returns:
        dict: JSON response containing the updated history.
    """
    print('clear!')
    history = []
    session['history'] = history
    return jsonify(history=session.get('history', []))

@app.route("/ask", methods=['POST'])
def ask():
    """Endpoint to process a user's question and return a response.

    Takes a question from the user, generates a context using the 
    EmbeddingsDB, and gets a response from the specified language model.

    Returns:
        dict: JSON response containing the answer, updated history, and sample questions.
    """
    data = request.json
    print(f"data: {data}")
    question = data.get("question")
    print(f"question: {question}")
    model = data.get("model")
    sample_question_num = data.get("q_num")
    image_data = data.get("image_data")  # base64 data URL: "data:image/png;base64,..."

    pil_image = None
    if image_data:
        try:
            _, b64 = image_data.split(',', 1)
            pil_image = PILImage.open(io.BytesIO(base64.b64decode(b64)))
        except Exception as e:
            print(f"Error decoding pasted image: {e}")

    agent = ShushindaAgent(llm_model_name=model)
    history = session.get('history', [])

    if question is not None:
        with weave.attributes({'sample_question_num': sample_question_num, "env": "prod", "has_image": pil_image is not None}):
            response = agent.predict(question=question, image=pil_image)
            answer = response["response"]
            call_id = response["call_id"]
            tool_steps = response.get("tool_steps", [])
        if answer is not None:
            history.extend(add_to_history(question, answer, call_id, tool_steps=tool_steps, has_image=pil_image is not None))
            answer = random.choice(GREETINGS)
    session['history'] = history
    sample_questions = get_sample_questions()

    return jsonify(answer=answer, history=history, sample_questions=sample_questions)

@app.route("/", methods=['POST', 'GET'])
def main():
    """The home page route for the application.

    Renders the home page and manages session history for user interactions.

    Returns:
        Response: Rendered HTML template for the home page.
    """
    answer = None
    question = None
    response = None
    history = []

    print(f"session history: {str(session.get('history'))[:40]}...")
    if session.get('history') is None:
        history = []
        session['history'] = history
    else:
        history = session.get('history')

    # Handle GET request: user has not yet submitted the form
    if request.method == 'GET':
        question = None
        answer = random.choice(GREETINGS)

    session['history'] = history

    # Prepare model data for rendering
    llms = [l['model'] for l in LLMS]
    model = {"message": answer, "input": question, "history": history, "llms": llms}

    qs = get_sample_questions()
    model = model | qs

    return render_template('index.html', model=model)

def get_sample_questions():
    """Generates a set of sample questions for display.

    Randomly selects a subset of questions, sometimes replacing one with a specific 
    unlock question, and shuffles them.

    Returns:
        dict: A dictionary of sample questions.
    """
    some_questions = random.sample(ALL_SAMPLE_QUESTIONS, 4)
    some_questions[0] = UNLOCK_QUESTION
    random.shuffle(some_questions)

    model = {}
    model["question1"] = some_questions.pop()
    model["question2"] = some_questions.pop()
    model["question3"] = some_questions.pop()
    model["question4"] = some_questions.pop()

    return model

def get_history():
    """Fetches a mock history of chat interactions for demonstration.

    Returns:
        list: A list of chat history entries.
    """
    chat_history = []
    chat_history.append({
        "is_her": False,
        "is_me": True,
        "text": "Got any spare silencing charms?",
        "timestamp": datetime.datetime.now()
    })
    chat_history.append({
        "is_her": True,
        "is_me": False,
        "text": "Oh, dear, I may have accidentally used the last one to quiet a particularly squawking raven... but perhaps a strategically placed cushion might help?",
        "timestamp": datetime.datetime.now()
    })
    return chat_history

def add_to_history(question, response, call_id, tool_steps=None, has_image=False):
    """Adds a question, tool steps, and response to the chat history.

    Args:
        question (str): The user's question.
        response (str): The model's response.
        call_id (str): The call ID for tracking.
        tool_steps (list): Tool calls made by the agent before answering.
        has_image (bool): Whether the question included a pasted image.

    Returns:
        list: A list of new history items to be appended.
    """
    items = []
    items.append({
        "is_her": False,
        "is_me": True,
        "text": question,
        "has_image": has_image,
        "timestamp": datetime.datetime.now()
    })
    for step in (tool_steps or []):
        items.append({
            "is_tool_step": True,
            "tool": step["tool"],
            "args": step["args"],
        })
    items.append({
        "is_her": True,
        "is_me": False,
        "text": response,
        "timestamp": datetime.datetime.now(),
        "call_id": call_id
    })
    return items

if __name__ == '__main__':
    # Environment variable checks
    if os.getenv("PROJECT") is None:
        print("PROJECT not set!")
    if os.getenv("REGION") is None:
        print("REGION not set!")
    if os.getenv("OPENAI_API_KEY") is None:
        print("OPENAI_API_KEY not set!")

    app.run( host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
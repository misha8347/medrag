from flask import Flask, render_template
from flask_socketio import SocketIO
from process.articles_extraction import SummarizedArticlesExtractor
from process.llama_response import ollama_response_recommendations_with_context, \
    ollama_response_recommendations_without_context

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# ML processors
summarized_articles_extractor = SummarizedArticlesExtractor()

@app.route('/')
def index():
    return render_template('chat.html')

@socketio.on('message')
def handle_message(message):
    print(f"User: {message}")
    context = summarized_articles_extractor.process(message, task='qa')
    response = ollama_response_recommendations_without_context(message)
    # response = ollama_response_recommendations_with_context(message, context)
    socketio.send(response)

if __name__ == '__main__':
    socketio.run(app, debug=True)

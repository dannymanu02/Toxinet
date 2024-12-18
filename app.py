from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

label_encoder = joblib.load('label_encoder.pkl')

best_model_name = open('best_model.txt', 'r').read().strip()

if best_model_name == 'LSTM':
    tokenizer = joblib.load('tokenizer_lstm.pkl')
    model = tf.keras.models.load_model('lstm_model.h5')
    model_type = 'LSTM'

elif best_model_name == 'LSTM+CNN':
    tokenizer = joblib.load('tokenizer_lstm.pkl')
    model = tf.keras.models.load_model('lstm_cnn_model.h5')
    model_type = 'LSTM+CNN'

else:
    raise ValueError("Best model name not recognized.")

label_mapping = {
    0: "Neutral or Ambiguous",
    1: "Offensive or Hate Speech",
    2: "Not Hate"
}

def predict(text):
    if model_type in ['LSTM', 'LSTM+CNN']:
        sequence = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

        pred_probs = model.predict(padded)
        pred_class = np.argmax(pred_probs, axis=1)[0]

    else:
        raise ValueError("Invalid model type.")

    label = label_mapping[pred_class]
    confidence = float(np.max(pred_probs))

    return label, confidence

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json(force=True)
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    label, confidence = predict(text)

    return jsonify({
        'text': text,
        'prediction': label,
        'confidence': confidence
    })

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

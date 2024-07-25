from flask import Flask, send_file
import os

app = Flask(__name__)

# Update this path to your image directory
IMAGE_DIR = r'E:\AIMT\Chatbot\images'

@app.route('/images/<filename>')
def serve_image(filename):
    image_path = os.path.join(IMAGE_DIR, filename)
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5000)

import os
from datetime import datetime
from dog_breed_detector import DogBreedDetector
from flask import Flask, jsonify
from flask import request, redirect, url_for, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_IMG_FOLDER = 'static/uploaded_imgs/'

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def uploadImage():
	'''
	Index webpage displays receives user input image to detect dog breed
	'''
	return render_template('index.html')


@app.route('/detectDogBreed', methods=['POST'])
def detectDogBreed():
	'''
	Receive an image and displays model results which dog breed is in this image.
	'''

	# Check if post request contains file
	if 'image' not in request.files:
		message = 'No file part'
		return jsonify({'message' : message})

	# Get file from request
	file = request.files['image']
	
	# Check if file is valid.
	if file.filename == '':
		message = 'No image selected for uploading'
		return jsonify({'message' : message})

	if file and allowed_file(file.filename) == False:
		message = 'This image format is not supported. Just support png, jpg, jpeg file.'
		return jsonify({'message' : message})

	# Save image into storage
	filename = secure_filename(file.filename)
	filename, file_extension = os.path.splitext(filename)
	filename = filename + '_' + str(int(datetime.timestamp(datetime.now()))) + '.' +  file_extension
	file_path = os.path.join(UPLOAD_IMG_FOLDER, filename)
	file_path = os.path.abspath(file_path)
	file.save(file_path)

	# Detect dog breed in the image
	detected_result = DogBreedDetector().detect_dog_breed(file_path)
	
	return jsonify({'detected_result': detected_result})

		

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == "__main__":
    app.run()
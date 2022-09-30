from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
 
app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}
verbose_name = {
	0: "Covid19",
	1: "Normal",
	2: "Pneumonia",
	3: "Turberculosis",
 
 
}



# routes

model = load_model('lung.h5')

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(180,180))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 180,180,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name[classes_x[0]]

 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/upload", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['imagefile']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/performance")
def performance():
	return render_template('performance.html')
    
@app.route("/chart")
def chart():
	return render_template('chart.html') 

	
if __name__ =='__main__':
	app.run(debug = True)

from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
import os
import ig

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        result = ig.pre()
        return redirect('/total')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('/total'))

@app.route('/total')
def total():
    result = ig.pre()
    return render_template('total.html',result=result)
    

if __name__ == '__main__':
    app.run(debug=True)


    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
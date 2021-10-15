# Pytorch-with-Flask
A space to help you create a webserver for deployment of a deep learning model. 
##Link to Download the pretrained model: https://drive.google.com/file/d/1vAKr5iASXgEdFogrSsKiczvhf1gknB07/view?usp=sharing
Create the flask environment with: `set FLASK_ENV=app.py`.

Run the server with : `flask run`

Predict with: `curl -X POST -F file=@cat_pic.jpeg http://localhost:5000/predict`    

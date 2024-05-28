from flask import Flask
from API.predict import predict_blueprint

app = Flask(__name__)

# Register blueprints
app.register_blueprint(predict_blueprint)

if __name__ == '__main__':
    app.run(debug=True, port=5003)

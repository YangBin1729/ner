import os

from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import InputRequired

from models import predict

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')


class InputForm(FlaskForm):
    input_text = TextAreaField("请输入文本", validators=[InputRequired(message="请输入有效的文本")])
    submit = SubmitField("提交")


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    form = InputForm()
    res = {}
    if form.validate_on_submit():
        input_text = form.input_text.data
        res = predict(input_text)
    return render_template('index.html', form=form, res=res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=True)


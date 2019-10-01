from flask import Flask, render_template, request
from bert import Ner

model = Ner("out_base/")
app = Flask(__name__)
app.jinja_env.filters['zip'] = zip

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process', methods=["POST"])
def process():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        output = model.predict(rawtext)
        entities = []
        tags = []
        scores = []
        print(output)
        for item in output:
            if item['tag'] != 'O':
                entities.append(item['word'])
                tags.append(item['tag'])
                scores.append(item['confidence'])
        return render_template("index.html", entities=entities, tags=tags, scores=scores, num_of_results=len(entities), text=rawtext)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

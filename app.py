from flask import Flask, render_template, request
from bert import Ner

model = Ner("out/")
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
        for word, tag in output.items():
            if tag['tag'] != 'O':
                entities.append(word)
                tags.append(tag['tag'])
                scores.append(tag['confidence'])
        return render_template("index.html", entities=entities, tags=tags, scores=scores, num_of_results=len(entities), text=rawtext)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

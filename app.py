import json
from flask import Flask, render_template, request, Response, stream_with_context
from agent import run_agent_stream

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/research", methods=["POST"])
def research():
    company = request.json.get("company", "").strip()
    if not company:
        return {"error": "Company name is required"}, 400

    def generate():
        for event in run_agent_stream(company):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

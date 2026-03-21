from flask import Flask
import click
from .explorer_ui import explorer_blueprint

app = Flask(__name__)


app.register_blueprint(explorer_blueprint)


@app.route("/health")
def health():
    return "OK", 200


@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Address to bind the server to.")
@click.option("--port", default=5000, type=int, show_default=True, help="Port to run the server on.")
@click.option("--debug/--no-debug", default=False, show_default=True, help="Enable Flask debug mode.")
def main(host, port, debug):
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()

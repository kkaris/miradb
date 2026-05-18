import click
from flask import Flask, redirect
from .explorer_ui import explorer_blueprint

app = Flask(__name__)


app.register_blueprint(explorer_blueprint)


@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/health")
def health():
    return "OK", 200


@app.route("/")
def index():
    # Redirect to /explorer
    redirect("/explorer", code=302)


@click.command()
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Address to bind the server to.",
)
@click.option(
    "--port", default=5000, show_default=True, help="Port to run the server on."
)
@click.option(
    "--debug/--no-debug",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable or disable Flask debug mode.",
)
def main(host, port, debug):
    """Run the Flask application."""
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()

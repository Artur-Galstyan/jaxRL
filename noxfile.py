import nox


@nox.session
def tests(session):
    # Install dependencies using poetry
    session.install("poetry")
    session.install("pytest")
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest")


@nox.session
def lint(session):
    # Use black formatter
    session.install("black")
    session.run("black", "--check", ".")

    # Use Ruff linter
    session.install("ruff")
    session.run("ruff", ".")

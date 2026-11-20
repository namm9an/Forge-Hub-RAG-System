"""Keep configuration/library imports free of application startup side effects."""


def create_app():
    from .main import create_app as factory
    return factory()


def __getattr__(name):
    if name == "app":
        application = create_app()
        globals()[name] = application
        return application
    raise AttributeError(name)


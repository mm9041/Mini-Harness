"""Expected conflicts between a browser action and current conversation state."""


class UiConflictError(RuntimeError):
    """The action is valid, but must wait for or change the current UI state."""

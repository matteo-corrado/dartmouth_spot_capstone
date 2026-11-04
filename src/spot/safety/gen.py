"""Safety generation counter.

Moved out of spot_dispatch's module globals ``_safety_gen`` and
``_dispatch_gen``. Plain attribute reads and writes, exactly like the old
globals: no lock, so the stop path stays lock-free.
"""


class SafetyGen:
    """``_gen`` is bumped by every aborting intent (any thread).
    ``_dispatch_gen`` is ``_gen`` when the current locked dispatch began
    (None outside one)."""

    def __init__(self):
        self._gen = 0
        self._dispatch_gen = None

    def current(self) -> int:
        """Snapshot for aborted(gen): any stop after it wins."""
        return self._gen

    def bump(self) -> int:
        self._gen += 1
        return self._gen

    def aborted(self, gen) -> bool:
        """True if an aborting intent arrived since generation `gen` was taken."""
        return gen is not None and gen != self._gen

    def dispatch_current(self):
        return self._dispatch_gen

    def dispatch_bump(self):
        """Enter a locked dispatch: the outermost one takes the current
        generation. Returns the outer value for dispatch_restore()."""
        outer = self._dispatch_gen
        if outer is None:
            self._dispatch_gen = self._gen
        return outer

    def dispatch_restore(self, outer):
        self._dispatch_gen = outer

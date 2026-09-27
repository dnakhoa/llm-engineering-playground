"""The reference Flagship Agent: Acme Notes' support agent, built from scratch.

``flagship.loop:run`` is its Spine 1 (Loop) state, ``flagship.knowledge:run``
its Spine 2 (Knowledge) and Spine 3 (Graded) state, and ``flagship.observed:run``
its Spine 4 (Observed) state. Each Spine module changes it, and each lesson
shows that change. It is deliberately small enough
to read in one sitting, and uses no agent framework (ADR 0003).
"""

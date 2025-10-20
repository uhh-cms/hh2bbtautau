# coding: utf-8

"""
Generic interface for loading and evaluating TensorFlow models in a separate process.
Data exchange is handled through multiprocessing pipes.
"""

from __future__ import annotations

import os
import time
import pathlib
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from dataclasses import dataclass
from typing import Any


STOP_SIGNAL = "STOP"


class TFEvaluator:
    """
    TensorFlow model evaluator that runs in a separate process with support for multiple models.

    .. code-block:: python

        evaluator = TFEvaluator()
        evaluator.add_model("model_name", "path/to/model")
        with evaluator:
            result = evaluator("model_name", input_data)
    """

    @dataclass
    class Model:
        name: str
        path: str
        signature_key: str = ""

    def __init__(self) -> None:
        super().__init__()

        self._models: dict[str, TFEvaluator.Model] = {}
        self._p: Process | None = None
        self._pipe: Connection | None = None

        self.delay = 0.2
        self.silent = False

    def __enter__(self) -> TFEvaluator:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        self.stop()

    def __call__(self, *args, **kwargs) -> Any:
        return self.evaluate(*args, **kwargs)

    @property
    def running(self) -> bool:
        return self._p is not None

    def add_model(self, name: str, path: str | pathlib.Path, signature_key: str = "") -> None:
        if self.running:
            raise ValueError("cannot add models while running")
        if name in self._models:
            raise ValueError(f"model with name '{name}' already exists")

        # normalize path
        path = str(path)
        path = os.path.expandvars(os.path.expanduser(path))
        path = os.path.abspath(os.path.abspath(path))

        # add it
        self._models[name] = TFEvaluator.Model(name=name, path=path, signature_key=signature_key)

    def start(self) -> None:
        if self.running:
            raise ValueError("process already started")

        # build the subprocess config
        config = []
        for model in self._models.values():
            parent_pipe, child_pipe = Pipe()
            config.append({"name": model.name, "path": model.path})

        # setup the pipes
        self._pipe, child_pipe = Pipe()

        # create and start the process
        self._p = Process(
            target=_tf_evaluate,
            args=(config, child_pipe),
            kwargs={"delay": self.delay, "silent": self.silent},
        )
        self._p.start()

    def evaluate(self, name: str, *args, **kwargs) -> Any:
        if not self.running:
            raise ValueError("process not started")

        # get the model
        if name not in self._models:
            raise ValueError(f"model with name '{name}' does not exist")

        # evaluate
        self._pipe.send((name, args, kwargs))  # type: ignore[union-attr]

        # wait for and receive result
        res_name, res = self._pipe.recv()  # type: ignore[union-attr]
        if res_name != name:
            raise RuntimeError(f"received result for unexpected model '{res_name}' (expected '{name}')")

        return res

    def stop(self, timeout: int | float = 5) -> None:
        # stop and remove pipe
        if self._pipe is not None:
            self._pipe.send(STOP_SIGNAL)
            self._pipe.close()
            self._pipe = None

        # nothing to do when not running
        if not self.running:
            return

        # join to wait for normal termination
        if self._p.is_alive():
            self._p.join(timeout)

            # kill if still alive
            if self._p.is_alive():
                self._p.kill()

        # reset
        self._p = None


def _tf_evaluate(
    config: list[dict[str, Any]],
    pipe: Connection,
    /,
    *,
    delay: int | float = 0.2,
    silent: bool = False,
) -> None:
    _print = (lambda *args, **kwargs: None) if silent else print

    _print("importing tensorflow ...")
    import numpy as np
    import tensorflow as tf  # type: ignore[import-not-found,import-untyped]
    _print("done")

    @dataclass
    class Model:
        name: str
        path: str
        signature_key: str = ""
        model: Any = None

        @classmethod
        def new(cls, config: dict[str, Any], /) -> Model:
            for attr in ["name", "path"]:
                if attr not in config:
                    raise ValueError(f"missing field '{attr}' in model config")
            if not os.path.exists(config["path"]):
                raise FileNotFoundError(f"model file '{config['path']}' does not exist")
            return cls(
                name=config["name"],
                path=config["path"],
                signature_key=config.get("signature_key", ""),
            )

        def load(self) -> None:
            sig_msg = f" (signature '{self.signature_key}')" if self.signature_key else ""
            _print(f"loading model '{self.name}'{sig_msg} from {self.path} ...")

            model = tf.saved_model.load(self.path)
            self.model = model if not self.signature_key else model.signatures[self.signature_key]

            _print("done")

        def evaluate(self, *args, **kwargs) -> np.ndarray:
            return self.model(*args, **kwargs).numpy()

        def clear(self) -> None:
            _print(f"clearing model '{self.name}'")
            self.model = None

    # convert to model objects and load
    models = {}
    for item in config:
        model = Model.new(item)
        models[model.name] = model
        model.load()

    # helper for gracefully shutting down
    def shutdown() -> None:
        for model in models.values():
            model.clear()
        models.clear()
        pipe.close()

    # start loop listening for data
    while models:
        # sleep if there is not data to process
        if not pipe.poll():
            time.sleep(delay)
            continue

        # receive data and select model
        data = pipe.recv()
        if isinstance(data, tuple) and len(data) == 3:
            # normal evaluation
            try:
                name, args, kwargs = data
                result = models[name].evaluate(*args, **kwargs)
                # send back result
                pipe.send((name, result))
            except:
                shutdown()
                raise
        elif isinstance(data, tuple) and len(data) == 2 and data[1] == STOP_SIGNAL:
            # stop a specific model
            try:
                name = data[0]
                models[name].clear()
                models.pop(name)
            except:
                shutdown()
                raise
        elif data == STOP_SIGNAL:
            # stop all models
            shutdown()
        else:
            raise ValueError(f"received unexpected data type through pipe: {type(data)}")

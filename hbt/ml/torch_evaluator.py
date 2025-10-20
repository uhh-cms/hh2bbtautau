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
from typing import Any, Callable


STOP_SIGNAL = "STOP"


class PyTEvaluator:
    """
    Pytorch model evaluator that runs in a separate process with support for multiple models.

    .. code-block:: python

        evaluator = PyTEvaluator()
        evaluator.add_model("model_name", "path/to/model")
        with evaluator:
            result = evaluator("model_name", input_data)
    """

    @dataclass
    class Model:
        name: str
        path: str
        pipe: Connection | None = None
        build_fn: None | Callable = None
        build_cfg: None | dict = None

    def __init__(self) -> None:
        super().__init__()

        self._models: dict[str, PyTEvaluator.Model] = {}
        self._p: Process | None = None

        self.delay = 0.2
        self.silent = False

    def __enter__(self) -> PyTEvaluator:
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

    def add_model(self, name: str, path: str | pathlib.Path, build_fn = None, build_cfg = None) -> None:
        if self.running:
            raise ValueError("cannot add models while running")
        if name in self._models:
            raise ValueError(f"model with name '{name}' already exists")

        # normalize path
        path = str(path)
        path = os.path.expandvars(os.path.expanduser(path))
        path = os.path.abspath(os.path.abspath(path))

        # add it
        self._models[name] = PyTEvaluator.Model(name=name, path=path, build_fn=build_fn, build_cfg=build_cfg)

    def start(self) -> None:
        if self.running:
            raise ValueError("process already started")

        # build the subprocess config
        config = []
        for model in self._models.values():
            parent_pipe, child_pipe = Pipe()
            model.pipe = parent_pipe
            config.append({"name": model.name, "path": model.path, "pipe": child_pipe, "build_fn": model.build_fn, "build_cfg": model.build_cfg})

        # create and start the process
        self._p = Process(
            target=_pyt_evaluate,
            args=(config,),
            kwargs={"delay": self.delay, "silent": self.silent},
        )
        self._p.start()

    def evaluate(self, name: str, *args, **kwargs) -> Any:
        if not self.running:
            raise ValueError("process not started")

        # get the model
        if name not in self._models:
            raise ValueError(f"model with name '{name}' does not exist")
        model = self._models[name]

        # evaluate and send back result
        from IPython import embed; embed(header="string - 110 in torch_evaluator.py ")
        model.pipe.send((args, kwargs))  # type: ignore[union-attr]
        return model.pipe.recv()  # type: ignore[union-attr]

    def stop(self, timeout: int | float = 5) -> None:
        # stop and remove model pipes
        for model in self._models.values():
            if model.pipe is not None:
                model.pipe.send(STOP_SIGNAL)
                model.pipe.close()
                model.pipe = None

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


def _pyt_evaluate(
    config: list[dict[str, Any]],
    /,
    *,
    delay: int | float = 0.2,
    silent: bool = False,
) -> None:
    _print = (lambda *args, **kwargs: None) if silent else print

    _print("importing pytorch ...")
    import numpy as np
    import torch as t  # type: ignore[import-not-found,import-untyped]
    _print("done")

    @dataclass
    class Model:
        name: str
        path: str
        pipe: Connection
        model: Any = None
        build_fn: None | Callable = None
        build_cfg: None | dict = None

        @classmethod
        def new(cls, config: dict[str, Any], /) -> Model:
            for attr in ("name", "path", "pipe"):
                if attr not in config:
                    raise ValueError(f"missing field '{attr}' in model config")
            if not os.path.exists(config["path"]):
                raise FileNotFoundError(f"model file '{config['path']}' does not exist")
            if not isinstance(config["pipe"], Connection):
                raise TypeError(f"'pipe' {config['pipe']} not of type '{Connection}'")
            return cls(
                name=config["name"],
                path=config["path"],
                pipe=config["pipe"],
                build_fn=config["build_fn"],
                build_cfg=config["build_cfg"],
            )

        def load(self) -> None:
            # build a model with a given function or load it directly from pt file
            if self.build_fn:
                _print(f"Create model structure'{self.name}' load state dict from {self.path} ...")
                model = self.build_fn(**self.build_fn_config)
                model.load_state_dict(self.path)
                self.model = model
            else:
                _print(f"loading model '{self.name}' from {self.path} ...")
                model = t.export.load(self.path).module()
            self.model = model
            _print("done")

        def evaluate(self, *args, **kwargs) -> np.ndarray:
            from IPython import embed; embed(header="string - 191 in torch_evaluator.py ")
            return self.model(*args, **kwargs).numpy()

        def clear(self) -> None:
            _print(f"clearing model '{self.name}'")
            self.model = None
            self.pipe.close()

    # convert to model objects
    models = [Model.new(item) for item in config]

    # load model objects
    for model in models:
        model.load()

    # helper for gracefully shutting down
    def shutdown() -> None:
        for model in models:
            model.clear()
        models.clear()

    # start loop listening for data
    while models:
        remove_models: list[int] = []
        for i, model in enumerate(models):
            # skip if there is no data to process
            if not model.pipe.poll():
                continue

            # get data and process
            data = model.pipe.recv()
            if isinstance(data, tuple) and len(data) == 2:
                # evaluate
                try:
                    args, kwargs = data
                    result = model.evaluate(*args, **kwargs)
                except:
                    shutdown()
                    raise
                # send back result
                model.pipe.send(result)

            elif data == STOP_SIGNAL:
                # remove model
                model.clear()
                remove_models.append(i)

            else:
                raise ValueError(f"unexpected data type {type(data)}")

        # reduce models and sleep
        models = [model for i, model in enumerate(models) if i not in remove_models]
        time.sleep(delay)

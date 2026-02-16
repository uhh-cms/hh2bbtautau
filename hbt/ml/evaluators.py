# coding: utf-8

"""
Generic interface for loading and evaluating TensorFlow and Torch models in a subprocess.
Data exchange is handled through multiprocessing pipes.
"""

from __future__ import annotations

import os
import abc
import time
import pathlib
import functools
import dataclasses
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Any, Type


STOP_SIGNAL = "STOP"


@dataclasses.dataclass
class BaseModel:
    name: str
    path: str
    model: Any = None

    @classmethod
    def new(cls, config: dict[str, Any], /) -> BaseModel:
        for attr in ["name", "path"]:
            if attr not in config:
                raise ValueError(f"missing field '{attr}' in model config")
        if not os.path.exists(config["path"]):
            raise FileNotFoundError(f"model file '{config['path']}' does not exist")
        return cls(**config)

    def clear(self) -> None:
        print(f"clearing {self.__class__.__name__} '{self.name}'")
        self.model = None

    def load(self) -> None:
        raise NotImplementedError

    def evaluate(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class BaseEvaluator(abc.ABC):

    @dataclasses.dataclass
    class Model:
        name: str
        path: str

    def __init__(self) -> None:
        super().__init__()

        self._models: dict[str, BaseEvaluator.Model] = {}
        self._p: Process | None = None
        self._pipe: Connection | None = None

        self.delay = 0.1

    @abc.abstractmethod
    def get_model_cls(self) -> Type[BaseModel]:
        ...

    def __enter__(self) -> BaseEvaluator:
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

    def add_model(self, name: str, path: str | pathlib.Path, **model_kwargs) -> None:
        if self.running:
            raise ValueError("cannot add models while running")
        if name in self._models:
            raise ValueError(f"model with name '{name}' already exists")

        # normalize path
        path = str(path)
        path = os.path.expandvars(os.path.expanduser(path))
        path = os.path.abspath(os.path.abspath(path))

        # add it
        self._models[name] = self.Model(name=name, path=path, **model_kwargs)

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
            target=evaluation_loop,
            args=(self.get_model_cls(), config, child_pipe),
            kwargs={"delay": self.delay},
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


def evaluation_loop(
    model_cls: Type[BaseModel],
    config: list[dict[str, Any]],
    pipe: Connection,
    /,
    *,
    delay: int | float = 0.2,
) -> None:
    # convert to model objects and load
    models = {}
    for item in config:
        model = model_cls.new(item)
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


#
# TensorFlow model and evaluator
#

@dataclasses.dataclass
class TFModel(BaseModel):
    signature_key: str = ""

    @classmethod
    @functools.cache
    def imports(cls):
        print("importing tensorflow ...")
        import tensorflow as tf  # type: ignore[import-not-found,import-untyped]
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        print("done")
        return tf

    @classmethod
    def cast_to_numpy(cls, obj: Any) -> Any:
        tf = cls.imports()

        if isinstance(obj, (list, tuple)):
            return type(obj)(cls.cast_to_numpy(o) for o in obj)
        if isinstance(obj, dict):
            return type(obj)((k, cls.cast_to_numpy(v)) for k, v in obj.items())
        if isinstance(obj, tf.Tensor):
            return obj.numpy()

        return obj

    def load(self) -> None:
        tf = self.imports()

        sig_msg = f" (signature '{self.signature_key}')" if self.signature_key else ""
        print(f"loading {self.__class__.__name__} '{self.name}'{sig_msg} from {self.path} ...")

        model = tf.saved_model.load(self.path)
        self.model = model if not self.signature_key else model.signatures[self.signature_key]

        print("done")

    def evaluate(self, *args, **kwargs) -> Any:
        out = self.model(*args, **kwargs)
        return self.cast_to_numpy(out)


class TFEvaluator(BaseEvaluator):
    """
    TensorFlow model evaluator that runs in a separate process with support for multiple models.

    .. code-block:: python

        evaluator = TFEvaluator()
        evaluator.add_model("model_name", "path/to/model", signature_key="serving_default")
        with evaluator:
            result = evaluator("model_name", input_data)
    """

    @dataclasses.dataclass
    class Model(BaseEvaluator.Model):
        signature_key: str = ""

    def get_model_cls(self) -> Type[BaseModel]:
        return TFModel


#
# Torch model and evaluator
#

@dataclasses.dataclass
class TorchModel(BaseModel):

    @classmethod
    @functools.cache
    def imports(cls):
        print("importing torch ...")
        import numpy as np
        import torch  # type: ignore[import-not-found,import-untyped]
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print("done")
        return torch, np

    @classmethod
    def cast_from_numpy(cls, obj: Any) -> Any:
        torch, np = cls.imports()

        if isinstance(obj, (list, tuple)):
            return type(obj)(cls.cast_from_numpy(o) for o in obj)
        if isinstance(obj, dict):
            return type(obj)((k, cls.cast_from_numpy(v)) for k, v in obj.items())
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)

        return obj

    @classmethod
    def cast_to_numpy(cls, obj: Any) -> Any:
        torch, _ = cls.imports()

        if isinstance(obj, (list, tuple)):
            return type(obj)(cls.cast_to_numpy(o) for o in obj)
        if isinstance(obj, dict):
            return type(obj)((k, cls.cast_to_numpy(v)) for k, v in obj.items())
        if isinstance(obj, torch.Tensor):
            return obj.numpy()

        return obj

    def load(self) -> None:
        torch, _ = self.imports()

        print(f"loading {self.__class__.__name__} '{self.name}' from {self.path} ...")

        self.model = torch.export.load(self.path).module()

        print("done")

    def evaluate(self, *args, **kwargs) -> Any:
        torch, _ = self.imports()

        # cast inputs to torch tensors
        with torch.no_grad():
            args = self.cast_from_numpy(args)
            kwargs = self.cast_from_numpy(kwargs)
            out = self.model(*args, **kwargs)
            return self.cast_to_numpy(out)


class TorchEvaluator(BaseEvaluator):
    """
    Torch model evaluator that runs in a separate process with support for multiple models.

    .. code-block:: python

        evaluator = TorchEvaluator()
        evaluator.add_model("model_name", "path/to/model")
        with evaluator:
            result = evaluator("model_name", input_data)
    """

    def get_model_cls(self) -> Type[BaseModel]:
        return TorchModel

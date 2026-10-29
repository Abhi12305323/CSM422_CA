from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

try:
    import tensorflow as tf
    from tensorflow import keras
except ModuleNotFoundError as error:
    tf = None
    keras = None
    _TF_ERROR: ModuleNotFoundError | None = error
else:
    _TF_ERROR = None


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def _kernel(value: object, index: int) -> tuple[int, int]:
    name = f"kernel_sizes[{index}]"
    if isinstance(value, Integral) and not isinstance(value, bool):
        size = _positive(value, name)
        return size, size
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be an integer or a pair of integers.")
    return _positive(value[0], f"{name}[0]"), _positive(value[1], f"{name}[1]")


@dataclass(frozen=True, slots=True)
class ConvLSTMConfig:
    filters: Sequence[int] = (64, 64)
    kernel_sizes: Sequence[int | Sequence[int]] = ((5, 5), (1, 1))
    variant: str = "causal"
    dropout: float = 0.0
    l2: float = 0.0
    batch_norm: bool = False
    output_channels: int = 1
    output_activation: str | None = None

    def __post_init__(self) -> None:
        filters = tuple(
            _positive(value, f"filters[{index}]")
            for index, value in enumerate(self.filters)
        )
        kernels = tuple(
            _kernel(value, index) for index, value in enumerate(self.kernel_sizes)
        )
        if not filters:
            raise ValueError("filters must contain at least one layer.")
        if len(filters) != len(kernels):
            raise ValueError("filters and kernel_sizes must have the same length.")
        if not isinstance(self.variant, str):
            raise ValueError("variant must be a string.")
        variant = self.variant.strip().lower()
        if variant not in {"causal", "legacy", "regularized_legacy"}:
            raise ValueError(f"unsupported model variant: {self.variant!r}.")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, Real)
            or not 0 <= float(self.dropout) < 1
        ):
            raise ValueError("dropout must be in [0, 1).")
        if (
            isinstance(self.l2, bool)
            or not isinstance(self.l2, Real)
            or not math.isfinite(float(self.l2))
            or self.l2 < 0
        ):
            raise ValueError("l2 must be finite and non-negative.")
        if not isinstance(self.batch_norm, bool):
            raise ValueError("batch_norm must be a boolean.")
        activation = self.output_activation
        if activation is not None and (
            not isinstance(activation, str) or not activation.strip()
        ):
            raise ValueError("output_activation must be a non-empty string or None.")

        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "kernel_sizes", kernels)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "l2", float(self.l2))
        object.__setattr__(
            self,
            "output_channels",
            _positive(self.output_channels, "output_channels"),
        )
        object.__setattr__(
            self,
            "output_activation",
            activation.strip() if activation else None,
        )


def _need_tensorflow() -> None:
    if tf is None:
        raise RuntimeError("TensorFlow is required to build the model.") from _TF_ERROR


def _input_shape(
    shape: Sequence[int | None],
) -> tuple[int | None, int | None, int | None, int]:
    if (
        isinstance(shape, (str, bytes))
        or not isinstance(shape, Sequence)
        or len(shape) != 4
    ):
        raise ValueError(
            "input_shape must have four dimensions: (time, height, width, channels)."
        )
    values: list[int | None] = []
    for index, value in enumerate(shape):
        if value is None and index < 3:
            values.append(None)
        elif isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            suffix = " or None." if index < 3 else "."
            raise ValueError(f"input_shape[{index}] must be positive{suffix}")
        else:
            values.append(int(value))
    return values[0], values[1], values[2], int(values[3])


if keras is not None:

    @keras.utils.register_keras_serializable(package="dlsrwf")
    class LegacyConvLSTM(keras.Model):
        def __init__(
            self,
            config: ConvLSTMConfig,
            name: str = "stacked_network",
            **kwargs: Any,
        ) -> None:
            if config.variant not in {"legacy", "regularized_legacy"}:
                raise ValueError("LegacyConvLSTM requires a legacy variant.")
            super().__init__(name=name, **kwargs)
            self.model_config = config

            # These attribute names are required by the original TensorFlow checkpoints.
            self.nLayers = len(config.filters)
            self.nFilters = list(config.filters)
            self.kernelSize = list(config.kernel_sizes)
            self.convLSTMs: list[keras.layers.Layer] = []
            self.batchNorms: list[keras.layers.Layer] = []

            regularized = config.variant == "regularized_legacy"
            if regularized:
                self.activationS: list[keras.layers.Layer] = []
                self.dOut: list[keras.layers.Layer] = []

            regularizer = keras.regularizers.l2(config.l2) if config.l2 else None
            for index, (filters, kernel_size) in enumerate(
                zip(config.filters, config.kernel_sizes, strict=True)
            ):
                options: dict[str, Any] = {
                    "filters": filters,
                    "kernel_size": kernel_size,
                    "padding": "same",
                    "return_sequences": True,
                }
                if regularized:
                    options["kernel_regularizer"] = regularizer
                else:
                    options["activation"] = "relu"
                self.convLSTMs.append(keras.layers.ConvLSTM2D(**options))
                if regularized or index < self.nLayers - 1:
                    self.batchNorms.append(keras.layers.BatchNormalization())
                if regularized:
                    self.activationS.append(keras.layers.Activation("relu"))
                    self.dOut.append(keras.layers.Dropout(config.dropout))

            self.convNN = keras.layers.Conv3D(
                config.output_channels,
                (3, 3, 3),
                padding="same",
                activation=config.output_activation or "sigmoid",
            )

        def call(
            self,
            inputs: tf.Tensor,
            training: bool | None = None,
            mask: tf.Tensor | None = None,
        ) -> tf.Tensor:
            x = inputs
            regularized = self.model_config.variant == "regularized_legacy"
            batch_norm = 0
            for index, layer in enumerate(self.convLSTMs):
                x = layer(x, training=training, mask=mask)
                if regularized or index < self.nLayers - 1:
                    x = self.batchNorms[batch_norm](x, training=training)
                    batch_norm += 1
                if regularized:
                    x = self.activationS[index](x)
                    x = self.dOut[index](x, training=training)
            return self.convNN(x, training=training)

        def get_config(self) -> dict[str, Any]:
            data = super().get_config()
            data["config"] = asdict(self.model_config)
            return data

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> LegacyConvLSTM:
            values = dict(config)
            model_config = ConvLSTMConfig(**values.pop("config"))
            return cls(model_config, **values)

    @keras.utils.register_keras_serializable(package="dlsrwf")
    class CausalConvLSTM(keras.Model):
        def __init__(
            self,
            config: ConvLSTMConfig,
            name: str = "causal_conv_lstm",
            **kwargs: Any,
        ) -> None:
            if config.variant != "causal":
                raise ValueError("CausalConvLSTM requires variant='causal'.")
            super().__init__(name=name, **kwargs)
            self.model_config = config
            regularizer = keras.regularizers.l2(config.l2) if config.l2 else None
            self.conv_lstms: list[keras.layers.Layer] = []
            self.batch_norms: list[keras.layers.Layer] = []
            self.dropouts: list[keras.layers.Layer] = []

            for filters, kernel_size in zip(
                config.filters, config.kernel_sizes, strict=True
            ):
                self.conv_lstms.append(
                    keras.layers.ConvLSTM2D(
                        filters,
                        kernel_size,
                        padding="same",
                        return_sequences=True,
                        activation="tanh",
                        kernel_regularizer=regularizer,
                        recurrent_regularizer=regularizer,
                    )
                )
                if config.batch_norm:
                    self.batch_norms.append(keras.layers.BatchNormalization())
                self.dropouts.append(keras.layers.Dropout(config.dropout))

            self.output_head = keras.layers.Conv3D(
                config.output_channels,
                (1, 3, 3),
                padding="same",
                activation=config.output_activation,
                name="causal_output",
            )

        def call(
            self,
            inputs: tf.Tensor,
            training: bool | None = None,
            mask: tf.Tensor | None = None,
        ) -> tf.Tensor:
            x = inputs
            for index, layer in enumerate(self.conv_lstms):
                x = layer(x, training=training, mask=mask)
                if self.model_config.batch_norm:
                    x = self.batch_norms[index](x, training=training)
                x = self.dropouts[index](x, training=training)
            return self.output_head(x, training=training)

        def get_config(self) -> dict[str, Any]:
            data = super().get_config()
            data["config"] = asdict(self.model_config)
            return data

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> CausalConvLSTM:
            values = dict(config)
            model_config = ConvLSTMConfig(**values.pop("config"))
            return cls(model_config, **values)

else:

    class LegacyConvLSTM:
        def __init__(self, *_: Any, **__: Any) -> None:
            _need_tensorflow()

    class CausalConvLSTM:
        def __init__(self, *_: Any, **__: Any) -> None:
            _need_tensorflow()


def build_model(
    config: ConvLSTMConfig | Mapping[str, Any],
    input_shape: Sequence[int | None],
) -> Any:
    if isinstance(config, Mapping):
        values = dict(config)
        values.pop("weights", None)
        config = ConvLSTMConfig(**values)
    if not isinstance(config, ConvLSTMConfig):
        raise TypeError("config must be a ConvLSTMConfig or mapping.")
    shape = _input_shape(input_shape)
    _need_tensorflow()
    inputs = keras.Input(shape=shape, name="frames")
    layer = (
        CausalConvLSTM(config) if config.variant == "causal" else LegacyConvLSTM(config)
    )
    return keras.Model(inputs, layer(inputs), name=f"dlsrwf_{config.variant}")


__all__ = ["CausalConvLSTM", "ConvLSTMConfig", "LegacyConvLSTM", "build_model"]

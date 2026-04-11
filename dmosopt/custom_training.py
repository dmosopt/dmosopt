import math
from typing import Optional, Sequence

import numpy as np
import keras
from keras import layers, ops
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.stats import qmc


def preprocess(x, y, yC=None, remove_outliers=False, nan="remove"):
    if nan == "max":
        m = np.max(np.nan_to_num(y), axis=0)
        for c in range(y.shape[1]):
            y[:, c] = np.nan_to_num(y[:, c], nan=2 * m[c])
    elif nan == "remove":
        r = ~np.any(np.isnan(y), axis=1)
        x = x[r]
        y = y[r]
        if yC is not None:
            yC = yC[r]
    else:
        raise ValueError("Invalid nan mode")

    if remove_outliers is True:
        remove_outliers = 2
    mask = slice(None)
    if remove_outliers is not False:
        ylog = np.log(y + 1)
        ylmean = np.mean(ylog, axis=0)
        ylstd = np.std(ylog, axis=0)
        zscores = (ylog - ylmean) / ylstd
        outlier = np.any(zscores > float(remove_outliers), axis=1)
        mask = ~outlier

    if yC is None:
        return x[mask], y[mask], yC

    return x[mask], y[mask], yC[mask]


def acc(y_true, y_pred):
    return ops.mean(
        ops.cast(
            ops.cast(ops.cast(y_pred, "float32") > 0.5, "int32")
            == ops.cast(ops.cast(y_true, "float32") > 0.5, "int32"),
            "float32",
        )
    )


class BoundsNormalization(layers.Layer):
    def __init__(self, xlb, xub, **kwargs):
        super().__init__(**kwargs)
        xrg = np.array(xub) - np.array(xlb)
        self.xlb = ops.convert_to_tensor(xlb, dtype="float32")
        self.xrg = ops.convert_to_tensor(xrg, dtype="float32")

    def call(self, inputs):
        return (inputs - self.xlb) / self.xrg

    def adapt(self, x):
        return None


class TransformerBlock(layers.Layer):
    def __init__(
        self,
        embedding_dimension,
        ff_dimension,
        num_heads,
        attention_dropout,
        ffn_dropout,
        head_dim,
        input_norm=True,
        output_block=False,
        residual_dropout=0.0,
        ffn_activation="gelu",
    ):
        super().__init__()
        self.output_block = output_block
        self.att_normalization = layers.LayerNormalization() if input_norm else None
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            dropout=attention_dropout,
            output_shape=embedding_dimension,
        )
        self.attention_output_dropout = layers.Dropout(residual_dropout)
        self.ffn_normalization = layers.LayerNormalization()
        self.ffn_dense_1 = layers.Dense(ff_dimension, activation=ffn_activation)
        self.ffn_dropout = layers.Dropout(ffn_dropout)
        self.ffn_dense_2 = layers.Dense(embedding_dimension)
        self.ffn_output_dropout = layers.Dropout(residual_dropout)

    def call(self, inputs, training=None):
        x = inputs
        residual = x
        if self.att_normalization is not None:
            x = self.att_normalization(x, training=training)
        attention_out = self.mha(x, x, training=training)
        attention_out = self.attention_output_dropout(attention_out, training=training)
        x = residual + attention_out

        residual = x
        x = self.ffn_normalization(x, training=training)
        x = self.ffn_dense_1(x)
        x = self.ffn_dropout(x, training=training)
        x = self.ffn_dense_2(x)
        x = self.ffn_output_dropout(x, training=training)
        x = residual + x

        if self.output_block:
            x = x[:, 0]

        return x


class JointFTTransformer(keras.Model):
    def __init__(
        self,
        num_parameters,
        num_constraints,
        num_objectives,
        mode="c+o",
        xlb=None,
        xub=None,
        learning_rate=0.001,
        outlier_threshold=0,
        exclude_infeasible=False,
        normalize_targets="range",
        architecture=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_parameters = num_parameters
        self.num_constraints = num_constraints
        self.num_objectives = num_objectives
        self.learning_rate = learning_rate
        self.outlier_threshold = outlier_threshold
        self.exclude_infeasible = exclude_infeasible
        self.normalize_targets = normalize_targets
        if mode not in ["c+o", "c", "o"]:
            raise ValueError("Invalid mode")
        self.mode = mode
        self.xlb = xlb
        self.xub = xub
        self.X_ = None
        self.X_raw_ = None
        self.y_ = None
        self.y_raw_ = None
        self.y_norm_ = None
        self.yC_ = None
        self.yC_raw_ = None

        if xlb is not None and xub is not None:
            self.input_norm_layer = BoundsNormalization(xlb, xub)
        else:
            self.input_norm_layer = layers.Normalization()

        self.timestep = None
        self.architecture = architecture if architecture else {}

        self.prepare_layers(**self.architecture)
        self.autocompile()

        self.min_mean_yR = self.add_weight(
            name="min_mean_yR",
            shape=[num_objectives],
            initializer="zeros",
            trainable=False,
        )
        self.max_std_yR = self.add_weight(
            name="max_std_yR",
            shape=[num_objectives],
            initializer="zeros",
            trainable=False,
        )

        self._last_fit_epochs = -1

    def prepare_layers(
        self,
        n_blocks=3,
        embedding_dim_per_head=32,
        num_heads=4,
        ffn_ratio=2.0,
        use_input_layer_norm=False,
        block_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.05,
        ffn_activation="gelu",
        pooling="cls",
        hidden_mlp_units=None,
        parameter_group_size=1,
        pool_every_n=0,
        pool_stride=2,
        pool_type="avg",
    ):
        embedding_dimension = embedding_dim_per_head * num_heads

        if parameter_group_size <= 1 and self.num_parameters > 64:
            parameter_group_size = max(
                parameter_group_size, math.ceil(self.num_parameters / 64)
            )

        total_targets = self.num_objectives + self.num_constraints
        if total_targets > 100:
            n_blocks = min(n_blocks, 2)
            embedding_dimension = min(embedding_dimension, 128)
            num_heads = min(num_heads, 4)
            block_dropout = max(block_dropout, 0.1)
            parameter_group_size = max(parameter_group_size, 2)
            if pool_every_n == 0:
                pool_every_n = 1
        if total_targets > 200:
            n_blocks = 2
            embedding_dimension = min(embedding_dimension, 96)
            num_heads = min(num_heads, 3)
            parameter_group_size = max(parameter_group_size, 4)
            pool_stride = max(pool_stride, 2)

        if pooling == "cls":
            pool_every_n = 0

        head_dim = max(8, embedding_dimension // num_heads)

        d_rsqrt = embedding_dimension**-0.5
        self.embedding = keras.Sequential(
            [
                layers.Dense(
                    self.num_parameters * embedding_dimension,
                    use_bias=True,
                    kernel_initializer=keras.initializers.RandomUniform(
                        minval=-d_rsqrt, maxval=d_rsqrt
                    ),
                    bias_initializer=keras.initializers.RandomUniform(
                        minval=-d_rsqrt, maxval=d_rsqrt
                    ),
                ),
                layers.Reshape((self.num_parameters, embedding_dimension)),
            ]
        )

        ffn_hidden = int(ffn_ratio * embedding_dimension)

        self.blocks = []
        self.token_poolers = []
        for block_index in range(n_blocks):
            block = TransformerBlock(
                embedding_dimension=embedding_dimension,
                ff_dimension=ffn_hidden,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                head_dim=head_dim,
                input_norm=use_input_layer_norm or block_index != 0,
                output_block=pooling == "cls" and block_index == n_blocks - 1,
                residual_dropout=block_dropout,
                ffn_activation=ffn_activation,
            )
            self.blocks.append(block)

            if (
                pooling != "cls"
                and pool_every_n
                and (block_index + 1) % pool_every_n == 0
            ):
                if pool_type == "avg":
                    pool_layer = layers.AveragePooling1D(
                        pool_size=pool_stride,
                        strides=pool_stride,
                        padding="same",
                    )
                elif pool_type == "max":
                    pool_layer = layers.MaxPooling1D(
                        pool_size=pool_stride,
                        strides=pool_stride,
                        padding="same",
                    )
                else:
                    raise ValueError(f"Unsupported pool_type: {pool_type}")
            else:
                pool_layer = None
            self.token_poolers.append(pool_layer)

        self.output_layer_norm = layers.LayerNormalization()

        if pooling == "mean":
            self.pooling_layer = layers.GlobalAveragePooling1D()
        elif pooling == "flatten":
            self.pooling_layer = layers.Flatten()
        else:
            self.pooling_layer = None

        hidden_mlp_units = list(hidden_mlp_units or [])
        self.head_mlp = keras.Sequential(
            [
                layer
                for units in hidden_mlp_units
                for layer in (
                    layers.Dense(units, activation="relu"),
                    layers.Dropout(block_dropout),
                )
            ]
        )

        self.parameter_group_size = max(1, int(parameter_group_size))
        self.pooling_strategy = pooling

        self.architecture.update(
            {
                "n_blocks": n_blocks,
                "embedding_dim_per_head": embedding_dim_per_head,
                "num_heads": num_heads,
                "ffn_ratio": ffn_ratio,
                "use_input_layer_norm": use_input_layer_norm,
                "block_dropout": block_dropout,
                "attention_dropout": attention_dropout,
                "ffn_dropout": ffn_dropout,
                "ffn_activation": ffn_activation,
                "pooling": pooling,
                "hidden_mlp_units": hidden_mlp_units,
                "parameter_group_size": self.parameter_group_size,
                "pool_every_n": pool_every_n,
                "pool_stride": pool_stride,
                "pool_type": pool_type,
            }
        )

        self.objectives_output = layers.Dense(self.num_objectives)
        self.constraints_output = layers.Dense(
            self.num_constraints, activation="sigmoid", name="constraints"
        )

    def call(self, inputs, training=None):
        x = self.input_norm_layer(inputs)
        x = self.embedding(x)

        if self.parameter_group_size > 1:
            group_size = self.parameter_group_size
            x_shape = ops.shape(x)
            num_tokens = x_shape[1]
            group_size_tensor = ops.convert_to_tensor(
                group_size, dtype=num_tokens.dtype
            )
            remainder = ops.mod(num_tokens, group_size_tensor)
            pad = ops.mod(group_size_tensor - remainder, group_size_tensor)
            padding = ops.zeros((x_shape[0], pad, x_shape[2]), dtype=x.dtype)
            x = ops.concatenate([x, padding], axis=1)
            x_shape = ops.shape(x)
            new_tokens = x_shape[1] // group_size_tensor
            new_shape = ops.stack(
                [x_shape[0], new_tokens, group_size_tensor, x_shape[2]]
            )
            x = ops.reshape(x, new_shape)
            x = ops.mean(x, axis=2)

        for block, pooler in zip(self.blocks, self.token_poolers):
            x = block(x, training=training)
            if pooler is not None:
                x = pooler(x)

        x = self.output_layer_norm(x, training=training)

        if self.pooling_layer is not None:
            x = self.pooling_layer(x)

        if self.head_mlp.layers:
            x = self.head_mlp(x, training=training)

        if self.mode == "c+o":
            return {
                "objectives": self.objectives_output(x),
                "constraints": self.constraints_output(x),
            }
        if self.mode == "c":
            return self.constraints_output(x)
        if self.mode == "o":
            return self.objectives_output(x)

    def objective_loss(self, y_true, y_pred, alpha=1, beta=0.01, verbose=False):
        mins = ops.min(self.y_norm_, axis=0)
        maxs = ops.max(self.y_norm_, axis=0)

        weights = 1

        if verbose:
            print("mins", mins)
            print("maxs", maxs)
            print(y_true, "y_true")
            print(y_pred, "y_pred")

        ytrue = y_true - mins
        ypred = y_pred - mins

        if verbose:
            print(ytrue, "ytrue")
            print(ypred, "ypred")

        rerr = ops.abs(ytrue - ypred) / (alpha * ops.abs(ytrue) + beta)

        if verbose:
            print(rerr, "rerr")

        rerr_ = ops.mean(rerr, axis=0)
        return ops.mean(weights * rerr_)

    def label(self):
        return "joint-" + self.mode

    def new(self):
        return self.__class__(
            num_parameters=self.num_parameters,
            num_constraints=self.num_constraints,
            num_objectives=self.num_objectives,
            mode=self.mode,
            xlb=self.xlb,
            xub=self.xub,
            learning_rate=self.learning_rate,
            outlier_threshold=self.outlier_threshold,
            exclude_infeasible=self.exclude_infeasible,
            normalize_targets=self.normalize_targets,
            architecture=self.architecture,
        )

    def build(self, input_shape=None):
        if input_shape is None:
            input_shape = [1, self.num_parameters]
        self.call(ops.ones([1, input_shape[-1]]))

    def preprocess(self, x, y, yC=None, remove_outliers=False, nan="remove"):
        return preprocess(
            x,
            y,
            yC,
            remove_outliers=self.outlier_threshold if remove_outliers else False,
            nan=nan,
        )

    def autocompile(self):
        if self.mode == "c+o":
            loss = {"objectives": "mse", "constraints": "binary_crossentropy"}
            metrics = {"objectives": ["mae"], "constraints": [acc]}
        elif self.mode == "c":
            loss = "binary_crossentropy"
            metrics = ["acc", keras.metrics.Precision(), keras.metrics.Recall()]
        elif self.mode == "o":
            loss = "mse"
            metrics = ["mae"]

        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics,
        )

    def autofit(
        self,
        x,
        y,
        yC,
        epochs="auto",
        batch_size=2048,
        verbose=2,
        **kwargs,
    ):
        if epochs == "auto":
            m = self.autoepoch(x, y, yC, verbose=1)
            print("Automatic epochs: ", m, " -> ", np.mean(m))
            epochs = np.mean(m)
        else:
            self.build(input_shape=x.shape)

        epochs = int(epochs)

        self.X_raw_ = x
        self.y_raw_ = y
        self.yC_raw_ = yC

        x, y, yC = self.preprocess(x, y, yC)

        if yC is not None and self.exclude_infeasible:
            feasible = np.argwhere(np.all(yC > 0.0, axis=1))
            if len(feasible) > 0:
                feasible = feasible.ravel()
                x = x[feasible, :]
                y = y[feasible, :]

        if self.mode == "c+o":
            Y = {"objectives": y, "constraints": yC}
        elif self.mode == "c":
            Y = yC
        elif self.mode == "o":
            Y = y

        return self.fit(
            x,
            Y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
            ],
            **kwargs,
        )

    def autoeval(self, x, y, yC, verbose=2):
        x, y, yC = self.preprocess(x, y, yC)
        if self.mode == "c+o":
            Y = {"objectives": y, "constraints": yC}
        elif self.mode == "c":
            Y = yC
        elif self.mode == "o":
            Y = y
        return self.eval(x, Y, verbose=verbose)

    def autoepoch(
        self, x, y, yC, n_splits=3, timeout_samples=1e8, verbose=1, cv="kfold"
    ):
        if x.shape[0] < n_splits * 2:
            return [1]

        x, y, yC = self.preprocess(x, y, yC)

        if yC is not None and self.exclude_infeasible:
            feasible = np.argwhere(np.all(yC > 0.0, axis=1))
            if len(feasible) > 0:
                feasible = feasible.ravel()
                x = x[feasible, :]
                y = y[feasible, :]

        if x.shape[0] < n_splits * 2:
            return [1]

        kf = {"kfold": KFold, "time_series": TimeSeriesSplit}[cv](n_splits=n_splits)
        stopped_after_epochs = []
        timeout_epochs = max(25, min(round(timeout_samples / x.shape[0]), 10000))
        epoch_increment = max(10, round(timeout_epochs / 10.0))

        def p(*args, **kwargs):
            if verbose > 0:
                print(*args, **kwargs)

        self.build(input_shape=x.shape)
        # Compile once to avoid repeated XLA re-tracing during CV sweeps
        self.autocompile()
        initial_weights = self.get_weights()
        optimizer_reset = getattr(self.optimizer, "reset_state", None)

        p("Autoepoch cross-validation ...")
        for s, (train_index, val_index) in enumerate(kf.split(x)):
            p(f"Split {s}")
            self.set_weights(initial_weights)
            if optimizer_reset is not None:
                optimizer_reset()
            else:
                self.autocompile()

            X_train, X_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]
            yC_train, yC_val = None, None
            if yC is not None:
                yC_train, yC_val = yC[train_index], yC[val_index]

            total_epochs = 0
            while total_epochs < timeout_epochs:
                p(f"{total_epochs} / {timeout_epochs} ({epoch_increment})")
                if self.mode == "c+o":
                    y_ = {"objectives": y_train, "constraints": yC_train}
                    val_ = (
                        X_val,
                        {
                            "objectives": y_val,
                            "constraints": yC_val,
                        },
                    )
                    monitor_metrics = ["val_objectives_loss"]
                elif self.mode == "c":
                    y_ = yC_train
                    val_ = (X_val, yC_val)
                    monitor_metrics = ["val_loss"]
                elif self.mode == "o":
                    y_ = y_train
                    val_ = (X_val, y_val)
                    monitor_metrics = ["val_mae"]

                history = self.fit(
                    X_train,
                    y_,
                    validation_data=val_,
                    epochs=total_epochs + epoch_increment,
                    batch_size=2048,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor=mon,
                            patience=250,
                            restore_best_weights=False,
                            mode="min",
                        )
                        for mon in monitor_metrics
                    ]
                    + [keras.callbacks.TerminateOnNaN()],
                    verbose=verbose,
                    initial_epoch=total_epochs,
                )
                epochs_this_round = len(history.epoch)
                total_epochs += epochs_this_round
                if epochs_this_round < epoch_increment:
                    p(
                        f"Stopping at {epochs_this_round} < {epoch_increment} (total: {total_epochs})"
                    )
                    break

            p(f"Stopped after {total_epochs} for split {s}")
            stopped_after_epochs.append(total_epochs)

        self.set_weights(initial_weights)
        if optimizer_reset is not None:
            optimizer_reset()
        else:
            self.autocompile()
        return stopped_after_epochs

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        *args,
        **kwargs,
    ):
        self.X_ = x
        if self.mode == "c+o":
            self.y_ = y["objectives"]
            self.yC_ = y["constraints"]
        elif self.mode == "c":
            self.y_ = None
            self.yC_ = y
        elif self.mode == "o":
            self.y_ = y
            self.yC_ = None

        if self.timestep is not None:
            self.timestep.assign(0)

        self.input_norm_layer.adapt(x)
        self._last_fit_epochs = epochs

        if self.mode == "c+o":
            self.y_norm_ = np.array(self.norm_output(y["objectives"], adapt=True))
            if validation_data is not None:
                validation_data = (
                    validation_data[0],
                    {
                        "objectives": np.array(
                            self.norm_output(validation_data[1]["objectives"])
                        ),
                        "constraints": validation_data[1]["constraints"],
                    },
                )
            return super().fit(
                x,
                {
                    "objectives": self.y_norm_,
                    "constraints": y["constraints"],
                },
                batch_size,
                epochs,
                verbose,
                callbacks,
                validation_split,
                validation_data,
                *args,
                **kwargs,
            )
        elif self.mode == "c":
            return super().fit(
                x,
                y,
                batch_size,
                epochs,
                verbose,
                callbacks,
                validation_split,
                validation_data,
                *args,
                **kwargs,
            )
        else:
            self.y_norm_ = np.array(self.norm_output(y, adapt=True))
            if validation_data is not None:
                validation_data = (
                    validation_data[0],
                    np.array(self.norm_output(validation_data[1])),
                )
            return super().fit(
                x,
                self.y_norm_,
                batch_size,
                epochs,
                verbose,
                callbacks,
                validation_split,
                validation_data,
                *args,
                **kwargs,
            )

    def norm_output(self, yR, inverse=False, adapt=False, method=None):
        if method is None:
            method = self.normalize_targets

        if method is False:
            return ops.convert_to_tensor(yR, dtype="float32")

        if adapt:
            if "max" in method or method == "range":
                if "0" in method:
                    self.min_mean_yR.assign(np.zeros([yR.shape[1]]))
                else:
                    self.min_mean_yR.assign(np.min(yR, axis=0))
                self.max_std_yR.assign(np.max(yR, axis=0))
            elif method == "standard":
                self.min_mean_yR.assign(np.mean(yR, axis=0))
                self.max_std_yR.assign(np.std(yR, axis=0))
            elif method == "log":
                pass

        if "max" in method:
            centering = 0.0
            if "c" in method:
                centering = 0.5
            if inverse:
                return (yR + centering) * (
                    self.max_std_yR - self.min_mean_yR
                ) + self.min_mean_yR
            return (
                (yR - self.min_mean_yR)
                / (self.max_std_yR - self.min_mean_yR + keras.backend.epsilon())
            ) - centering
        elif method == "range":
            top = ops.max(self.max_std_yR)
            bottom = ops.max(self.min_mean_yR)
            if inverse:
                exps = ops.exp(yR) - 1
                scaled = (exps - bottom) / (top - bottom + keras.backend.epsilon())
                return scaled * (self.max_std_yR - self.min_mean_yR) + self.min_mean_yR
            normalized = (yR - self.min_mean_yR) / (
                self.max_std_yR - self.min_mean_yR + keras.backend.epsilon()
            )
            normalized = ops.clip(normalized, 0.0, 1.0)
            upscaled = normalized * (top - bottom) + bottom
            lower_bound = ops.convert_to_tensor(-1.0 + 1e-6, dtype=upscaled.dtype)
            upscaled = ops.maximum(upscaled, lower_bound)
            return ops.log1p(upscaled)
        elif method == "standard":
            if inverse:
                return yR * self.max_std_yR + self.min_mean_yR
            return (yR - self.min_mean_yR) / (self.max_std_yR + keras.backend.epsilon())
        elif "log" in method:
            if inverse:
                if method == "log_":
                    return ops.convert_to_tensor(yR)
                return ops.exp(yR) - 1
            return ops.log1p(yR)
        else:
            raise ValueError(f"Invalid scaling method: {method}.")

    def get_output_norm(self):
        if self.normalize_targets is False:
            return None
        return np.array(self.min_mean_yR).tolist(), np.array(self.max_std_yR).tolist()

    def eval(self, X_test, y_test, per_feature=False, verbose=1):
        def normed(metric):
            def _w(y_true, y_pred, *args, **kwargs):
                return metric(
                    np.nan_to_num(np.array(self.norm_output(y_true))),
                    np.nan_to_num(np.array(self.norm_output(y_pred))),
                    *args,
                    **kwargs,
                )

            return _w

        if self.mode == "c+o":
            assert not per_feature, (
                "Joint model does not support per_feature evaluation"
            )
            y_pred = self.predict(X_test, verbose=verbose)

            y_test_prime = y_test["constraints"].all(axis=1).astype(int)
            y_pred_prime = (
                (y_pred["constraints"] > 0.5).astype(int).all(axis=1)
            ).astype(int)

            y_pred_C = (y_pred["constraints"] > 0.5).astype(int)
            y_test_C = y_test["constraints"].astype(int)

            return {
                "epochs": self._last_fit_epochs,
                "accuracy": float(accuracy_score(y_test_C, y_pred_C)),
                "precision": float(
                    precision_score(y_test_C, y_pred_C, average="micro")
                ),
                "recall": float(recall_score(y_test_C, y_pred_C, average="micro")),
                "f1": float(f1_score(y_test_C, y_pred_C, average="micro")),
                "global_accuracy": float(accuracy_score(y_test_prime, y_pred_prime)),
                "global_precision": float(precision_score(y_test_prime, y_pred_prime)),
                "global_recall": float(recall_score(y_test_prime, y_pred_prime)),
                "global_f1": float(f1_score(y_test_prime, y_pred_prime)),
                "mdae": float(
                    normed(median_absolute_error)(
                        y_test["objectives"],
                        y_pred["objectives"],
                    )
                ),
                "mae": float(
                    normed(mean_absolute_error)(
                        y_test["objectives"],
                        y_pred["objectives"],
                    )
                ),
            }

        if self.mode == "c":
            y_prob = self.predict(X_test, verbose=verbose)
            y_pred = (y_prob > 0.5).astype(int)

            y_test_prime = y_test.all(axis=1).astype(int)
            y_pred_prime = y_pred.all(axis=1).astype(int)

            if per_feature:
                tbl = [["Constraint", "Precision", "Recall", "F1"]]
                labels = per_feature
                prec = precision_score(y_test, y_pred, average=None)
                rec = recall_score(y_test, y_pred, average=None)
                f1 = f1_score(y_test, y_pred, average=None)
                for t in zip(labels, prec, rec, f1):
                    tbl.append(t)
                tbl.append(
                    [
                        "Total",
                        precision_score(y_test_prime, y_pred_prime),
                        recall_score(y_test_prime, y_pred_prime),
                        f1_score(y_test_prime, y_pred_prime),
                    ]
                )
                return tbl

            if verbose > 2:
                print("\nMisclassified samples:")
                diff_mask = y_pred != y_test
                for i in range(len(y_test)):
                    if diff_mask[i].any():
                        print(f"Row {i}:")
                        print(f"Predicted: {y_pred[i]}")
                        print(f"Actual:    {y_test[i]}")
                        print()

            return {
                "epochs": self._last_fit_epochs,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="macro")),
                "recall": float(recall_score(y_test, y_pred, average="macro")),
                "f1": float(f1_score(y_test, y_pred, average="macro")),
                "global_accuracy": float(accuracy_score(y_test_prime, y_pred_prime)),
                "global_precision": float(precision_score(y_test_prime, y_pred_prime)),
                "global_recall": float(recall_score(y_test_prime, y_pred_prime)),
                "global_f1": float(f1_score(y_test_prime, y_pred_prime)),
            }

        if self.mode == "o":
            y_pred = self.predict(X_test, verbose=verbose)
            return {
                "epochs": self._last_fit_epochs,
                "mdae": normed(median_absolute_error)(
                    y_test, y_pred, multioutput="raw_values"
                ).tolist(),
                "r2": normed(r2_score)(
                    y_test, y_pred, multioutput="raw_values"
                ).tolist(),
                "mae": normed(mean_absolute_error)(
                    y_test, y_pred, multioutput="raw_values"
                ).tolist(),
            }

    def global_accuracy(self, y_true, y_pred):
        y_true = ops.cast(y_true, "bool")
        y_pred = ops.cast(y_pred, "bool")
        y_true = ops.cast(ops.cast(ops.all(y_true, axis=1), "int32"), "float32")
        y_pred = ops.cast(ops.cast(ops.all(y_pred, axis=1), "int32"), "float32")
        return keras.metrics.binary_accuracy(y_true, y_pred)

    def predict_objectives(self, X, nan_to_num=False, max_zero=False, verbose=0):
        yR = self.predict(X, verbose=verbose)
        if self.mode == "c+o":
            yR = yR["objectives"]
        yR = np.array(yR)
        if nan_to_num:
            yR = np.nan_to_num(yR)
        if max_zero:
            yR = np.maximum(np.zeros_like(yR), yR)
        return yR

    def predict(self, x, *args, **kwargs):
        y_pred = super().predict(x, *args, **kwargs)
        if self.mode == "c+o":
            return {
                "constraints": y_pred["constraints"],
                "objectives": np.array(
                    self.norm_output(y_pred["objectives"], inverse=True)
                ),
            }
        if self.mode == "c":
            return y_pred
        if self.mode == "o":
            return np.array(self.norm_output(y_pred, inverse=True))

    def raw_predict(self, x, *args, **kwargs):
        return super().predict(x, *args, **kwargs)

    def _compute_input_gradients(self, inputs, forward_fn, key):
        backend_name = keras.backend.backend()

        if backend_name == "jax":
            import jax

            def scalar_fn(inp):
                return ops.sum(forward_fn(inp)[key])

            return jax.grad(scalar_fn)(inputs)

        elif backend_name == "tensorflow":
            import tensorflow as tf

            inputs_tensor = tf.cast(inputs, tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(inputs_tensor)
                scalar = tf.reduce_sum(forward_fn(inputs_tensor)[key])
            return tape.gradient(scalar, inputs_tensor)

        elif backend_name == "torch":
            import torch

            if not isinstance(inputs, torch.Tensor):
                inputs_t = torch.tensor(
                    np.array(inputs), dtype=torch.float32, requires_grad=True
                )
            else:
                inputs_t = inputs.detach().requires_grad_(True)
            scalar = torch.sum(forward_fn(inputs_t)[key])
            scalar.backward()
            return inputs_t.grad

        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

    def sensitivity(
        self,
        X,
        reduction=lambda x: ops.mean(ops.abs(x), axis=0),
        batch_size=1024,
    ):
        X = ops.convert_to_tensor(X, dtype="float32")
        num_samples = X.shape[0]

        def forward(batch_inputs):
            outputs = self(batch_inputs, training=False)
            if self.mode == "o":
                outputs = {"objectives": outputs, "constraints": None}
            elif self.mode == "c":
                outputs = {"objectives": None, "constraints": outputs}
            return outputs

        sens: dict[str, Optional[np.ndarray]] = {}
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X[start_idx:end_idx]

            outputs = forward(X_batch)

            for key, val in outputs.items():
                if val is None:
                    sens[key] = None
                    continue

                grads = self._compute_input_gradients(X_batch, forward, key)
                grads = grads * X_batch

                batch_sens = np.array(reduction(grads))
                if key not in sens or sens[key] is None:
                    sens[key] = np.zeros_like(batch_sens)
                sens[key] += batch_sens * (end_idx - start_idx) / num_samples

        return sens


def salib_finite_difference_sampling(
    num_vars: int,
    bounds: Sequence[tuple[float, float]],
    N: int,
    delta: float = 0.01,
    skip_values: int = 1024,
) -> np.ndarray:
    engine = qmc.Sobol(d=num_vars, scramble=False)
    m = math.ceil(math.log2(N + skip_values)) if N + skip_values > 0 else 0
    base_sequence = np.array(
        engine.random_base2(m=m)[: N + skip_values], dtype=np.float32
    )
    lower = np.array([b[0] for b in bounds], dtype=np.float32)
    upper = np.array([b[1] for b in bounds], dtype=np.float32)
    span = upper - lower

    base_sequence = lower + base_sequence * span
    base_sequence = base_sequence[skip_values : skip_values + N]

    base_delta = base_sequence * delta
    eye = np.eye(num_vars, dtype=np.float32)

    # [N, D + 1, D]
    perturbed = base_sequence[:, None, :] + base_delta[:, :, None] * eye
    perturbed = np.clip(perturbed, lower, upper)
    combined = np.concatenate([base_sequence[:, None, :], perturbed], axis=1)

    return combined.reshape((-1, num_vars))


def joint(
    optimizer_cls,
    Xinit,
    Yinit,
    C,
    xlb,
    xub,
    file_path,
    options,
    # -
    mode="c+o",
    objectives=True,
    constraints=False,
    sensitivity=True,
    epochs="auto",
    iterations=[],
):
    """
    Joint model surrogate custom training function

    Can be used as `surrogate_custom_training` in dmosopt configuration

    Arguments:
        mode: What information to use when training the model
            ("c" = constraints, "o" = objectives, "c+o" = both)
        objectives: Whether to use this model for objective prediction
        constraints: Whether to use this model for constraint/feasibility prediction
        sensitivity: Whether to use this model for sensitivity analysis
        epochs: Number of training epochs ("auto" for cross-validated selection)
    """
    x = Xinit.copy()
    y = Yinit.copy()
    yC = None
    if C is not None:
        yC = (C > 0).astype(int)

    class _Model:
        def __init__(self, model) -> None:
            self._wrapped = model

        def rank(self, x):
            # return dummy; will be reset in Optimizer
            return None

        def evaluate(self, x):
            return self.predict_objectives(x)

        def di_dict(self):
            points = salib_finite_difference_sampling(
                num_vars=len(xlb), bounds=list(zip(xlb, xub)), N=10000
            )

            sens = self.sensitivity(
                points, reduction=lambda x: ops.mean(ops.square(x), axis=0)
            )["objectives"]

            sens = sens / (ops.max(sens) + 1e-7)

            computed_di_crossover = 1 + (np.abs(sens) * 20)
            computed_di_mutation = 1 + (np.abs(sens) * 20)
            di_crossover = np.maximum(1, np.minimum(30, computed_di_crossover))
            di_mutation = np.maximum(1, np.minimum(30, computed_di_mutation))

            return {
                "di_mutation": di_mutation,
                "di_crossover": di_crossover,
            }

        def __call__(self, *args, **kwargs):
            return self._wrapped(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    model = _Model(
        JointFTTransformer(
            num_parameters=Xinit.shape[1],
            num_constraints=C.shape[1] if C is not None else 0,
            num_objectives=Yinit.shape[1],
            mode=mode,
            xlb=xlb,
            xub=xub,
        )
    )

    model.autofit(x, y, yC, verbose=1, epochs=epochs)

    scores = model.autoeval(x, y, yC)

    scores["num_samples"] = x.shape[0]
    scores["iteration"] = len(iterations)

    model.stats = {f"model_{k}": np.mean(v) for k, v in scores.items()}

    return (
        optimizer_cls,
        model if objectives else None,
        model if constraints else None,
        model if sensitivity else None,
    )

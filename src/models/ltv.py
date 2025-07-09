from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras
import tf_keras.layers as tfkl
from tensorflow.python.ops.linalg.linear_operator_block_diag import (
    LinearOperatorBlockDiag,
)
from tensorflow.python.ops.linalg.linear_operator_full_matrix import (
    LinearOperatorFullMatrix,
)

tfpl = tfp.layers
tfd = tfp.distributions


class RBFKernelLayer(tf_keras.layers.Layer):
    def __init__(
        self,
        kernel_amplitude: float,
        kernel_length_scale: float,
        num_timesteps: int,
        eps: float = 1e-5,
        dtype: tf.DType = tf.float32,
    ) -> None:
        """
        A truncated RBF kernel layer with trainable amplitude and length scale.

        Args:
            kernel_amplitude: The initial amplitude of the RBF kernel.
            kernel_length_scale: The initial scale of the RBF kernel.
            num_timesteps: The number of timesteps in the RBF kernel.
            eps: The epsilon parameter added to the diagonal of the RBF kernel to ensure numeric stability.
            dtype: The dtype of the RBF kernel.
        """
        super(RBFKernelLayer, self).__init__(dtype=dtype)
        self.kernel_amplitude_init: float = kernel_amplitude
        self.kernel_length_scale_init: float = kernel_length_scale
        self.num_timesteps: int = num_timesteps
        self.eps: float = eps
        self.kernel_amplitude_var: tf.Variable
        self.kernel_length_scale_var: tf.Variable

    def build(self, input_shape):
        """
        Creates the trainable variables here.
        """
        self.kernel_amplitude_var = self.add_weight(
            name="kernel_amplitude",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.kernel_amplitude_init),
            trainable=True,
        )

        self.kernel_length_scale_var = self.add_weight(
            name="kernel_length_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.kernel_length_scale_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        dist = tf.reshape(tf.range(self.num_timesteps, dtype=self.dtype), [1, -1])

        kernel = self.kernel_amplitude_var * tf.exp(
            -tf.square(dist - tf.transpose(dist))
            / (2.0 * tf.square(self.kernel_length_scale_var))
        )

        kernel += (
            tf.eye(self.num_timesteps, dtype=self.dtype) * self.eps
        )  # Ensure positive definiteness

        return kernel


class PriorBlockCovarianceLayer(tf_keras.layers.Layer):
    def __init__(self, num_taps, dtype=tf.float32) -> None:
        """
        A custom Block Diagonal Covariance Layer to specify intra tap dependence over time but inter dependence across
        taps.
        Args:
            num_taps: The number of taps in the Covariance layer.
            dtype: The dtype of the Covariance layer.
        """
        super(PriorBlockCovarianceLayer, self).__init__(dtype=dtype)
        self.num_taps = num_taps

    def call(self, kernel, *args, **kwargs) -> tf.Tensor:
        block_diag_cov: LinearOperatorBlockDiag = LinearOperatorBlockDiag(
            [
                LinearOperatorFullMatrix(
                    kernel, is_positive_definite=True, is_self_adjoint=True
                )
                for _ in range(self.num_taps)
            ]
        )
        return block_diag_cov.to_dense()


class LtvGpModel(tf_keras.Model):
    def __init__(
        self,
        input_length: int,
        init_filters: int,
        num_taps: int,
        kernel_amplitude: float,
        kernel_length_scale: float,
        kl_scaler: float = 1.0,
        kernel_eps: float = 1e-5,
        dtype: tf.DType = tf.float32,
    ) -> None:
        """
        A Variational Bayesian model of a LTV channel that is parametrized by a Gaussian Process IR with time varying
        mean and covariance. The Gaussian Process is parameterized by a CNN network.

        Args:
            input_length: The length of the input sequence.
            init_filters: The initial filter size.
            num_taps: The number of taps in the IR.
            kernel_amplitude: The initial amplitude of the RBF kernel.
            kernel_length_scale: The initial scale of the RBF kernel.
            kernel_eps: The epsilon parameter added to the diagonal of the kernel to ensure numeric stability.
            dtype: The dtype of the model.
        """
        super().__init__(dtype=dtype)

        self.layers_ = []
        self.input_length: int = input_length
        self.init_filters: int = init_filters
        self.num_taps: int = num_taps
        self.prior_kernel_size: int = input_length
        self.kl_scaler: float = kl_scaler
        kernel_size: int = input_length
        scale_factor: int = 4

        while input_length > 0:
            self.layers_.append(
                tfkl.Conv1D(
                    filters=init_filters,
                    kernel_size=kernel_size,
                    strides=1 if input_length == self.input_length else scale_factor,
                    padding="same",
                    activation=tf.nn.leaky_relu,
                )
            )
            input_length //= scale_factor
            init_filters *= 2
            kernel_size //= scale_factor

        self.layers_.append(tfkl.Flatten())

        self.layers_.append(
            tfkl.Dense(
                tfpl.MultivariateNormalTriL.params_size(
                    self.prior_kernel_size * num_taps
                ),
                activation=None,
            )
        )

        self.layers_.append(
            tfpl.MultivariateNormalTriL(self.prior_kernel_size * num_taps)
        )

        self.kernel_layer: RBFKernelLayer = RBFKernelLayer(
            kernel_amplitude=kernel_amplitude,
            kernel_length_scale=kernel_length_scale,
            num_timesteps=self.prior_kernel_size,
            eps=kernel_eps,
            dtype=dtype,
        )

    def call(self, inputs, *args, **kwargs):
        net_inputs = inputs["net_inputs"]
        conv_inputs = inputs["conv_inputs"]
        batch_size = tf.shape(net_inputs)[0]

        for layer in self.layers_:
            net_inputs = layer(net_inputs)

        kernel = self.kernel_layer(None)
        prior_cov_matrix = PriorBlockCovarianceLayer(self.num_taps)(kernel)

        prior_gp: tfd.MultivariateNormalTriL = tfd.MultivariateNormalTriL(
            loc=tf.zeros([self.prior_kernel_size * self.num_taps]),
            scale_tril=tf.linalg.cholesky(prior_cov_matrix),
        )

        # Calculate the KL divergence, note keras fit adds this loss automatically.
        kl_per_sample = tfd.kl_divergence(net_inputs, prior_gp, allow_nan_stats=False)
        scalar_kl_loss = tf.reduce_mean(kl_per_sample)
        self.add_loss(scalar_kl_loss * self.kl_scaler)

        sample = tf.transpose(
            tf.reshape(
                net_inputs.sample(), (batch_size, self.num_taps, self.prior_kernel_size)
            ),
            perm=[0, 2, 1],
        )
        pred = tf.reduce_sum(conv_inputs * sample, axis=-1)
        return pred

    def params(self, inputs):
        net_inputs = inputs["net_inputs"]
        batch_size = tf.shape(net_inputs)[0]

        for i, layer in enumerate(self.layers_):
            net_inputs = layer(net_inputs)

        means: tf.Tensor = tf.transpose(
            tf.reshape(
                net_inputs.mean(), (batch_size, self.num_taps, self.prior_kernel_size)
            ),
            perm=[0, 2, 1],
        )

        std: tf.Tensor = tf.transpose(
            tf.reshape(
                net_inputs.stddev(), (batch_size, self.num_taps, self.prior_kernel_size)
            ),
            perm=[0, 2, 1],
        )

        return means, std


def get_ltv_model(
    input_length: int,
    init_filters: int,
    num_taps: int,
    kernel_amplitude: float,
    kernel_length_scale: float,
    initial_learning_rate: float,
    target_learning_rate: float,
    warmup_steps: int,
    epochs: int,
    kl_scaler: float,
    kernel_eps: float = 1e-5,
) -> LtvGpModel:
    with tf.device("/GPU:0"):
        model: LtvGpModel = LtvGpModel(
            input_length=input_length,
            init_filters=init_filters,
            num_taps=num_taps,
            kernel_amplitude=kernel_amplitude,
            kernel_length_scale=kernel_length_scale,
            kl_scaler=kl_scaler,
            kernel_eps=kernel_eps,
        )

        loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = lambda y, y_hat: tf.sqrt(
            2 * tf.nn.l2_loss(y - y_hat)
        )

        decay_steps: int = epochs - warmup_steps

        lr_warmup_decayed_fn: tf_keras.optimizers.schedules.CosineDecay = (
            tf_keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                warmup_target=target_learning_rate,
                warmup_steps=warmup_steps,
            )
        )

        model.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=lr_warmup_decayed_fn),
            loss=loss,
        )
    return model

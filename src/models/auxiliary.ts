import * as tf from '@tensorflow/tfjs-node';

/**
 * Volatility prediction auxiliary task.
 *
 * Shares the market embedding with the Q-network and predicts
 * future price volatility (variance of returns). This forces the
 * embedding to encode risk-relevant information, making the agent
 * risk-aware without explicitly penalizing risky actions.
 *
 * Loss: MSE between predicted and actual volatility.
 * Total training loss = L_Q + alpha * L_volatility.
 */
export class VolatilityPredictor {
  private model: tf.LayersModel;
  readonly inputSize: number;

  /**
   * @param inputSize Dimension of the market embedding input.
   * @param hiddenSize Hidden layer size in the prediction MLP.
   */
  constructor(inputSize: number, hiddenSize = 64) {
    this.inputSize = inputSize;

    const input = tf.input({ shape: [inputSize] });
    const h = tf.layers.dense({ units: hiddenSize, activation: 'relu' }).apply(input);
    const output = tf.layers.dense({ units: 1, activation: 'linear' }).apply(h);

    this.model = tf.model({ inputs: input, outputs: output as tf.SymbolicTensor });
  }

  /** Predict volatility from market embedding. Returns [batch, 1]. */
  predict(embedding: tf.Tensor2D): tf.Tensor2D {
    return this.model.predict(embedding) as tf.Tensor2D;
  }

  /** MSE loss between predicted and actual volatility. */
  loss(embedding: tf.Tensor2D, actualVolatility: tf.Tensor2D): tf.Scalar {
    const predicted = this.predict(embedding);
    const mse = predicted.sub(actualVolatility).square().mean() as tf.Scalar;
    predicted.dispose();
    return mse;
  }

  get layers(): tf.layers.Layer[] {
    return this.model.layers;
  }

  /**
   * Compute realized volatility from a price sequence.
   * Volatility = variance of returns over the window.
   */
  static computeVolatility(prices: number[], windowSize = 30): number[] {
    const volatilities: number[] = [];

    for (let t = 0; t < prices.length; t++) {
      const start = Math.max(0, t - windowSize + 1);
      const window = prices.slice(start, t + 1);

      if (window.length < 2) {
        volatilities.push(0);
        continue;
      }

      // Returns within the window
      const returns: number[] = [];
      for (let i = 1; i < window.length; i++) {
        if (window[i - 1] !== 0) {
          returns.push(window[i] / window[i - 1] - 1);
        }
      }

      if (returns.length === 0) {
        volatilities.push(0);
        continue;
      }

      // Variance of returns
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / returns.length;
      volatilities.push(variance);
    }

    return volatilities;
  }
}

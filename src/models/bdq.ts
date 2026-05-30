import * as tf from '@tensorflow/tfjs-node';

/**
 * Branching Dueling Q-Network (BDQ) for 2D action spaces.
 *
 * Splits the Q-value estimation into:
 * - A shared state value V(s)
 * - Independent advantage branches A_price(s, a) and A_qty(s, a)
 *
 * Q_d(s, a_d) = V(s) + (A_d(s, a_d) - mean(A_d(s, .)))
 *
 * This reduces action space complexity from priceLevels * qtyLevels
 * to priceLevels + qtyLevels.
 */
export class BranchingDuelingQNetwork {
  private sharedNet: tf.LayersModel;
  private valueBranch: tf.LayersModel;
  private priceBranch: tf.LayersModel;
  private quantityBranch: tf.LayersModel;

  readonly inputSize: number;
  readonly priceLevels: number;
  readonly quantityLevels: number;

  constructor(inputSize: number, priceLevels = 5, quantityLevels = 5, hiddenSize = 128) {
    this.inputSize = inputSize;
    this.priceLevels = priceLevels;
    this.quantityLevels = quantityLevels;

    // Shared feature network
    const sharedInput = tf.input({ shape: [inputSize] });
    const s1 = tf.layers.dense({ units: hiddenSize, activation: 'relu' }).apply(sharedInput);
    const s2 = tf.layers.dense({ units: hiddenSize, activation: 'relu' }).apply(s1);
    this.sharedNet = tf.model({ inputs: sharedInput, outputs: s2 as tf.SymbolicTensor });

    // State value branch: V(s) -> scalar
    const vInput = tf.input({ shape: [hiddenSize] });
    const v1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(vInput);
    const vOut = tf.layers.dense({ units: 1 }).apply(v1);
    this.valueBranch = tf.model({ inputs: vInput, outputs: vOut as tf.SymbolicTensor });

    // Price advantage branch: A_price(s, .) -> [priceLevels]
    const pInput = tf.input({ shape: [hiddenSize] });
    const p1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(pInput);
    const pOut = tf.layers.dense({ units: priceLevels }).apply(p1);
    this.priceBranch = tf.model({ inputs: pInput, outputs: pOut as tf.SymbolicTensor });

    // Quantity advantage branch: A_qty(s, .) -> [quantityLevels]
    const qInput = tf.input({ shape: [hiddenSize] });
    const q1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(qInput);
    const qOut = tf.layers.dense({ units: quantityLevels }).apply(q1);
    this.quantityBranch = tf.model({ inputs: qInput, outputs: qOut as tf.SymbolicTensor });
  }

  /**
   * Forward pass: compute Q-values for both action dimensions.
   * Returns { priceQ, quantityQ } each of shape [batch, levels].
   */
  predict(embedding: tf.Tensor2D): { priceQ: tf.Tensor2D; quantityQ: tf.Tensor2D } {
    const shared = this.sharedNet.predict(embedding) as tf.Tensor2D;
    const value = this.valueBranch.predict(shared) as tf.Tensor2D; // [batch, 1]

    const priceAdv = this.priceBranch.predict(shared) as tf.Tensor2D;
    const qtyAdv = this.quantityBranch.predict(shared) as tf.Tensor2D;

    // Dueling: Q = V + (A - mean(A))
    const priceQ = this.duelingCombine(value, priceAdv);
    const quantityQ = this.duelingCombine(value, qtyAdv);

    shared.dispose();
    value.dispose();
    priceAdv.dispose();
    qtyAdv.dispose();

    return { priceQ, quantityQ };
  }

  /** Select greedy actions: argmax over each branch's Q-values. */
  selectAction(embedding: tf.Tensor2D): { priceIndex: number; quantityIndex: number } {
    const { priceQ, quantityQ } = this.predict(embedding);

    const priceIndex = priceQ.argMax(1).dataSync()[0];
    const quantityIndex = quantityQ.argMax(1).dataSync()[0];

    priceQ.dispose();
    quantityQ.dispose();

    return { priceIndex, quantityIndex };
  }

  /** Copy weights from another BDQ network (for target network updates). */
  copyWeightsFrom(source: BranchingDuelingQNetwork): void {
    this.sharedNet.setWeights(source.sharedNet.getWeights());
    this.valueBranch.setWeights(source.valueBranch.getWeights());
    this.priceBranch.setWeights(source.priceBranch.getWeights());
    this.quantityBranch.setWeights(source.quantityBranch.getWeights());
  }

  /** Get all trainable weights across all sub-networks. */
  getTrainableWeights(): tf.Variable[] {
    return [
      ...this.sharedNet.trainableWeights,
      ...this.valueBranch.trainableWeights,
      ...this.priceBranch.trainableWeights,
      ...this.quantityBranch.trainableWeights,
    ].map(w => w.read() as unknown as tf.Variable);
  }

  /** Get all sub-models for optimizer assignment. */
  get models(): tf.LayersModel[] {
    return [this.sharedNet, this.valueBranch, this.priceBranch, this.quantityBranch];
  }

  /** Dueling aggregation: V(s) + (A(s,a) - mean(A(s,.))) */
  private duelingCombine(value: tf.Tensor2D, advantage: tf.Tensor2D): tf.Tensor2D {
    const advMean = advantage.mean(1, true); // [batch, 1]
    const centered = advantage.sub(advMean);
    const q = value.add(centered) as tf.Tensor2D;
    advMean.dispose();
    centered.dispose();
    return q;
  }
}

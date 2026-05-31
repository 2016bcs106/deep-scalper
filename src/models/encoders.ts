import tf from '../tf.js';

/**
 * Macro-level encoder: MLP that processes OHLCV + technical indicators.
 * Input: [batchSize, macroFeatures] (16 values: 11 indicators + 5 OHLCV)
 * Output: [batchSize, embeddingSize]
 */
export class MacroEncoder {
  private model: tf.LayersModel;
  readonly inputSize: number;
  readonly embeddingSize: number;

  constructor(inputSize = 16, embeddingSize = 64, hiddenSize = 128) {
    this.inputSize = inputSize;
    this.embeddingSize = embeddingSize;

    const input = tf.input({ shape: [inputSize] });
    const h1 = tf.layers.dense({ units: hiddenSize, activation: 'relu' }).apply(input);
    const h2 = tf.layers.dense({ units: embeddingSize, activation: 'relu' }).apply(h1);

    this.model = tf.model({ inputs: input, outputs: h2 as tf.SymbolicTensor });
  }

  /** Forward pass: macro features -> embedding vector. */
  encode(input: tf.Tensor2D): tf.Tensor2D {
    return this.model.predict(input) as tf.Tensor2D;
  }

  getWeights(): tf.Variable[] {
    return this.model.trainableWeights.map(w => w.read() as unknown as tf.Variable);
  }

  get layers(): tf.layers.Layer[] {
    return this.model.layers;
  }
}

/**
 * Micro-level encoder: two LSTMs processing LOB sequence and private state sequence.
 * Input LOB: [batchSize, seqLength, lobFeatures] (e.g., 20-dim per timestep)
 * Input Private: [batchSize, seqLength, privateFeatures] (3-dim per timestep)
 * Output: [batchSize, embeddingSize] (concatenated final hidden states)
 */
export class MicroEncoder {
  private lobLSTM: tf.LayersModel;
  private privateLSTM: tf.LayersModel;
  readonly lobFeatures: number;
  readonly privateFeatures: number;
  readonly seqLength: number;
  readonly embeddingSize: number;

  constructor(
    seqLength = 10,
    lobFeatures = 20,
    privateFeatures = 3,
    hiddenSize = 32,
  ) {
    this.lobFeatures = lobFeatures;
    this.privateFeatures = privateFeatures;
    this.seqLength = seqLength;
    this.embeddingSize = hiddenSize * 2; // concat of both LSTM outputs

    // LOB LSTM
    const lobInput = tf.input({ shape: [seqLength, lobFeatures] });
    const lobOut = tf.layers.lstm({ units: hiddenSize }).apply(lobInput);
    this.lobLSTM = tf.model({ inputs: lobInput, outputs: lobOut as tf.SymbolicTensor });

    // Private state LSTM
    const privInput = tf.input({ shape: [seqLength, privateFeatures] });
    const privOut = tf.layers.lstm({ units: hiddenSize }).apply(privInput);
    this.privateLSTM = tf.model({ inputs: privInput, outputs: privOut as tf.SymbolicTensor });
  }

  /** Forward pass: LOB sequence + private state sequence -> micro embedding. */
  encode(lobSeq: tf.Tensor3D, privateSeq: tf.Tensor3D): tf.Tensor2D {
    const lobEmb = this.lobLSTM.predict(lobSeq) as tf.Tensor2D;
    const privEmb = this.privateLSTM.predict(privateSeq) as tf.Tensor2D;
    return tf.concat([lobEmb, privEmb], 1) as tf.Tensor2D;
  }

  get layers(): tf.layers.Layer[] {
    return [...this.lobLSTM.layers, ...this.privateLSTM.layers];
  }
}

/**
 * Combined market embedding: concatenates macro + micro embeddings.
 * This is the shared representation fed to the BDQ and auxiliary task.
 */
export class MarketEmbedding {
  readonly macroEncoder: MacroEncoder;
  readonly microEncoder: MicroEncoder;
  readonly totalEmbeddingSize: number;

  constructor(macroEncoder: MacroEncoder, microEncoder: MicroEncoder) {
    this.macroEncoder = macroEncoder;
    this.microEncoder = microEncoder;
    this.totalEmbeddingSize = macroEncoder.embeddingSize + microEncoder.embeddingSize;
  }

  /** Produce full market embedding from all input modalities. */
  encode(macro: tf.Tensor2D, lobSeq: tf.Tensor3D, privateSeq: tf.Tensor3D): tf.Tensor2D {
    const macroEmb = this.macroEncoder.encode(macro);
    const microEmb = this.microEncoder.encode(lobSeq, privateSeq);
    return tf.concat([macroEmb, microEmb], 1) as tf.Tensor2D;
  }
}

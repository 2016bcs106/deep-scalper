import tf from '../tf.js';
import { OHLCVBar } from '../data/features.js';
import { TradingEnvironment, TradingEnvConfig, Action, StepResult } from '../env/trading-env.js';
import { MacroEncoder, MicroEncoder, MarketEmbedding } from '../models/encoders.js';
import { BranchingDuelingQNetwork } from '../models/bdq.js';
import { VolatilityPredictor } from '../models/auxiliary.js';
import { ModelStore, ModelMetadata } from '../models/model-store.js';
import { PrioritizedReplayBuffer, Transition } from './replay-buffer.js';
import { RewardCalculator, RewardConfig } from './rewards.js';

/** Training hyperparameters. */
export interface TrainerConfig {
  /** Learning rate for Adam optimizer. */
  learningRate: number;
  /** Discount factor for future rewards. */
  gamma: number;
  /** Batch size for replay sampling. */
  batchSize: number;
  /** How often to sync target network (in steps). */
  targetUpdateFreq: number;
  /** Perform a gradient update every N environment steps. */
  trainEveryNSteps: number;
  /** Starting exploration rate. */
  epsilonStart: number;
  /** Final exploration rate. */
  epsilonEnd: number;
  /** Steps over which epsilon decays. */
  epsilonDecaySteps: number;
  /** Weight of volatility auxiliary loss (alpha in paper). */
  auxiliaryWeight: number;
  /** Replay buffer capacity. */
  bufferCapacity: number;
  /** Minimum buffer size before training starts. */
  minBufferSize: number;
  /** Max gradient norm for clipping (prevents exploding gradients). */
  gradientClipNorm: number;
  /** Reward function config. */
  reward: Partial<RewardConfig>;
  /** Trading environment config. */
  env: Partial<TradingEnvConfig>;
}

const DEFAULT_CONFIG: TrainerConfig = {
  learningRate: 0.001,
  gamma: 0.99,
  batchSize: 64,
  targetUpdateFreq: 500,
  trainEveryNSteps: 10,
  epsilonStart: 1.0,
  epsilonEnd: 0.01,
  epsilonDecaySteps: 10000,
  auxiliaryWeight: 1.0,
  bufferCapacity: 100000,
  minBufferSize: 1000,
  gradientClipNorm: 1.0,
  reward: {},
  env: {},
};

/** Per-episode training statistics. */
export interface EpisodeStats {
  episode: number;
  totalReward: number;
  netValue: number;
  steps: number;
  epsilon: number;
  avgLoss: number;
}

/**
 * Orchestrates the full DeepScalper training loop.
 *
 * Manages: environment interaction, epsilon-greedy exploration,
 * experience replay, Q-network updates with target network,
 * hindsight reward relabeling, and volatility auxiliary loss.
 */
export class Trainer {
  private config: TrainerConfig;
  private env: TradingEnvironment;
  private marketEmbedding: MarketEmbedding;
  private qNetwork: BranchingDuelingQNetwork;
  private targetNetwork: BranchingDuelingQNetwork;
  private volPredictor: VolatilityPredictor;
  private buffer: PrioritizedReplayBuffer;
  private rewardCalc: RewardCalculator;
  private optimizer: tf.Optimizer;
  private totalSteps = 0;
  private epsilon: number;

  constructor(config: Partial<TrainerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.epsilon = this.config.epsilonStart;

    // Environment
    this.env = new TradingEnvironment(this.config.env);

    // Models
    // Observation vector: 16 macro (11 indicators + 5 OHLCV) + 3 private = 19
    const obsSize = 19;

    const macroEncoder = new MacroEncoder(16, 64);
    const microEncoder = new MicroEncoder(10, 20, 3, 32);
    this.marketEmbedding = new MarketEmbedding(macroEncoder, microEncoder);

    // BDQ takes raw observation directly (full encoder pipeline used when LOB data available)
    this.qNetwork = new BranchingDuelingQNetwork(obsSize);
    this.targetNetwork = new BranchingDuelingQNetwork(obsSize);
    this.targetNetwork.copyWeightsFrom(this.qNetwork);

    this.volPredictor = new VolatilityPredictor(obsSize);

    // Training infrastructure
    this.buffer = new PrioritizedReplayBuffer(this.config.bufferCapacity);
    this.rewardCalc = new RewardCalculator(this.config.reward);
    this.optimizer = tf.train.adam(this.config.learningRate);
  }

  /**
   * Train for one episode (one trading day).
   * Returns episode statistics.
   */
  trainEpisode(dayBars: OHLCVBar[]): EpisodeStats {
    const obs = this.env.reset(dayBars);
    const prices = dayBars.map(b => b.close);
    const volatilities = VolatilityPredictor.computeVolatility(prices);

    const episodeTransitions: { state: number[]; action: Action; nextState: number[]; position: number; tradeAmount: number; done: boolean }[] = [];
    let totalReward = 0;
    let totalLoss = 0;
    let lossCount = 0;

    let currentObs = obs;
    let step = 0;

    while (!this.env.isDone) {
      const stateVec = this.obsToStateVector(currentObs);
      const action = this.selectAction(stateVec);

      const result: StepResult = this.env.step(action);
      const nextStateVec = this.obsToStateVector(result.observation);

      episodeTransitions.push({
        state: stateVec,
        action,
        nextState: nextStateVec,
        position: result.info.position,
        tradeAmount: 0, // simplified: fee handled by env
        done: result.done,
      });

      // Compute enhanced reward with hindsight
      const reward = this.rewardCalc.enhancedReward(prices, step, result.info.position, 0);
      totalReward += reward;

      this.buffer.push({
        state: stateVec,
        action,
        reward,
        nextState: nextStateVec,
        done: result.done,
      });

      // Train on a batch every N steps if buffer is ready
      this.totalSteps++;
      if (this.buffer.size >= this.config.minBufferSize && this.totalSteps % this.config.trainEveryNSteps === 0) {
        const loss = this.trainStep(volatilities, step);
        totalLoss += loss;
        lossCount++;
      }

      // Update target network periodically
      if (this.totalSteps % this.config.targetUpdateFreq === 0) {
        this.targetNetwork.copyWeightsFrom(this.qNetwork);
      }

      this.decayEpsilon();
      currentObs = result.observation;
      step++;
    }

    return {
      episode: 0,
      totalReward,
      netValue: this.env.getNetValue(),
      steps: step,
      epsilon: this.epsilon,
      avgLoss: lossCount > 0 ? totalLoss / lossCount : 0,
    };
  }

  /**
   * Run one episode with greedy policy (no exploration, no training).
   * Used for validation to detect overfitting.
   */
  validateEpisode(dayBars: OHLCVBar[]): { netValue: number; trades: number } {
    const obs = this.env.reset(dayBars);
    let currentObs = obs;
    let trades = 0;

    while (!this.env.isDone) {
      const stateVec = this.obsToStateVector(currentObs);
      const action = tf.tidy(() => {
        const stateTensor = tf.tensor2d([stateVec]);
        return this.qNetwork.selectAction(stateTensor);
      });

      if (action.quantityIndex !== Math.floor(this.qNetwork.quantityLevels / 2)) {
        trades++;
      }

      const result = this.env.step(action);
      currentObs = result.observation;
    }

    return { netValue: this.env.getNetValue(), trades };
  }

  /** Current exploration rate. */
  getEpsilon(): number {
    return this.epsilon;
  }

  /** Epsilon-greedy action selection. */
  private selectAction(stateVec: number[]): Action {
    if (Math.random() < this.epsilon) {
      return {
        priceIndex: Math.floor(Math.random() * this.qNetwork.priceLevels),
        quantityIndex: Math.floor(Math.random() * this.qNetwork.quantityLevels),
      };
    }

    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([stateVec]);
      return this.qNetwork.selectAction(stateTensor);
    });
  }

  /** Single gradient update step on a sampled batch (vectorized, wrapped in tidy). */
  private trainStep(volatilities: number[], currentTimeStep: number): number {
    const { transitions, indices, weights } = this.buffer.sample(this.config.batchSize);
    const batchSize = this.config.batchSize;

    const statesData = transitions.map(t => t.state);
    const nextStatesData = transitions.map(t => t.nextState);
    const rewardsArr = transitions.map(t => t.reward);
    const donesArr = transitions.map(t => t.done ? 0 : 1);
    const priceActions = transitions.map(t => t.action.priceIndex);
    const qtyActions = transitions.map(t => t.action.quantityIndex);

    // Compute TD targets outside tidy (we need the JS arrays)
    const targetValues = tf.tidy(() => {
      const nextStates = tf.tensor2d(nextStatesData);
      const { priceQ: targetPriceQ, quantityQ: targetQtyQ } = this.targetNetwork.predict(nextStates);
      const maxP = targetPriceQ.max(1).dataSync();
      const maxQ = targetQtyQ.max(1).dataSync();
      return { maxP, maxQ };
    });

    const targets: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      targets.push(rewardsArr[i] + this.config.gamma * donesArr[i] * (targetValues.maxP[i] + targetValues.maxQ[i]) / 2);
    }

    // Gradient update wrapped in tidy to prevent all leaks
    let loss = 0;
    const tdErrors: number[] = new Array(batchSize);

    const states = tf.tensor2d(statesData);
    const qLoss = this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const { priceQ, quantityQ } = this.qNetwork.predict(states);

        const priceMask = tf.oneHot(priceActions, this.qNetwork.priceLevels);
        const qtyMask = tf.oneHot(qtyActions, this.qNetwork.quantityLevels);

        const selectedPriceQ = priceQ.mul(priceMask).sum(1);
        const selectedQtyQ = quantityQ.mul(qtyMask).sum(1);
        const currentQ = selectedPriceQ.add(selectedQtyQ).div(2);

        const targetsTensor = tf.tensor1d(targets);
        const weightsTensor = tf.tensor1d(weights);
        const tdTensor = targetsTensor.sub(currentQ);
        const weightedLoss = tdTensor.square().mul(weightsTensor).mean();

        // Store TD errors for priority update
        const tdData = tdTensor.abs().dataSync();
        for (let i = 0; i < batchSize; i++) tdErrors[i] = tdData[i];

        // Auxiliary volatility loss
        const volIdx = Math.min(currentTimeStep, volatilities.length - 1);
        const actualVol = tf.tensor2d([[volatilities[volIdx]]]).tile([batchSize, 1]);
        const volLoss = this.volPredictor.loss(states, actualVol);

        const combined = weightedLoss.add(volLoss.mul(this.config.auxiliaryWeight));
        loss = combined.dataSync()[0];

        return combined as tf.Scalar;
      });
    }, true);

    this.buffer.updatePriorities(indices, tdErrors);
    states.dispose();
    if (qLoss) qLoss.dispose();

    return loss;
  }

  private decayEpsilon(): void {
    const { epsilonStart, epsilonEnd, epsilonDecaySteps } = this.config;
    this.epsilon = Math.max(
      epsilonEnd,
      epsilonStart - (epsilonStart - epsilonEnd) * (this.totalSteps / epsilonDecaySteps),
    );
  }

  /** Save trained model to disk for a given symbol. */
  async saveModel(symbol: string, epochs: number, episodes: number): Promise<string> {
    const store = new ModelStore();
    const metadata: ModelMetadata = {
      symbol,
      trainedAt: new Date().toISOString(),
      epochs,
      episodes,
      finalNetValue: this.env.getNetValue(),
      config: this.config,
    };

    const subModels = this.qNetwork.getSubModels();
    const dir = await store.save(symbol, {
      ...subModels,
      volatility: this.volPredictor.getModel(),
    }, metadata);

    return dir;
  }

  /** Load a previously saved model for a given symbol. */
  async loadModel(symbol: string): Promise<ModelMetadata> {
    const store = new ModelStore();
    const { shared, value, price, quantity, volatility, metadata } = await store.load(symbol);

    this.qNetwork.loadFrom({ shared, value, price, quantity });
    this.targetNetwork.copyWeightsFrom(this.qNetwork);
    this.volPredictor.loadFrom(volatility);

    return metadata;
  }

  /** Get the Q-network for inference (e.g., evaluation without training). */
  getQNetwork(): BranchingDuelingQNetwork {
    return this.qNetwork;
  }

  /** Flatten observation into a fixed-size vector for the Q-network. */
  private obsToStateVector(obs: { macro: number[]; private: number[] }): number[] {
    return [...obs.macro, ...obs.private];
  }
}

import * as tf from '@tensorflow/tfjs-node';
import { OHLCVBar } from '../data/features.js';
import { TradingEnvironment, TradingEnvConfig, Action, StepResult } from '../env/trading-env.js';
import { MacroEncoder, MicroEncoder, MarketEmbedding } from '../models/encoders.js';
import { BranchingDuelingQNetwork } from '../models/bdq.js';
import { VolatilityPredictor } from '../models/auxiliary.js';
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
  epsilonStart: 1.0,
  epsilonEnd: 0.01,
  epsilonDecaySteps: 10000,
  auxiliaryWeight: 1.0,
  bufferCapacity: 100000,
  minBufferSize: 1000,
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

      // Train on a batch if buffer is ready
      if (this.buffer.size >= this.config.minBufferSize) {
        const loss = this.trainStep(volatilities, step);
        totalLoss += loss;
        lossCount++;
      }

      // Update target network periodically
      this.totalSteps++;
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

    const stateTensor = tf.tensor2d([stateVec]);
    const action = this.qNetwork.selectAction(stateTensor);
    stateTensor.dispose();
    return action;
  }

  /** Single gradient update step on a sampled batch. */
  private trainStep(volatilities: number[], currentTimeStep: number): number {
    const { transitions, indices, weights } = this.buffer.sample(this.config.batchSize);

    const states = tf.tensor2d(transitions.map(t => t.state));
    const nextStates = tf.tensor2d(transitions.map(t => t.nextState));
    const rewards = transitions.map(t => t.reward);
    const dones = transitions.map(t => t.done ? 0 : 1);
    const priceActions = transitions.map(t => t.action.priceIndex);
    const qtyActions = transitions.map(t => t.action.quantityIndex);

    // Compute TD targets using target network
    const { priceQ: targetPriceQ, quantityQ: targetQtyQ } = this.targetNetwork.predict(nextStates);
    const maxTargetPrice = targetPriceQ.max(1).dataSync();
    const maxTargetQty = targetQtyQ.max(1).dataSync();

    const tdErrors: number[] = [];
    let loss = 0;

    // Q-network update
    const qLoss = this.optimizer.minimize(() => {
      const { priceQ, quantityQ } = this.qNetwork.predict(states);

      let totalLoss = tf.scalar(0);

      for (let i = 0; i < this.config.batchSize; i++) {
        const target = rewards[i] + this.config.gamma * dones[i] *
          (maxTargetPrice[i] + maxTargetQty[i]) / 2;

        const pQ = priceQ.slice([i, priceActions[i]], [1, 1]).reshape([]);
        const qQ = quantityQ.slice([i, qtyActions[i]], [1, 1]).reshape([]);
        const currentQ = pQ.add(qQ).div(2);

        const td = tf.scalar(target).sub(currentQ);
        const weightedTd = td.square().mul(weights[i]);
        totalLoss = totalLoss.add(weightedTd) as tf.Scalar;

        tdErrors.push(Math.abs(target - currentQ.dataSync()[0]));

        pQ.dispose();
        qQ.dispose();
        currentQ.dispose();
        td.dispose();
        weightedTd.dispose();
      }

      // Auxiliary volatility loss
      const volIdx = Math.min(currentTimeStep, volatilities.length - 1);
      const actualVol = tf.tensor2d([[volatilities[volIdx]]]).tile([this.config.batchSize, 1]);
      const volLoss = this.volPredictor.loss(states, actualVol);
      actualVol.dispose();

      const combined = totalLoss.div(this.config.batchSize).add(volLoss.mul(this.config.auxiliaryWeight));
      loss = combined.dataSync()[0];

      priceQ.dispose();
      quantityQ.dispose();
      volLoss.dispose();
      totalLoss.dispose();

      return combined as tf.Scalar;
    }, true);

    // Update priorities
    this.buffer.updatePriorities(indices, tdErrors);

    // Cleanup
    states.dispose();
    nextStates.dispose();
    targetPriceQ.dispose();
    targetQtyQ.dispose();
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

  /** Flatten observation into a fixed-size vector for the Q-network. */
  private obsToStateVector(obs: { macro: number[]; private: number[] }): number[] {
    return [...obs.macro, ...obs.private];
  }
}

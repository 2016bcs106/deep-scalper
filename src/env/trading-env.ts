import { OHLCVBar, FeatureCalculator, FeatureVector } from '../data/features.js';

/** Configuration for the trading environment. */
export interface TradingEnvConfig {
  /** Maximum shares the agent can hold (long or short). */
  maxPosition: number;
  /** Transaction fee rate per trade (e.g., 0.0003 = 0.03%). */
  feeRate: number;
  /** Number of discrete price levels in action space. */
  priceLevels: number;
  /** Number of discrete quantity levels in action space. */
  quantityLevels: number;
  /** Initial cash allocated to the agent. */
  initialCash: number;
}

/** The 2D action output: which price level and quantity level to use. */
export interface Action {
  priceIndex: number;
  quantityIndex: number;
}

/** Observation returned to the agent at each step. */
export interface Observation {
  /** Macro-level: 11 technical indicators + 5 OHLCV values. */
  macro: number[];
  /** Private state: [position_ratio, cash_ratio, time_remaining_ratio]. */
  private: number[];
}

/** Result of taking one step in the environment. */
export interface StepResult {
  observation: Observation;
  reward: number;
  done: boolean;
  info: {
    netValue: number;
    position: number;
    cash: number;
    closePrice: number;
  };
}

const DEFAULT_CONFIG: TradingEnvConfig = {
  maxPosition: 50,
  feeRate: 0.0003,
  priceLevels: 5,
  quantityLevels: 5,
  initialCash: 100000,
};

/**
 * Intraday trading environment implementing the DeepScalper MDP.
 *
 * One episode = one trading day. The agent observes market state,
 * submits a 2D action (price level, quantity), receives P&L reward,
 * and all positions are closed at end-of-day.
 */
export class TradingEnvironment {
  private config: TradingEnvConfig;
  private bars: OHLCVBar[] = [];
  private featureCalc: FeatureCalculator;
  private currentStep = 0;
  private position = 0;
  private cash: number;
  private done = false;

  constructor(config: Partial<TradingEnvConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.cash = this.config.initialCash;
    this.featureCalc = new FeatureCalculator();
  }

  /** Load a single day's bars and prepare for a new episode. */
  reset(dayBars: OHLCVBar[]): Observation {
    if (dayBars.length < 2) {
      throw new Error('Need at least 2 bars for an episode');
    }

    this.bars = dayBars;
    this.featureCalc.reset();
    this.featureCalc.addBars(dayBars);
    this.currentStep = 0;
    this.position = 0;
    this.cash = this.config.initialCash;
    this.done = false;

    return this.getObservation();
  }

  /** Execute one action and advance the environment by one time step. */
  step(action: Action): StepResult {
    if (this.done) {
      throw new Error('Episode is done. Call reset() with new day data.');
    }

    const prevClose = this.bars[this.currentStep].close;
    const tradeQty = this.decodeAction(action);

    // Execute trade
    if (tradeQty !== 0) {
      const tradeCost = Math.abs(tradeQty) * prevClose * this.config.feeRate;
      this.cash -= tradeQty * prevClose + tradeCost;
      this.position += tradeQty;
    }

    this.currentStep++;

    const isLastStep = this.currentStep >= this.bars.length - 1;

    // Force-close position at end-of-day
    if (isLastStep) {
      this.closePosition();
      this.done = true;
    }

    // P&L reward: price change * position held before this step's price move
    const currClose = this.bars[this.currentStep].close;
    const reward = (currClose - prevClose) * (this.position);

    return {
      observation: this.getObservation(),
      reward,
      done: this.done,
      info: {
        netValue: this.getNetValue(),
        position: this.position,
        cash: this.cash,
        closePrice: currClose,
      },
    };
  }

  /** Current net value: (cash + position * price) / initial_cash. */
  getNetValue(): number {
    const price = this.bars[this.currentStep].close;
    return (this.cash + this.position * price) / this.config.initialCash;
  }

  /** Total number of steps in this episode. */
  get totalSteps(): number {
    return this.bars.length - 1;
  }

  /** Whether the episode has ended. */
  get isDone(): boolean {
    return this.done;
  }

  /**
   * Decode a 2D action into a trade quantity.
   * Positive = buy, negative = sell, 0 = hold.
   */
  private decodeAction(action: Action): number {
    const { priceLevels, quantityLevels, maxPosition } = this.config;

    // Clamp indices
    const qIdx = Math.max(0, Math.min(quantityLevels - 1, action.quantityIndex));

    // Quantity: map index to proportion [-1, +1] of max position
    // e.g., 5 levels -> [-1, -0.5, 0, +0.5, +1]
    const proportion = (2 * qIdx / (quantityLevels - 1)) - 1;
    const targetPosition = Math.round(proportion * maxPosition);

    // Trade delta: difference between target and current position
    let delta = targetPosition - this.position;

    // Clamp to position limits
    const newPos = this.position + delta;
    if (Math.abs(newPos) > maxPosition) {
      delta = (newPos > 0 ? maxPosition : -maxPosition) - this.position;
    }

    return delta;
  }

  /** Flatten position at current market price (end-of-day). */
  private closePosition(): void {
    if (this.position === 0) return;

    const price = this.bars[this.currentStep].close;
    const tradeCost = Math.abs(this.position) * price * this.config.feeRate;
    this.cash += this.position * price - tradeCost;
    this.position = 0;
  }

  private getObservation(): Observation {
    const features: FeatureVector = this.featureCalc.computeAt(this.currentStep);
    const bar = this.bars[this.currentStep];

    const macro = [
      ...FeatureCalculator.toArray(features),
      bar.open, bar.high, bar.low, bar.close, bar.volume,
    ];

    const priv = [
      this.position / this.config.maxPosition,
      this.cash / this.config.initialCash,
      (this.bars.length - 1 - this.currentStep) / (this.bars.length - 1),
    ];

    return { macro, private: priv };
  }
}

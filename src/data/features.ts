/**
 * Technical indicator calculations from the DeepScalper paper (Table 2).
 * All 11 indicators are computed from raw OHLCV data.
 */

export interface OHLCVBar {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FeatureVector {
  zopen: number;
  zhigh: number;
  zlow: number;
  zclose: number;
  zadjcp: number;
  zd_5: number;
  zd_10: number;
  zd_15: number;
  zd_20: number;
  zd_25: number;
  zd_30: number;
}

export class FeatureCalculator {
  private bars: OHLCVBar[] = [];

  addBar(bar: OHLCVBar): void {
    this.bars.push(bar);
  }

  addBars(bars: OHLCVBar[]): void {
    this.bars.push(...bars);
  }

  reset(): void {
    this.bars = [];
  }

  get length(): number {
    return this.bars.length;
  }

  computeAt(index: number): FeatureVector {
    if (index < 0 || index >= this.bars.length) {
      throw new RangeError(`Index ${index} out of bounds [0, ${this.bars.length - 1}]`);
    }

    const bar = this.bars[index];
    const prevBar = index > 0 ? this.bars[index - 1] : bar;

    const zopen = bar.close !== 0 ? bar.open / bar.close - 1 : 0;
    const zhigh = bar.close !== 0 ? bar.high / bar.close - 1 : 0;
    const zlow = bar.close !== 0 ? bar.low / bar.close - 1 : 0;
    const zclose = prevBar.close !== 0 ? bar.close / prevBar.close - 1 : 0;
    const zadjcp = prevBar.close !== 0 ? bar.close / prevBar.close - 1 : 0;

    const zd_5 = this.movingAverageRelative(index, 5);
    const zd_10 = this.movingAverageRelative(index, 10);
    const zd_15 = this.movingAverageRelative(index, 15);
    const zd_20 = this.movingAverageRelative(index, 20);
    const zd_25 = this.movingAverageRelative(index, 25);
    const zd_30 = this.movingAverageRelative(index, 30);

    return { zopen, zhigh, zlow, zclose, zadjcp, zd_5, zd_10, zd_15, zd_20, zd_25, zd_30 };
  }

  computeLatest(): FeatureVector {
    return this.computeAt(this.bars.length - 1);
  }

  computeAll(): FeatureVector[] {
    return this.bars.map((_, i) => this.computeAt(i));
  }

  static toArray(f: FeatureVector): number[] {
    return [f.zopen, f.zhigh, f.zlow, f.zclose, f.zadjcp,
            f.zd_5, f.zd_10, f.zd_15, f.zd_20, f.zd_25, f.zd_30];
  }

  private movingAverageRelative(index: number, window: number): number {
    const close = this.bars[index].close;
    if (close === 0) return 0;

    const start = Math.max(0, index - window + 1);
    const slice = this.bars.slice(start, index + 1);
    const avg = slice.reduce((sum, b) => sum + b.close, 0) / slice.length;

    return avg / close - 1;
  }
}

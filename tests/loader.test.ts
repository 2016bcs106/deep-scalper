import { describe, it, expect } from 'vitest';
import { DataLoader } from '../src/data/loader.js';

const SAMPLE_CSV = `timestamp,open,high,low,close,volume
2024-01-15T09:15:00,100,105,98,103,5000
2024-01-15T09:16:00,103,106,102,105,3000
2024-01-15T09:17:00,105,107,104,106,4000
2024-01-16T09:15:00,106,108,105,107,6000
2024-01-16T09:16:00,107,109,106,108,2000`;

describe('DataLoader', () => {
  it('should parse CSV content into OHLCV bars', () => {
    const loader = new DataLoader();
    const bars = loader.parseCSVContent(SAMPLE_CSV);

    expect(bars).toHaveLength(5);
    expect(bars[0].open).toBe(100);
    expect(bars[0].close).toBe(103);
    expect(bars[0].volume).toBe(5000);
  });

  it('should parse timestamps as epoch milliseconds', () => {
    const loader = new DataLoader();
    const bars = loader.parseCSVContent(SAMPLE_CSV);

    expect(bars[0].timestamp).toBe(new Date('2024-01-15T09:15:00').getTime());
  });

  it('should handle unix timestamp format', () => {
    const unixCSV = `ts,open,high,low,close,volume\n1705305300000,100,105,98,103,5000`;
    const loader = new DataLoader({ timestampColumn: 'ts', dateFormat: 'unix' });
    const bars = loader.parseCSVContent(unixCSV);

    expect(bars[0].timestamp).toBe(1705305300000);
    expect(bars[0].close).toBe(103);
  });

  it('should handle custom column names', () => {
    const customCSV = `time,o,h,l,c,vol\n2024-01-15T09:15:00,100,105,98,103,5000`;
    const loader = new DataLoader({
      timestampColumn: 'time',
      openColumn: 'o',
      highColumn: 'h',
      lowColumn: 'l',
      closeColumn: 'c',
      volumeColumn: 'vol',
    });
    const bars = loader.parseCSVContent(customCSV);

    expect(bars[0].open).toBe(100);
    expect(bars[0].high).toBe(105);
  });

  it('should split bars by calendar day', () => {
    const loader = new DataLoader();
    const bars = loader.parseCSVContent(SAMPLE_CSV);
    const days = loader.splitByDay(bars);

    expect(days).toHaveLength(2);
    expect(days[0]).toHaveLength(3); // Jan 15
    expect(days[1]).toHaveLength(2); // Jan 16
  });

  it('should split into train/test chronologically', () => {
    const loader = new DataLoader();
    const bars = loader.parseCSVContent(SAMPLE_CSV);
    const { train, test } = loader.trainTestSplit(bars, 0.6);

    expect(train).toHaveLength(3);
    expect(test).toHaveLength(2);
    // No lookahead: last train bar is before first test bar
    expect(train[train.length - 1].timestamp).toBeLessThan(test[0].timestamp);
  });
});

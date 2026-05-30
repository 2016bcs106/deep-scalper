import { readFileSync } from 'fs';
import { parse } from 'csv-parse/sync';
import { OHLCVBar } from './features.js';

/** Column name mappings and format for parsing CSV market data. */
export interface LoaderOptions {
  timestampColumn?: string;
  openColumn?: string;
  highColumn?: string;
  lowColumn?: string;
  closeColumn?: string;
  volumeColumn?: string;
  /** 'unix' for epoch ms/s, 'iso' for ISO-8601 strings. */
  dateFormat?: 'unix' | 'iso';
}

const DEFAULT_OPTIONS: Required<LoaderOptions> = {
  timestampColumn: 'timestamp',
  openColumn: 'open',
  highColumn: 'high',
  lowColumn: 'low',
  closeColumn: 'close',
  volumeColumn: 'volume',
  dateFormat: 'iso',
};

/** Loads and pre-processes OHLCV market data from CSV files. */
export class DataLoader {
  private options: Required<LoaderOptions>;

  constructor(options: LoaderOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /** Read a CSV file from disk and parse into OHLCV bars. */
  loadCSV(filePath: string): OHLCVBar[] {
    const content = readFileSync(filePath, 'utf-8');
    return this.parseCSVContent(content);
  }

  /** Parse raw CSV string content into OHLCV bars. */
  parseCSVContent(content: string): OHLCVBar[] {
    const records = parse(content, {
      columns: true,
      skip_empty_lines: true,
      trim: true,
    }) as Record<string, string>[];

    return records.map(record => this.recordToBar(record));
  }

  /** Group bars by calendar date (for intraday episode boundaries). */
  splitByDay(bars: OHLCVBar[]): OHLCVBar[][] {
    const days = new Map<string, OHLCVBar[]>();

    for (const bar of bars) {
      const date = new Date(bar.timestamp);
      const key = `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;

      if (!days.has(key)) {
        days.set(key, []);
      }
      days.get(key)!.push(bar);
    }

    return Array.from(days.values());
  }

  /** Chronological train/test split (no lookahead bias). */
  trainTestSplit(bars: OHLCVBar[], trainRatio = 0.8): { train: OHLCVBar[]; test: OHLCVBar[] } {
    const splitIndex = Math.floor(bars.length * trainRatio);
    return {
      train: bars.slice(0, splitIndex),
      test: bars.slice(splitIndex),
    };
  }

  private recordToBar(record: Record<string, string>): OHLCVBar {
    const { timestampColumn, openColumn, highColumn, lowColumn, closeColumn, volumeColumn, dateFormat } = this.options;

    const timestamp = dateFormat === 'unix'
      ? Number(record[timestampColumn])
      : new Date(record[timestampColumn]).getTime();

    return {
      timestamp,
      open: Number(record[openColumn]),
      high: Number(record[highColumn]),
      low: Number(record[lowColumn]),
      close: Number(record[closeColumn]),
      volume: Number(record[volumeColumn]),
    };
  }
}

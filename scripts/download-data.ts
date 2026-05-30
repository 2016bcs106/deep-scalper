/**
 * Download 1-minute OHLCV data from Paytm Money's internal API.
 *
 * Prerequisites:
 * - Paytm Money trading account
 * - Extract SSO token, device ID, user ID, and request ID from
 *   an authenticated browser session (Network tab -> request headers)
 *
 * Usage:
 *   export PAYTM_SSO_TOKEN="..."
 *   export PAYTM_DEVICE_ID="..."
 *   export PAYTM_USER_ID="..."
 *   export PAYTM_REQUEST_ID="..."
 *   npm run download-data -- RELIANCE
 *   npm run download-data -- TCS --from 2024-01-01 --to 2024-06-30
 *
 * Output: data/<SYMBOL>_1m.csv with columns: timestamp, open, high, low, close, volume
 */

import { writeFileSync, mkdirSync } from 'fs';
import { OHLCVBar } from '../src/data/features.js';

interface PaytmMoneyConfig {
  ssoToken: string;
  deviceId: string;
  userId: string;
  requestId: string;
}

interface DownloadOptions {
  /** NSE symbol (e.g., 'RELIANCE', 'TCS', 'INFY'). */
  symbol: string;
  /** Start date in YYYY-MM-DD format. */
  from: string;
  /** End date in YYYY-MM-DD format. */
  to: string;
  /** Candle interval. */
  interval: 'DAY' | 'MINUTE';
  /** Output file path. */
  outputPath: string;
}

/** Downloads historical OHLCV data from Paytm Money. */
class PaytmMoneyDataDownloader {
  private config: PaytmMoneyConfig;
  private readonly baseUrl = 'https://api-eq.paytmmoney.com';

  constructor(config: PaytmMoneyConfig) {
    this.config = config;
  }

  /** Fetch candle data for a symbol and write to CSV. */
  async download(options: DownloadOptions): Promise<OHLCVBar[]> {
    console.log(`[PaytmMoney] Downloading ${options.interval} data for ${options.symbol}`);
    console.log(`  Range: ${options.from} to ${options.to}`);

    const pmlId = await this.resolveSymbol(options.symbol);
    console.log(`  Resolved PML ID: ${pmlId}`);

    const bars = await this.fetchCandles(pmlId, options);
    console.log(`  Received ${bars.length} candles`);

    this.writeCSV(bars, options.outputPath);
    console.log(`  Written to: ${options.outputPath}`);

    return bars;
  }

  /** Search for a symbol and return its internal PML ID. */
  private async resolveSymbol(symbol: string): Promise<string> {
    const url = `${this.baseUrl}/data/v2/suggest?is-advanced-user=false&search-scope=ALL&q=${encodeURIComponent(symbol)}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: this.buildHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Symbol search failed: ${response.status} ${response.statusText}`);
    }

    const json = await response.json() as any;
    const results = json.data?.results;

    if (!results || results.length === 0) {
      throw new Error(`No results found for symbol: ${symbol}`);
    }

    const match = results.find(
      (r: any) => r.exch_symbol === symbol && r.exchange === 'NSE'
    );

    if (!match) {
      throw new Error(`Symbol ${symbol} not found on NSE. Available: ${results.map((r: any) => r.exch_symbol).join(', ')}`);
    }

    return match.id;
  }

  /** Fetch historical price candles for a resolved PML ID. */
  private async fetchCandles(pmlId: string, options: DownloadOptions): Promise<OHLCVBar[]> {
    // Note: Paytm API expects fromDate/toDate in reversed order
    const response = await fetch(`${this.baseUrl}/charts/price/v1/price-charts`, {
      method: 'POST',
      headers: this.buildHeaders(),
      body: JSON.stringify({
        fromDate: options.to,
        toDate: options.from,
        interval: options.interval,
        pmlId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Candle fetch failed: ${response.status} ${response.statusText}`);
    }

    const json = await response.json() as any;

    if (!json.data || !Array.isArray(json.data)) {
      throw new Error(`Unexpected response format: ${JSON.stringify(json).slice(0, 200)}`);
    }

    // API returns: [timestamp, open, high, low, close, volume]
    return json.data.map((item: any[]) => ({
      timestamp: item[0],
      open: item[1],
      high: item[2],
      low: item[3],
      close: item[4],
      volume: item[5],
    }));
  }

  /** Write OHLCV bars to a CSV file, converting timestamps to ISO format. */
  private writeCSV(bars: OHLCVBar[], outputPath: string): void {
    const dir = outputPath.substring(0, outputPath.lastIndexOf('/'));
    if (dir) mkdirSync(dir, { recursive: true });

    const header = 'timestamp,open,high,low,close,volume';
    const lines = bars.map(b => {
      const ts = this.parsePaytmTimestamp(b.timestamp as unknown as string);
      return `${ts},${b.open},${b.high},${b.low},${b.close},${b.volume}`;
    });

    writeFileSync(outputPath, [header, ...lines].join('\n'));
  }

  /** Convert "DD-MM-YYYY HH:mm" to ISO-8601 string. */
  private parsePaytmTimestamp(raw: string): string {
    // "28-05-2025 09:15" -> "2025-05-28T09:15:00"
    const [datePart, timePart] = raw.split(' ');
    const [dd, mm, yyyy] = datePart.split('-');
    return `${yyyy}-${mm}-${dd}T${timePart}:00`;
  }

  /** Common auth headers required by Paytm Money API. */
  private buildHeaders(): Record<string, string> {
    return {
      'accept': 'application/json',
      'content-type': 'application/json',
      'x-pmngx-key': 'paytmmoney',
      'x-request-id': this.config.requestId,
      'x-sso-token': this.config.ssoToken,
      'x-user-agent': JSON.stringify({
        platform: 'web',
        user_id: this.config.userId,
        device_id: this.config.deviceId,
      }),
      'Referer': 'https://www.paytmmoney.com/',
    };
  }
}

function formatDate(date: Date): string {
  return date.toISOString().split('T')[0];
}

function parseArgs(): DownloadOptions {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log('Usage: npm run download-data -- <SYMBOL> [--from YYYY-MM-DD] [--to YYYY-MM-DD]');
    console.log('');
    console.log('Arguments:');
    console.log('  SYMBOL          NSE symbol (required, e.g., RELIANCE, TCS, INFY)');
    console.log('  --from          Start date (default: 1 year ago)');
    console.log('  --to            End date (default: today)');
    process.exit(args.includes('--help') ? 0 : 1);
  }

  const symbol = args[0];
  const now = new Date();
  const oneYearAgo = new Date(now);
  oneYearAgo.setFullYear(now.getFullYear() - 1);

  let from = formatDate(oneYearAgo);
  let to = formatDate(now);

  for (let i = 1; i < args.length; i++) {
    if (args[i] === '--from' && args[i + 1]) from = args[++i];
    if (args[i] === '--to' && args[i + 1]) to = args[++i];
  }

  return {
    symbol,
    from,
    to,
    interval: 'MINUTE',
    outputPath: `data/${symbol}_1m.csv`,
  };
}

async function main() {
  const options = parseArgs();

  const config: PaytmMoneyConfig = {
    ssoToken: process.env.PAYTM_SSO_TOKEN || '',
    deviceId: process.env.PAYTM_DEVICE_ID || '',
    userId: process.env.PAYTM_USER_ID || '',
    requestId: process.env.PAYTM_REQUEST_ID || '',
  };

  const missing = Object.entries(config)
    .filter(([_, v]) => !v)
    .map(([k]) => k);

  if (missing.length > 0) {
    console.error('Error: Missing environment variables:');
    missing.forEach(k => {
      const envVar = 'PAYTM_' + k.replace(/([A-Z])/g, '_$1').toUpperCase();
      console.error(`  ${envVar}`);
    });
    console.error('\nExtract these from an authenticated Paytm Money browser session.');
    process.exit(1);
  }

  const downloader = new PaytmMoneyDataDownloader(config);
  await downloader.download(options);
}

await main();

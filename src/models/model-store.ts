import tf from '../tf.js';
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

/** Metadata saved alongside model weights. */
export interface ModelMetadata {
  symbol: string;
  trainedAt: string;
  epochs: number;
  episodes: number;
  finalNetValue: number;
  config: Record<string, any>;
}

/**
 * Handles saving and loading of all DeepScalper model components.
 *
 * Directory structure:
 *   models/<symbol>/
 *     shared/        - BDQ shared network weights
 *     value/         - BDQ state value branch
 *     price/         - BDQ price advantage branch
 *     quantity/      - BDQ quantity advantage branch
 *     volatility/    - Auxiliary volatility predictor
 *     metadata.json  - Training metadata
 */
export class ModelStore {
  private readonly baseDir: string;

  constructor(baseDir = 'models') {
    this.baseDir = baseDir;
  }

  /** Save all model components for a given symbol. */
  async save(
    symbol: string,
    models: { shared: tf.LayersModel; value: tf.LayersModel; price: tf.LayersModel; quantity: tf.LayersModel; volatility: tf.LayersModel },
    metadata: ModelMetadata,
  ): Promise<string> {
    const dir = this.modelDir(symbol);
    mkdirSync(dir, { recursive: true });

    await Promise.all([
      models.shared.save(`file://${join(dir, 'shared')}`),
      models.value.save(`file://${join(dir, 'value')}`),
      models.price.save(`file://${join(dir, 'price')}`),
      models.quantity.save(`file://${join(dir, 'quantity')}`),
      models.volatility.save(`file://${join(dir, 'volatility')}`),
    ]);

    writeFileSync(join(dir, 'metadata.json'), JSON.stringify(metadata, null, 2));
    return dir;
  }

  /** Load all model components for a given symbol. */
  async load(symbol: string): Promise<{
    shared: tf.LayersModel;
    value: tf.LayersModel;
    price: tf.LayersModel;
    quantity: tf.LayersModel;
    volatility: tf.LayersModel;
    metadata: ModelMetadata;
  }> {
    const dir = this.modelDir(symbol);

    if (!existsSync(dir)) {
      throw new Error(`No saved model found for symbol: ${symbol}. Expected at: ${dir}`);
    }

    const [shared, value, price, quantity, volatility] = await Promise.all([
      tf.loadLayersModel(`file://${join(dir, 'shared', 'model.json')}`),
      tf.loadLayersModel(`file://${join(dir, 'value', 'model.json')}`),
      tf.loadLayersModel(`file://${join(dir, 'price', 'model.json')}`),
      tf.loadLayersModel(`file://${join(dir, 'quantity', 'model.json')}`),
      tf.loadLayersModel(`file://${join(dir, 'volatility', 'model.json')}`),
    ]);

    const metadata = JSON.parse(readFileSync(join(dir, 'metadata.json'), 'utf-8'));

    return { shared, value, price, quantity, volatility, metadata };
  }

  /** Check whether a saved model exists for a symbol. */
  exists(symbol: string): boolean {
    return existsSync(join(this.modelDir(symbol), 'metadata.json'));
  }

  /** List all saved model symbols. */
  list(): string[] {
    const { readdirSync } = require('fs');
    if (!existsSync(this.baseDir)) return [];
    return readdirSync(this.baseDir).filter((name: string) =>
      existsSync(join(this.baseDir, name, 'metadata.json'))
    );
  }

  private modelDir(symbol: string): string {
    return join(this.baseDir, symbol);
  }
}

import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs-node';
import { MacroEncoder, MicroEncoder, MarketEmbedding } from '../src/models/encoders.js';

describe('MacroEncoder', () => {
  it('should produce embedding of correct shape', () => {
    const encoder = new MacroEncoder(16, 64);
    const input = tf.randomNormal([4, 16]); // batch of 4

    const output = encoder.encode(input);

    expect(output.shape).toEqual([4, 64]);
    input.dispose();
    output.dispose();
  });

  it('should work with different embedding sizes', () => {
    const encoder = new MacroEncoder(16, 32, 64);
    const input = tf.randomNormal([1, 16]);

    const output = encoder.encode(input);

    expect(output.shape).toEqual([1, 32]);
    input.dispose();
    output.dispose();
  });
});

describe('MicroEncoder', () => {
  it('should produce embedding of correct shape', () => {
    const encoder = new MicroEncoder(10, 20, 3, 32);
    const lobSeq = tf.randomNormal([4, 10, 20]);
    const privSeq = tf.randomNormal([4, 10, 3]);

    const output = encoder.encode(lobSeq, privSeq);

    // 32 (LOB LSTM) + 32 (private LSTM) = 64
    expect(output.shape).toEqual([4, 64]);
    lobSeq.dispose();
    privSeq.dispose();
    output.dispose();
  });

  it('should report correct embedding size', () => {
    const encoder = new MicroEncoder(10, 20, 3, 32);
    expect(encoder.embeddingSize).toBe(64);
  });
});

describe('MarketEmbedding', () => {
  it('should concatenate macro and micro embeddings', () => {
    const macroEnc = new MacroEncoder(16, 64);
    const microEnc = new MicroEncoder(10, 20, 3, 32);
    const market = new MarketEmbedding(macroEnc, microEnc);

    const macro = tf.randomNormal([2, 16]);
    const lobSeq = tf.randomNormal([2, 10, 20]);
    const privSeq = tf.randomNormal([2, 10, 3]);

    const output = market.encode(macro, lobSeq, privSeq);

    // 64 (macro) + 64 (micro) = 128
    expect(output.shape).toEqual([2, 128]);
    expect(market.totalEmbeddingSize).toBe(128);

    macro.dispose();
    lobSeq.dispose();
    privSeq.dispose();
    output.dispose();
  });
});

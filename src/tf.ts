/**
 * Centralized TensorFlow.js import.
 * Uses tfjs-node (native, fast) when available, falls back to pure JS (portable).
 */
let tf: typeof import('@tensorflow/tfjs-node');

try {
  tf = await import('@tensorflow/tfjs-node');
} catch {
  tf = await import('@tensorflow/tfjs') as any;
}

export default tf;

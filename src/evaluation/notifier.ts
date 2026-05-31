import nodemailer from 'nodemailer';
import { EvaluationResult } from './metrics.js';

/** Thresholds for classifying metric quality. */
interface MetricThreshold {
  good: number;
  bad: number;
}

const THRESHOLDS: Record<string, MetricThreshold> = {
  totalReturn: { good: 0.02, bad: -0.01 },
  sharpeRatio: { good: 1.0, bad: 0 },
  calmarRatio: { good: 0.5, bad: 0 },
  sortinoRatio: { good: 1.5, bad: 0 },
  maxDrawdown: { good: 0.05, bad: 0.15 },
};

/** Email notification config loaded from environment. */
export interface NotifierConfig {
  gmailUser: string;
  gmailAppPassword: string;
}

/**
 * Sends email notifications with color-coded training/evaluation metrics.
 * Uses Gmail SMTP with app passwords.
 *
 * Required env vars:
 *   GMAIL_USER         - your Gmail address
 *   GMAIL_APP_PASSWORD - app password (not your regular password)
 *
 * Generate app password: Google Account > Security > 2FA > App Passwords
 */
export class EmailNotifier {
  private transporter: nodemailer.Transporter;
  private config: NotifierConfig;

  constructor(config: NotifierConfig) {
    this.config = config;
    this.transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
        user: config.gmailUser,
        pass: config.gmailAppPassword,
      },
    });
  }

  /** Load config from environment variables. Throws if missing. */
  static fromEnv(): EmailNotifier {
    const gmailUser = process.env.GMAIL_USER || '';
    const gmailAppPassword = process.env.GMAIL_APP_PASSWORD || '';

    if (!gmailUser || !gmailAppPassword) {
      throw new Error(
        'Missing email config. Set GMAIL_USER and GMAIL_APP_PASSWORD in .env\n' +
        'Generate app password: Google Account > Security > 2FA > App Passwords'
      );
    }

    return new EmailNotifier({ gmailUser, gmailAppPassword });
  }

  /** Send training completion notification with metrics. */
  async sendTrainingReport(opts: {
    symbol: string;
    epochs: number;
    episodes: number;
    trainingNetValue: number;
    evaluation: EvaluationResult;
    duration?: string;
  }): Promise<void> {
    const subject = this.buildSubject(opts.symbol, opts.evaluation);
    const html = this.buildHTML(opts);

    await this.transporter.sendMail({
      from: this.config.gmailUser,
      to: this.config.gmailUser,
      subject,
      html,
    });
  }

  private buildSubject(symbol: string, eval_: EvaluationResult): string {
    const returnPct = (eval_.totalReturn * 100).toFixed(2);
    const emoji = eval_.totalReturn > 0 ? '📈' : eval_.totalReturn < -0.02 ? '📉' : '➡️';
    return `${emoji} DeepScalper ${symbol}: ${returnPct}% return`;
  }

  private buildHTML(opts: {
    symbol: string;
    epochs: number;
    episodes: number;
    trainingNetValue: number;
    evaluation: EvaluationResult;
    duration?: string;
  }): string {
    const { symbol, epochs, episodes, trainingNetValue, evaluation, duration } = opts;

    const metrics = [
      { name: 'Total Return', value: `${(evaluation.totalReturn * 100).toFixed(2)}%`, rating: this.rate('totalReturn', evaluation.totalReturn) },
      { name: 'Sharpe Ratio', value: evaluation.sharpeRatio.toFixed(4), rating: this.rate('sharpeRatio', evaluation.sharpeRatio) },
      { name: 'Calmar Ratio', value: evaluation.calmarRatio.toFixed(4), rating: this.rate('calmarRatio', evaluation.calmarRatio) },
      { name: 'Sortino Ratio', value: evaluation.sortinoRatio.toFixed(4), rating: this.rate('sortinoRatio', evaluation.sortinoRatio) },
      { name: 'Max Drawdown', value: `${(evaluation.maxDrawdown * 100).toFixed(2)}%`, rating: this.rateInverse('maxDrawdown', evaluation.maxDrawdown) },
    ];

    const rows = metrics.map(m => {
      const color = m.rating === 'good' ? '#22c55e' : m.rating === 'bad' ? '#ef4444' : '#f59e0b';
      const label = m.rating === 'good' ? 'GOOD' : m.rating === 'bad' ? 'BAD' : 'OK';
      return `
        <tr>
          <td style="padding: 8px 12px; border-bottom: 1px solid #eee;">${m.name}</td>
          <td style="padding: 8px 12px; border-bottom: 1px solid #eee; font-weight: bold;">${m.value}</td>
          <td style="padding: 8px 12px; border-bottom: 1px solid #eee;">
            <span style="background: ${color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">${label}</span>
          </td>
        </tr>`;
    }).join('');

    return `
      <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #1e293b;">DeepScalper Training Report: ${symbol}</h2>

        <table style="margin-bottom: 16px; font-size: 14px; color: #475569;">
          <tr><td><strong>Epochs:</strong> ${epochs}</td></tr>
          <tr><td><strong>Episodes:</strong> ${episodes}</td></tr>
          <tr><td><strong>Final Training Net Value:</strong> ${trainingNetValue.toFixed(4)}</td></tr>
          ${duration ? `<tr><td><strong>Duration:</strong> ${duration}</td></tr>` : ''}
          <tr><td><strong>Time:</strong> ${new Date().toLocaleString()}</td></tr>
        </table>

        <h3 style="color: #334155;">Test Evaluation Metrics</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
          <tr style="background: #f8fafc;">
            <th style="padding: 8px 12px; text-align: left;">Metric</th>
            <th style="padding: 8px 12px; text-align: left;">Value</th>
            <th style="padding: 8px 12px; text-align: left;">Rating</th>
          </tr>
          ${rows}
        </table>

        <p style="margin-top: 20px; font-size: 12px; color: #94a3b8;">
          Thresholds: Return > 2% = Good, SR > 1.0 = Good, MDD < 5% = Good
        </p>
      </div>
    `;
  }

  /** Higher is better (return, sharpe, calmar, sortino). */
  private rate(metric: string, value: number): 'good' | 'ok' | 'bad' {
    const t = THRESHOLDS[metric];
    if (!t) return 'ok';
    if (value >= t.good) return 'good';
    if (value <= t.bad) return 'bad';
    return 'ok';
  }

  /** Lower is better (max drawdown). */
  private rateInverse(metric: string, value: number): 'good' | 'ok' | 'bad' {
    const t = THRESHOLDS[metric];
    if (!t) return 'ok';
    if (value <= t.good) return 'good';
    if (value >= t.bad) return 'bad';
    return 'ok';
  }
}

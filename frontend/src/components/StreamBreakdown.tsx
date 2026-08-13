/**
 * Per-stream decomposition.
 *
 * Always rendered, never collapsed by default. The proposal requires per-stream
 * contribution to be visible on every result, and a disclosure control that
 * hides it satisfies the letter of that while defeating its purpose — nobody
 * expands a panel to check whether the answer they already read was sound.
 *
 * Absent streams are listed with their reason rather than omitted. A stream that
 * silently disappears is indexed identically to one that was never attempted,
 * and the difference matters: "the recording was judged synthetic" is a finding,
 * "no transaction records exist" is a data-access problem.
 */

import React from 'react';

import { assertPermitted } from '../safety/language';
import type { StreamContribution } from '../api/types';

const ABSENCE_LABELS: Record<string, string> = {
  no_data: 'No data',
  insufficient_data: 'Too little data',
  excluded_by_validity_gate: 'Excluded — recording judged synthetic',
  quality_below_threshold: 'Quality below threshold',
  model_unavailable: 'No calibrated model deployed',
  out_of_calibration_domain: 'Outside the calibrated domain',
};

/** Bar width for a log10 LR, clamped so an extreme value cannot fill the row. */
function barWidth(log10Lr: number): number {
  return Math.min(Math.abs(log10Lr) / 6, 1) * 100;
}

export const StreamBreakdown: React.FC<{ streams: StreamContribution[] }> = ({
  streams,
}) => {
  const contributing = streams.filter((s) => s.status === 'evidence');
  const absent = streams.filter((s) => s.status === 'absent');

  return (
    <section className="stream-breakdown" aria-label="Contribution by evidence stream">
      <h4 className="stream-breakdown__title">
        Contribution by stream
        <span className="stream-breakdown__count">
          {contributing.length} of {streams.length} contributing
        </span>
      </h4>

      <ul className="stream-breakdown__list">
        {contributing.map((stream) => {
          const value = stream.log10_lr ?? 0;
          const spansNeutral =
            (stream.interval_lower_log10 ?? 0) <= 0 &&
            (stream.interval_upper_log10 ?? 0) >= 0;
          return (
            <li key={stream.stream} className="stream-row">
              <span className="stream-row__name">{stream.stream}</span>
              <span className="stream-row__bar-track">
                <span
                  className={`stream-row__bar stream-row__bar--${
                    value >= 0 ? 'supports' : 'opposes'
                  }`}
                  style={{ width: `${barWidth(value)}%` }}
                />
              </span>
              <span className="stream-row__value">
                {value >= 0 ? '+' : ''}
                {value.toFixed(2)}
              </span>
              {spansNeutral && (
                <span
                  className="stream-row__flag"
                  title="This stream's interval spans the neutral point, so its direction of support is not reliably determined."
                >
                  indeterminate
                </span>
              )}
            </li>
          );
        })}

        {absent.map((stream) => (
          <li key={stream.stream} className="stream-row stream-row--absent">
            <span className="stream-row__name">{stream.stream}</span>
            <span className="stream-row__absence">
              {ABSENCE_LABELS[stream.absence_reason ?? ''] ?? 'Absent'}
            </span>
            {stream.absence_detail && (
              <span className="stream-row__detail">
                {assertPermitted(stream.absence_detail, 'StreamBreakdown.detail')}
              </span>
            )}
          </li>
        ))}
      </ul>

      {absent.length > 0 && (
        <p className="stream-breakdown__note">
          Absent streams contributed nothing. They were not treated as evidence of
          similarity or of difference.
        </p>
      )}
    </section>
  );
};

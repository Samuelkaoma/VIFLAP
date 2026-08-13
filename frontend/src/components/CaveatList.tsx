/**
 * Caveats that must accompany a result.
 *
 * The list comes from the server, derived from the result's own structure rather
 * than assembled by a caller — so a presentation layer cannot omit one by
 * forgetting it exists.
 *
 * Rendered as a `<section>` with a heading rather than as small print. The
 * caveats are frequently the most decision-relevant text on the card: "the
 * posterior remains less probable than not" changes what an investigator should
 * do, and styling it as a footnote invites skipping it.
 */

import React from 'react';

import { assertPermitted } from '../safety/language';

export const CaveatList: React.FC<{ caveats: string[] }> = ({ caveats }) => {
  if (caveats.length === 0) return null;
  return (
    <section className="caveats" aria-label="Conditions on this result">
      <h4 className="caveats__title">Conditions on this result</h4>
      <ul className="caveats__list">
        {caveats.map((caveat, index) => (
          <li key={index} className="caveats__item">
            {assertPermitted(caveat, `CaveatList[${index}]`)}
          </li>
        ))}
      </ul>
    </section>
  );
};

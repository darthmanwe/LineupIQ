/**
 * The front door.
 *
 * This page went stale and stayed stale, which is worth recording because it is
 * the most damaging kind of staleness in the project. It said "Milestone 1 of 8
 * — every analytics endpoint returns 501" for the entire time sixteen endpoints
 * were live and both models were fitted. Every published *number* in this
 * repository is generated from a run log and re-derived in CI, and none of that
 * machinery covers prose. A visitor's first screen said the project was not
 * built.
 *
 * The fix is not to import the registry and count it — that would couple a
 * workerd-targeted package into the Next build for three integers. It is to
 * stop writing sentences that go stale, and to guard the class of claim that
 * did: `apps/api/test/registry.test.ts` fails the build if this page asserts
 * endpoints are unbuilt while the registry says they are live.
 */

export default function OverviewPage() {
  return (
    <>
      <style>{`
        h1 { font-size: 1.6rem; margin: 0 0 0.25rem; }
        .lede { color: var(--muted); max-width: 44rem; margin: 0 0 1.5rem; }
      `}</style>
      <h1>What this is, what it found, and where it stops</h1>
      <p className="lede">
        The tool, the result that came back opposite to what was predicted, the state of the build,
        and the questions this refuses to answer &mdash; in that order.
      </p>

      <section className="card">
        <h2 style={{ marginTop: 0 }}>What this does</h2>
        <p>
          Pick any five NBA players — including five who have never shared a floor — and LineupIQ
          estimates how that lineup moves a shooter&rsquo;s attempts between zones, and what the
          shift is worth in points. Every number ships the possession count behind it.
        </p>
        <p style={{ marginBottom: 0 }}>
          The harder half is knowing when <em>not</em> to answer. Most five-man lineups play under
          200 possessions a season, and at that sample the measurement noise on a lineup&rsquo;s
          offensive rating is about as large as the entire spread between lineups. So the API has a
          refusal contract with pre-registered, hash-pinned thresholds, and CI fails the build if a
          low-support lineup ever returns a confident number.
        </p>
      </section>

      <section className="card">
        <h2 style={{ marginTop: 0 }}>What it found</h2>
        <p>
          Lineup context does almost nothing to <em>whether</em> a shot goes in, and something real
          and small to <em>which</em> shot gets taken. Measuring conversion and concluding
          &ldquo;lineups don&rsquo;t matter&rdquo; answers the wrong question well: spacing
          doesn&rsquo;t make you a better corner shooter, it gets you a corner three instead of a
          contested pull-up.
        </p>
        <p style={{ marginBottom: 0 }}>
          The sign of every coefficient was registered before fitting, and one came back
          contradicted — decisively, not marginally. It is published as written, next to the
          coefficient that broke it. See <a href="/quality/">Data Quality &amp; Eval</a> for the
          full ladder, the negative controls and the sign audit.
        </p>
      </section>

      <section className="card">
        <h2 style={{ marginTop: 0 }}>Current state</h2>
        <p>
          Three seasons are ingested, lineups are reconstructed and validated against box-score
          minutes, and both shot models are fitted and served. The lineup scorer and the play
          ranking run live against real coefficients. Endpoints that are not backed yet return{" "}
          <code>501 NOT_YET_BACKED</code> naming what will back them, and a few return{" "}
          <code>410 METRIC_WITHDRAWN</code> because they need data this project does not have.
          Nothing returns a placeholder. The full list, with its state, is at{" "}
          <a href="/api/">/api/</a>.
        </p>
        <p style={{ marginBottom: 0 }}>
          The trade projection is among the <code>501</code>s <em>deliberately</em>. Its backtest
          ran, and the power analysis says the smallest effect the sample could detect is the same
          size as the effects it projects — so no accuracy claim follows, and shipping the number
          anyway is the failure this project is built to avoid.
        </p>
      </section>

      <section className="card">
        <h2 style={{ marginTop: 0 }}>What it will not do</h2>
        <ul style={{ marginBottom: 0 }}>
          <li>
            <strong>No tracking data.</strong> Shot difficulty is inferred from location and
            context, not observed defender position. There is no gravity metric and no contest
            quality — those endpoints return <code>410 METRIC_WITHDRAWN</code>, because they are not
            coming rather than not built.
          </li>
          <li>
            <strong>Three seasons.</strong> Nothing here generalises across rule eras.
          </li>
          <li>
            <strong>Nothing is causal.</strong> Trade projections assume a stated
            minutes-reallocation rule, which is a visible input rather than a hidden assumption.
          </li>
          <li>
            <strong>No language model has ever been called by this repository.</strong> The
            retrieval and groundedness harnesses run offline against committed fixtures — a
            deliberate choice for cost and determinism, and it means that layer is an evaluation
            harness rather than a feature.
          </li>
        </ul>
      </section>
    </>
  );
}

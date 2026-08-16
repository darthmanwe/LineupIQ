export default function OverviewPage() {
  return (
    <>
      <section className="card">
        <h2 style={{ marginTop: 0 }}>What this will do</h2>
        <p>
          Pick any five NBA players. LineupIQ estimates what each should shoot, from where, given
          who else is on the floor — then projects how a trade changes it. Every number ships the
          possession count behind it and the model&rsquo;s own measured error.
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
        <h2 style={{ marginTop: 0 }}>Current state</h2>
        <p>
          Milestone 1 of 8. The skeleton is deployed and honest: the route registry is live, and
          every analytics endpoint returns <code>501 NOT_YET_BACKED</code> naming the dataset and
          milestone that will back it. Nothing returns a placeholder.
        </p>
        <p style={{ marginBottom: 0 }}>
          Next: ingest three seasons of play-by-play and reconstruct which five players were on the
          floor for every event — validated against box-score minutes, which is a physical invariant
          and the one genuinely independent check available.
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
        </ul>
      </section>
    </>
  );
}

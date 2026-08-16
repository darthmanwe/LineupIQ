/**
 * The UI counterpart of the API's 501.
 *
 * It carries the same four fields the problem document does, so the page and
 * the endpoint tell an identical story. A page that renders an empty chart
 * while its endpoint returns 501 is the inconsistency this prevents.
 */
export type NotYetBackedProps = {
  page: string;
  willServe: string;
  milestone: string;
  backedBy: string;
  endpoints: readonly string[];
  /** What this page will refuse to answer, and why. */
  refusal?: string;
};

export function NotYetBacked({
  page,
  willServe,
  milestone,
  backedBy,
  endpoints,
  refusal,
}: NotYetBackedProps) {
  return (
    <section className="card pending">
      <span className="tag">not yet backed</span>
      <h2 style={{ marginTop: "0.6rem" }}>{page}</h2>
      <p style={{ marginBottom: 0 }}>{willServe}</p>

      <dl className="spec">
        <dt>arrives in</dt>
        <dd>{milestone}</dd>

        <dt>backed by</dt>
        <dd>
          <code>{backedBy}</code>
        </dd>

        <dt>endpoints</dt>
        <dd>
          {endpoints.map((e, i) => (
            <span key={e}>
              {i > 0 && ", "}
              <code>{e}</code>
            </span>
          ))}
        </dd>

        {refusal && (
          <>
            <dt>will refuse</dt>
            <dd>{refusal}</dd>
          </>
        )}
      </dl>
    </section>
  );
}

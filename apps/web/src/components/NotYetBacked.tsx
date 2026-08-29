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
  /**
   * The part of this page's subject that *is* built, if any.
   *
   * A page can be unbacked as a product surface while the work behind it is
   * done and published. The Evidence page had exactly that shape and listed a
   * live endpoint among the ones it said were missing -- so the page was
   * understating the repository, which is a nicer failure than overstating it
   * and still a failure.
   */
  alreadyDone?: { what: string; endpoint: string };
};

export function NotYetBacked({
  page,
  willServe,
  milestone,
  backedBy,
  endpoints,
  refusal,
  alreadyDone,
}: NotYetBackedProps) {
  return (
    <section className="card pending">
      <span className="tag">not yet backed</span>
      {/*
        An `<h1>`, because this card *is* the page it is rendered on -- it names
        the surface and nothing else on that page outranks it. The banner's
        wordmark is deliberately not a heading, so this is the document's only
        level one.
      */}
      <h1 style={{ marginTop: "0.6rem" }}>{page}</h1>
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

        {alreadyDone && (
          <>
            <dt>already built</dt>
            <dd>
              {alreadyDone.what} <code>{alreadyDone.endpoint}</code>
            </dd>
          </>
        )}
      </dl>
    </section>
  );
}

import { NotYetBacked } from "@/components/NotYetBacked";

export default function EvidencePage() {
  return (
    <NotYetBacked
      page="Evidence / Comps"
      willServe="Free-text search over historical lineup documents, with a retriever toggle so you can watch lexical and dense retrieval disagree."
      milestone="M6 - retrieval and the LLM evaluation harness"
      backedBy="D1 FTS5 + Vectorize, mirrored offline by rank_bm25 + hnswlib"
      endpoints={["GET /api/evidence/search"]}
      refusal="A query matching nothing above the score floor says so and shows the nearest three with their scores, rather than returning an empty list that looks like a bug."
      alreadyDone={{
        what: "The retrieval evaluation itself is built and published - three corpora ablated, Recall@10 and nDCG@10 per retriever, and a finding that BM25 alone beats the dense leg on this corpus.",
        endpoint: "GET /api/eval/retrieval",
      }}
    />
  );
}

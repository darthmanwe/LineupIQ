import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "LineupIQ",
  description:
    "Given any five NBA players, which shots each should take — with the possession count behind every number, and an explicit refusal when there isn't one.",
};

const TABS = [
  { href: "/", label: "Overview" },
  { href: "/lineup/", label: "Lineup Optimizer" },
  { href: "/trade/", label: "Trade Simulator" },
  { href: "/evidence/", label: "Evidence" },
  { href: "/quality/", label: "Data Quality & Eval" },
] as const;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <header className="masthead">
            {/*
              The wordmark is not a heading.

              It used to be an `<h1>`, which put the same level-one heading on
              every page of the site -- and three pages then added an `<h1>` of
              their own, so those carried two. A heading outline that says
              "LineupIQ" first on all five pages tells a screen-reader user
              nothing about which page they are on, which is the one job the
              outline has.

              So the site name is a styled paragraph in the banner, and each
              page owns exactly one `<h1>` describing itself.
              `registry.test.ts` fails the build if that ever stops being true.
            */}
            <p className="wordmark">LineupIQ</p>
            <p>
              Given any five NBA players, which shots each should take — with the possession count
              behind every number, and an explicit refusal when there isn&rsquo;t one.
            </p>
            <nav className="tabs">
              {TABS.map((t) => (
                // `prefetch={false}` because this is a static export. The App
                // Router's prefetch asks for an RSC payload
                // (`__next.<route>.__PAGE__.txt`) that `output: "export"` never
                // emits, so every page load fired four 404s and logged four
                // console errors -- on every page, for every visitor with
                // devtools open. Clicking was never affected, which is why it
                // went unnoticed: navigation works, the prefetch does not.
                <Link key={t.href} href={t.href} prefetch={false}>
                  {t.label}
                </Link>
              ))}
            </nav>
          </header>

          <main>{children}</main>

          <footer className="colophon">
            <p>
              Every number here is generated from a run log, never typed by hand. The full endpoint
              list and each one&rsquo;s state is at <a href="/api">/api</a>.
            </p>
            <p>
              <a href="https://github.com/darthmanwe/LineupIQ">Source</a> · MIT · Kutlu Mizrak
            </p>
          </footer>
        </div>
      </body>
    </html>
  );
}

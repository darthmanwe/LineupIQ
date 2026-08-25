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
            <h1>LineupIQ</h1>
            <p>
              Given any five NBA players, which shots each should take — with the possession count
              behind every number, and an explicit refusal when there isn&rsquo;t one.
            </p>
            <nav className="tabs">
              {TABS.map((t) => (
                <Link key={t.href} href={t.href}>
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

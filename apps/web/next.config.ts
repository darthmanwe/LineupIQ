import type { NextConfig } from "next";

/**
 * Static export, served by the same Worker that serves the API.
 *
 * One deploy, one origin, no CORS, and asset requests never invoke the Worker --
 * so page loads cost nothing against the request budget and the demo stays free
 * regardless of traffic.
 */
const nextConfig: NextConfig = {
  output: "export",
  // The Worker serves these as static files; there is no image optimiser behind them.
  images: { unoptimized: true },
  // Trailing slashes make directory-style static hosting resolve predictably.
  trailingSlash: true,
  reactStrictMode: true,
};

export default nextConfig;

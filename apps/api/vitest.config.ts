import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

/**
 * Tests execute inside workerd, not Node.
 *
 * This matters more here than usual: the serving design rests on a closed form
 * that must produce bit-identical results to the Python fit, and floating point
 * in a Node process proves nothing about the runtime that serves requests.
 *
 * Note: pool-workers 0.21 replaced the old `defineWorkersConfig` helper with
 * this Vite plugin. Most guides online still show the helper; it is gone.
 */
export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.toml", environment: "dev" },
      miniflare: {
        compatibilityDate: "2026-08-01",
        compatibilityFlags: ["nodejs_compat"],
      },
      // Off explicitly. This defaults to true in 0.21, which would let the suite
      // bind to real Cloudflare resources over the network. Tests must be
      // reproducible offline and must never reach a live account.
      remoteBindings: false,
    }),
  ],
});

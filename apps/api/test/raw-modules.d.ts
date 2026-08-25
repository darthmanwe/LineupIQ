/**
 * `?raw` imports, declared for `tsc`.
 *
 * Vite inlines a file's text at build time for a `?raw` suffix, and `tsc` knows
 * nothing about it — so the type-check leg failed on CI while the test suite
 * passed locally, which is exactly the split a separate `js-quality` job exists
 * to catch.
 *
 * Used by `registry.test.ts` to read the overview page's source. `readFileSync`
 * is not an option: there is no filesystem in workerd, and an `import.meta.url`
 * path resolves to `/C:/...` on Windows, which its shim rejects.
 */
declare module "*?raw" {
  const content: string;
  export default content;
}

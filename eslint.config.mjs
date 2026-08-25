import js from "@eslint/js";
import prettier from "eslint-config-prettier";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "**/node_modules/**",
      "**/.next/**",
      "**/out/**",
      "**/.wrangler/**",
      "**/dist/**",
      "**/*.d.ts",
      "data/**",
      "services/ml/**",
    ],
  },

  js.configs.recommended,
  ...tseslint.configs.recommended,

  {
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
    rules: {
      // Unused args are often part of a signature the framework dictates;
      // an underscore prefix is the explicit "I know" marker.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // `any` erases the type safety the envelope and problem shapes exist to
      // provide, so it is an error rather than a warning.
      "@typescript-eslint/no-explicit-any": "error",
      "no-console": ["warn", { allow: ["warn", "error"] }],
      eqeqeq: ["error", "always", { null: "ignore" }],
    },
  },

  // Tests assert on loosely-typed JSON bodies; casting every fixture would add
  // noise without adding safety.
  {
    files: ["**/test/**/*.ts", "**/*.test.ts"],
    rules: { "@typescript-eslint/no-explicit-any": "off" },
  },

  // Build scripts are CLIs. `no-console` exists to stop stray debugging output
  // reaching a Worker log; in a script whose entire job is to report what it
  // captured, stdout *is* the interface.
  {
    files: ["scripts/**/*.mjs"],
    rules: { "no-console": "off" },
  },

  // Must stay last: turns off every stylistic rule Prettier already owns.
  prettier
);

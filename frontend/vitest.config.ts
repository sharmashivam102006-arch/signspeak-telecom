import { defineConfig } from "vitest/config";

// Frontend unit tests: pure logic in lib/ (sign smoothing, caption helpers,
// API client). Default environment is node — tests that need a DOM use the
// `// @vitest-environment jsdom` docblock instead.
export default defineConfig({
  test: {
    include: ["lib/__tests__/**/*.test.{ts,tsx}"],
    environment: "node",
  },
});

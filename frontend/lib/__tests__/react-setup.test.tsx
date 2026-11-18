// @vitest-environment jsdom
// Smoke test for the @testing-library/react setup itself: proves the
// jsdom environment + React rendering pipeline work, so future component
// tests can build on it.
import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

describe("React testing setup", () => {
  it("renders an element and finds it by text", () => {
    render(<p>SignSpeak test setup</p>);
    expect(screen.getByText("SignSpeak test setup")).toBeTruthy();
  });
});

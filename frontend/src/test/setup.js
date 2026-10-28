import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import React from "react";
import { afterEach, vi } from "vitest";

// jsdom has no layout, so Recharts' ResponsiveContainer would measure 0 x 0 and draw nothing.
// In tests, give every chart a fixed size so the SVG really renders.
vi.mock("recharts", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    ResponsiveContainer: ({ children }) =>
      React.createElement("div", { style: { width: 800, height: 300 } }, React.cloneElement(children, { width: 800, height: 300 })),
  };
});

afterEach(() => cleanup());

import { describe, expect, it } from "vitest";
import { addDays, inRange, isIsoDate, longDate, shortDate, ukDateOf, ukFullDateTime, ukTime, ukZoneName } from "./dates.js";

describe("calendar dates", () => {
  it("adds days across month and year ends", () => {
    expect(addDays("2026-01-31", 1)).toBe("2026-02-01");
    expect(addDays("2026-01-01", -1)).toBe("2025-12-31");
  });

  it("accepts only real YYYY-MM-DD dates", () => {
    expect(isIsoDate("2026-01-15")).toBe(true);
    expect(isIsoDate("2026-02-30")).toBe(false);
    expect(isIsoDate("15/01/2026")).toBe(false);
    expect(isIsoDate("")).toBe(false);
    expect(isIsoDate(null)).toBe(false);
  });

  it("checks the decision window inclusively", () => {
    expect(inRange("2026-01-01", "2026-01-01", "2026-01-31")).toBe(true);
    expect(inRange("2026-01-31", "2026-01-01", "2026-01-31")).toBe(true);
    expect(inRange("2026-02-01", "2026-01-01", "2026-01-31")).toBe(false);
  });

  it("formats calendar dates without shifting them", () => {
    expect(longDate("2026-01-15")).toMatch(/Thu,? 15 Jan 2026/);
    expect(shortDate("2026-01-01")).toBe("1 Jan");
  });
});

describe("UK local time", () => {
  it("is GMT in January, so the 09:50 cutoff is 09:50 UTC", () => {
    expect(ukTime("2026-01-14T09:50:00Z")).toBe("09:50");
    expect(ukZoneName("2026-01-14T09:50:00Z")).toBe("GMT");
  });

  it("is BST in summer, so 08:50 UTC is the 09:50 UK cutoff", () => {
    expect(ukTime("2025-10-25T08:50:00Z")).toBe("09:50");
    expect(ukZoneName("2025-10-25T08:50:00Z")).toBe("BST");
  });

  it("formats a full UK date and time with the year", () => {
    expect(ukFullDateTime("2026-10-25T19:31:49Z")).toMatch(/^Sun,? 25 Oct 2026, 19:31$/); // GMT again after the 25 Oct clock change
  });

  it("assigns the UK calendar day (23:00 UTC in BST is already the next UK day)", () => {
    expect(ukDateOf("2025-10-25T23:00:00Z")).toBe("2025-10-26");
    expect(ukDateOf("2026-01-15T23:00:00Z")).toBe("2026-01-15");
  });
});

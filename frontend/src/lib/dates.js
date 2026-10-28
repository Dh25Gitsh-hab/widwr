/**
 * Date helpers. The backend sends UTC timestamps ("...Z") and plain calendar dates
 * ("2026-01-15"). Timestamps are always shown in UK local time (Europe/London), which is
 * GMT throughout January; calendar dates are formatted as dates, never shifted.
 */

export const UK_TIME_ZONE = "Europe/London";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

/** Add whole days to a YYYY-MM-DD calendar date. */
export function addDays(isoDate, days) {
  const [year, month, day] = isoDate.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day + days)).toISOString().slice(0, 10);
}

/** True for a real calendar date written as YYYY-MM-DD (rejects 2026-02-30). */
export function isIsoDate(value) {
  return typeof value === "string" && ISO_DATE.test(value) && addDays(value, 0) === value;
}

export function inRange(isoDate, first, last) {
  return isoDate >= first && isoDate <= last;
}

function ukFormat(options) {
  return new Intl.DateTimeFormat("en-GB", { timeZone: UK_TIME_ZONE, ...options });
}

/** "09:50" in UK local time. */
export function ukTime(timestamp) {
  return ukFormat({ hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(timestamp));
}

/** "Wed 14 Jan, 09:30" in UK local time. */
export function ukDateTime(timestamp) {
  return ukFormat({ weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(
    new Date(timestamp),
  );
}

/** "Sun 25 Oct 2026, 19:31" in UK local time (with the year, for timestamps outside the decision window). */
export function ukFullDateTime(timestamp) {
  return ukFormat({ weekday: "short", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(
    new Date(timestamp),
  );
}

/** "GMT" or "BST" for a timestamp. */
export function ukZoneName(timestamp) {
  const part = ukFormat({ timeZoneName: "short" }).formatToParts(new Date(timestamp)).find((p) => p.type === "timeZoneName");
  return part?.value ?? "UK time";
}

/** UK calendar date (YYYY-MM-DD) of a UTC timestamp. */
export function ukDateOf(timestamp) {
  const parts = Object.fromEntries(
    ukFormat({ year: "numeric", month: "2-digit", day: "2-digit" })
      .formatToParts(new Date(timestamp))
      .map((p) => [p.type, p.value]),
  );
  return `${parts.year}-${parts.month}-${parts.day}`;
}

/** "Thu 15 Jan 2026" for a calendar date (no time-zone shift). */
export function longDate(isoDate) {
  return new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", weekday: "short", day: "numeric", month: "short", year: "numeric" }).format(
    new Date(`${isoDate}T00:00:00Z`),
  );
}

/** "15 Jan" for a calendar date. */
export function shortDate(isoDate) {
  return new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", day: "numeric", month: "short" }).format(new Date(`${isoDate}T00:00:00Z`));
}

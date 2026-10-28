/**
 * Makes a backend error reason safe to show under "Technical details".
 *
 * The backend reports Python exception text, which can contain absolute file paths of the
 * machine it runs on and developer commands (quoted in backticks). Paths are cut to the file
 * name and command clauses are dropped; everything else (the reason itself) is kept.
 */

const README_NOTE = "Setup steps are in the project README.";

// Quoted absolute paths may contain spaces ('E:\\My Project\\file.joblib', '/srv/app/file.joblib').
const QUOTED_PATH = /(['"])((?:[A-Za-z]:[\\/]|\\\\|\/)[^'"]*)\1/g;
// Unquoted Windows paths, and POSIX paths with at least two segments (not the "//" of a URL).
const WINDOWS_PATH = /(?<!\w)[A-Za-z]:[\\/][^\s'"`,;)]*/g; // (?<!\w): not the "p:/" of "http://"
const POSIX_PATH = /(?<![\w.:/])\/(?:[\w.-]+\/)+[\w.-]+/g;

function fileName(path) {
  const parts = path.split(/[\\/]+/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : path;
}

export function safeDetail(text) {
  if (typeof text !== "string" || !text.trim()) return null;
  const clauses = text.split(" - ");
  const kept = clauses.filter((clause) => !clause.includes("`"));
  let out = kept
    .join(" - ")
    .replace(/`[^`]*`/g, "")
    .replace(QUOTED_PATH, (_, quote, path) => `${quote}${fileName(path)}${quote}`)
    .replace(WINDOWS_PATH, fileName)
    .replace(POSIX_PATH, fileName)
    .trim();
  if (kept.length !== clauses.length || text.includes("`")) {
    const separator = !out ? "" : /[.!?]$/.test(out) ? " " : ". ";
    out = `${out}${separator}${README_NOTE}`;
  }
  return out;
}

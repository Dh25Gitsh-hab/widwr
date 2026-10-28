import { describe, expect, it } from "vitest";
import { safeDetail } from "./diagnostics.js";

describe("safeDetail (backend reasons shown under Technical details)", () => {
  it("cuts a quoted Windows path with spaces (as Python reports it) to the file name", () => {
    const raw = "ML unavailable: [Errno 2] No such file or directory: 'E:\\\\GridBricks Job\\\\backend\\\\ml\\\\models\\\\error_regressor.joblib'";
    expect(safeDetail(raw)).toBe("ML unavailable: [Errno 2] No such file or directory: 'error_regressor.joblib'");
  });

  it("cuts unquoted Windows and POSIX paths to the file name", () => {
    expect(safeDetail("Could not read C:\\Users\\someone\\data\\fuelhh.csv")).toBe("Could not read fuelhh.csv");
    expect(safeDetail("Could not read /home/someone/app/data/fuelhh.csv now")).toBe("Could not read fuelhh.csv now");
  });

  it("drops developer commands and points to the README instead", () => {
    const raw = "ML unavailable: missing file - the artifacts are built by `python -m ml.build_dataset`, `python -m ml.train` and `python -m ml.holdout.predict`";
    expect(safeDetail(raw)).toBe("ML unavailable: missing file. Setup steps are in the project README.");
  });

  it("leaves URLs and ordinary messages unchanged", () => {
    const network = "Cannot reach the backend at http://127.0.0.1:8000. Check that it is running and that its CORS settings (WFM_CORS_ORIGINS) allow http://localhost:5173.";
    expect(safeDetail(network)).toBe(network);
    expect(safeDetail("Delivery dates must be within 2026-01-02 .. 2026-02-01")).toBe("Delivery dates must be within 2026-01-02 .. 2026-02-01");
  });

  it("returns null when there is nothing to show", () => {
    expect(safeDetail(null)).toBeNull();
    expect(safeDetail("   ")).toBeNull();
  });
});

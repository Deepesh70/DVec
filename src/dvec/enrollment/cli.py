"""Command-Line Interface for Speaker Enrollment."""

import argparse
import sys
from pathlib import Path
from dvec.enrollment.pipeline import enroll_user, EnrollmentError


def main():
    parser = argparse.ArgumentParser(
        prog="dvec-enroll",
        description="Enroll a user voice profile for real-time Target Speaker Extraction.",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to 10-second reference voice recording (16 kHz WAV recommended).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="enrollment.bin",
        help="Output destination for serialized 1024-byte embedding (default: enrollment.bin).",
    )

    args = parser.parse_args()

    print("==================================================")
    print("        DVec Speaker Enrollment Engine            ")
    print("==================================================")
    print(f"Reading input audio: {args.input}")

    try:
        saved_path, report = enroll_user(args.input, args.output)
        print("\n[SUCCESS] Voice recording passed all sanity gates:")
        print(f"  • RMS Level:      {report.rms_dbfs:.1f} dBFS (Valid: [-35, -1])")
        print(f"  • Peak Level:     {report.peak_dbfs:.1f} dBFS")
        print(f"  • Voicing Ratio:  {report.voicing_ratio*100:.1f}% (Minimum: 65%)")
        print(f"  • Speaker Purity: {report.min_subsegment_similarity:.2f} (Minimum: 0.75)")
        print(f"\nSaved 256-d vector artifact: {saved_path} ({saved_path.stat().st_size} bytes)")
        print("Ready for real-time daemon extraction conditioning.")
        sys.exit(0)
    except EnrollmentError as e:
        print(f"\n[REJECTED] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected enrollment failure: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

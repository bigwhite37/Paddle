#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass

import numpy as np

paddle = None

DTYPE_TOLERANCE = {
    "float32": (1e-4, 1e-4),
    "float16": (5e-2, 5e-2),
    "bfloat16": (1e-1, 1e-1),
}

STATUS_PASS = "PASS"
STATUS_FAIL_NUMERIC = "FAIL_NUMERIC"
STATUS_FAIL_RUNTIME_CUBLAS = "FAIL_RUNTIME_CUBLAS"
STATUS_FAIL_CUBLASLT = "FAIL_CUBLASLT"
STATUS_FAIL_OTHER = "FAIL_OTHER"


@dataclass(frozen=True)
class Case:
    name: str
    paddle_api: str
    runner: object
    cublas_path: object
    uses_cublaslt: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify Paddle top-level Python APIs that exercise cuBLAS/cuBLASLt paths."
    )
    parser.add_argument(
        "--dtypes",
        default="float32,float16,bfloat16",
        help="Comma-separated dtype list: float32,float16,bfloat16.",
    )
    parser.add_argument(
        "--legacy-linear",
        action="store_true",
        help="Set FLAGS_use_legacy_linear=true before importing Paddle.",
    )
    parser.add_argument(
        "--skip-cublaslt",
        action="store_true",
        help="Skip cases whose expected backend is cuBLASLt.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON result document instead of human-readable rows.",
    )
    parser.add_argument(
        "--show-traceback",
        action="store_true",
        help="Print Python traceback for failed cases in human-readable mode.",
    )
    return parser.parse_args()


def import_paddle(args):
    global paddle
    if args.legacy_linear:
        os.environ["FLAGS_use_legacy_linear"] = "true"
    import paddle as paddle_module

    paddle = paddle_module


def make_array(shape, scale, offset):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return data * scale + offset


def make_tensor(array, dtype):
    tensor = paddle.to_tensor(array.astype("float32"))
    if dtype != "float32":
        tensor = tensor.astype(dtype)
    return tensor


def as_float32_numpy(tensor):
    return tensor.astype("float32").numpy()


def cublas_gemm_path(dtype):
    if dtype == "float32":
        return "cublasSgemm/cublasGemm"
    return "cublasGemmEx"


def cublas_strided_batched_path(dtype):
    if dtype == "float32":
        return "cublasGemmStridedBatchedEx"
    return "cublasGemmStridedBatchedEx"


def linear_path(dtype):
    if os.environ.get("FLAGS_use_legacy_linear", "").lower() in ("1", "true", "on", "yes"):
        return f"{cublas_gemm_path(dtype)} + elementwise_add"
    return "cublasLtMatmul"


def run_matmul(dtype):
    x_np = make_array((3, 5), 0.07, -0.3)
    y_np = make_array((5, 4), -0.05, 0.2)
    out = paddle.matmul(make_tensor(x_np, dtype), make_tensor(y_np, dtype))
    return as_float32_numpy(out), x_np @ y_np


def run_matmul_transpose_y(dtype):
    x_np = make_array((3, 5), 0.03, -0.1)
    y_np = make_array((4, 5), 0.04, 0.2)
    out = paddle.matmul(make_tensor(x_np, dtype), make_tensor(y_np, dtype), transpose_y=True)
    return as_float32_numpy(out), x_np @ y_np.T


def run_mm(dtype):
    x_np = make_array((4, 6), 0.02, -0.2)
    y_np = make_array((6, 3), 0.06, 0.1)
    out = paddle.mm(make_tensor(x_np, dtype), make_tensor(y_np, dtype))
    return as_float32_numpy(out), x_np @ y_np


def run_bmm(dtype):
    x_np = make_array((2, 3, 5), 0.02, -0.1)
    y_np = make_array((2, 5, 4), -0.03, 0.25)
    out = paddle.bmm(make_tensor(x_np, dtype), make_tensor(y_np, dtype))
    return as_float32_numpy(out), np.matmul(x_np, y_np)


def run_linear(dtype):
    x_np = make_array((2, 5), 0.03, -0.4)
    weight_np = make_array((5, 4), -0.02, 0.15)
    bias_np = make_array((4,), 0.01, -0.02)
    out = paddle.nn.functional.linear(
        make_tensor(x_np, dtype),
        make_tensor(weight_np, dtype),
        make_tensor(bias_np, dtype),
    )
    return as_float32_numpy(out), x_np @ weight_np + bias_np


def build_cases():
    linear_is_cublaslt = linear_path("float32") == "cublasLtMatmul"
    return [
        Case("matmul", "paddle.matmul", run_matmul, cublas_gemm_path),
        Case(
            "matmul_transpose_y",
            "paddle.matmul(transpose_y=True)",
            run_matmul_transpose_y,
            cublas_gemm_path,
        ),
        Case("mm", "paddle.mm", run_mm, cublas_gemm_path),
        Case("bmm", "paddle.bmm", run_bmm, cublas_strided_batched_path),
        Case(
            "linear",
            "paddle.nn.functional.linear",
            run_linear,
            linear_path,
            uses_cublaslt=linear_is_cublaslt,
        ),
    ]


def numeric_error(actual, expected):
    expected = expected.astype("float32")
    diff = np.abs(actual - expected)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / (np.abs(expected) + 1e-12)))
    return max_abs, max_rel


def first_line(text):
    return str(text).strip().splitlines()[0] if str(text).strip() else ""


def classify_exception(exc, case):
    message = str(exc)
    lowered = message.lower()
    if case.uses_cublaslt or "cublaslt" in lowered or "no gemm algorithm" in lowered:
        return STATUS_FAIL_CUBLASLT
    if "cublas" in lowered or "cublas_status" in lowered:
        return STATUS_FAIL_RUNTIME_CUBLAS
    return STATUS_FAIL_OTHER


def run_case(case, dtype, show_traceback):
    path = case.cublas_path(dtype)
    try:
        actual, expected = case.runner(dtype)
        max_abs, max_rel = numeric_error(actual, expected)
        atol, rtol = DTYPE_TOLERANCE[dtype]
        status = STATUS_PASS if max_abs <= atol or max_rel <= rtol else STATUS_FAIL_NUMERIC
        result = {
            "status": status,
            "case": case.name,
            "paddle_api": case.paddle_api,
            "dtype": dtype,
            "expected_backend": path,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "atol": atol,
            "rtol": rtol,
        }
    except Exception as exc:
        result = {
            "status": classify_exception(exc, case),
            "case": case.name,
            "paddle_api": case.paddle_api,
            "dtype": dtype,
            "expected_backend": path,
            "error": first_line(exc),
        }
        if show_traceback:
            result["traceback"] = traceback.format_exc()
    return result


def print_human_result(result, show_traceback):
    prefix = f"[{result['status']}]"
    details = (
        f"api={result['paddle_api']:<34} "
        f"dtype={result['dtype']:<8} "
        f"backend={result['expected_backend']}"
    )
    if result["status"] == STATUS_PASS:
        print(
            f"{prefix:<22} {details} "
            f"max_abs={result['max_abs_error']:.6g} max_rel={result['max_rel_error']:.6g}"
        )
    elif result["status"] == STATUS_FAIL_NUMERIC:
        print(
            f"{prefix:<22} {details} "
            f"max_abs={result['max_abs_error']:.6g} max_rel={result['max_rel_error']:.6g} "
            f"tol=({result['atol']},{result['rtol']})"
        )
    else:
        print(f"{prefix:<22} {details} error={result.get('error', '')}")
        if show_traceback and result.get("traceback"):
            print(result["traceback"].rstrip())


def parse_dtypes(value):
    dtypes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [dtype for dtype in dtypes if dtype not in DTYPE_TOLERANCE]
    if invalid:
        raise ValueError(f"unsupported dtype(s): {','.join(invalid)}")
    return dtypes


def summarize(results):
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return counts


def main():
    args = parse_args()
    try:
        dtypes = parse_dtypes(args.dtypes)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    import_paddle(args)

    if not paddle.device.is_compiled_with_cuda():
        print("[FAIL] installed Paddle is not compiled with CUDA/M100 GPU support")
        return 1

    paddle.disable_static()
    paddle.set_device("gpu:0")
    paddle.seed(20260604)
    np.random.seed(20260604)

    cases = [case for case in build_cases() if not (args.skip_cublaslt and case.uses_cublaslt)]
    results = []

    if not args.json:
        print(f"paddle_version={paddle.__version__}")
        print(f"paddle_file={paddle.__file__}")
        print(f"device={paddle.device.get_device()}")
        print(f"FLAGS_use_legacy_linear={os.environ.get('FLAGS_use_legacy_linear', '')}")

    with paddle.no_grad():
        for dtype in dtypes:
            for case in cases:
                result = run_case(case, dtype, args.show_traceback)
                results.append(result)
                if not args.json:
                    print_human_result(result, args.show_traceback)

    counts = summarize(results)
    failed = sum(count for status, count in counts.items() if status != STATUS_PASS)

    if args.json:
        print(
            json.dumps(
                {
                    "paddle_version": paddle.__version__,
                    "paddle_file": paddle.__file__,
                    "device": paddle.device.get_device(),
                    "legacy_linear": bool(args.legacy_linear),
                    "summary": counts,
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        summary = ", ".join(f"{key.lower()}={value}" for key, value in sorted(counts.items()))
        print(f"Results: {summary}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

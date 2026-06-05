#!/usr/bin/env python3
import argparse
import sys
import traceback

import numpy as np
import paddle


EXPECTED_M100_BLOCK_MESSAGES = (
    "not supported by xtrans",
    "not support by xtrans",
)


DTYPE_TOLERANCE = {
    "float32": (1e-4, 1e-4),
    "float16": (5e-2, 5e-2),
    "bfloat16": (1e-1, 1e-1),
}


def make_array(shape, scale=0.1, offset=0.0):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return data * scale + offset


def make_tensor(array, dtype):
    tensor = paddle.to_tensor(array.astype("float32"))
    if dtype != "float32":
        tensor = tensor.astype(dtype)
    return tensor


def as_float32_numpy(tensor):
    return tensor.astype("float32").numpy()


def assert_close(name, actual, expected, dtype):
    atol, rtol = DTYPE_TOLERANCE[dtype]
    np.testing.assert_allclose(actual, expected.astype("float32"), atol=atol, rtol=rtol)
    print(f"[PASS] {name:<28} dtype={dtype}")


def run_matmul(dtype):
    x_np = make_array((3, 5), 0.07, -0.3)
    y_np = make_array((5, 4), -0.05, 0.2)
    out = paddle.matmul(make_tensor(x_np, dtype), make_tensor(y_np, dtype))
    assert_close("paddle.matmul", as_float32_numpy(out), x_np @ y_np, dtype)


def run_matmul_transpose(dtype):
    x_np = make_array((3, 5), 0.03, -0.1)
    y_np = make_array((4, 5), 0.04, 0.2)
    out = paddle.matmul(make_tensor(x_np, dtype), make_tensor(y_np, dtype), transpose_y=True)
    assert_close("paddle.matmul(trans_y)", as_float32_numpy(out), x_np @ y_np.T, dtype)


def run_mm(dtype):
    x_np = make_array((4, 6), 0.02, -0.2)
    y_np = make_array((6, 3), 0.06, 0.1)
    out = paddle.mm(make_tensor(x_np, dtype), make_tensor(y_np, dtype))
    assert_close("paddle.mm", as_float32_numpy(out), x_np @ y_np, dtype)


def run_bmm(dtype):
    x_np = make_array((2, 3, 5), 0.02, -0.1)
    y_np = make_array((2, 5, 4), -0.03, 0.25)
    out = paddle.bmm(make_tensor(x_np, dtype), make_tensor(y_np, dtype))
    assert_close("paddle.bmm", as_float32_numpy(out), np.matmul(x_np, y_np), dtype)


def run_linear(dtype):
    x_np = make_array((2, 5), 0.03, -0.4)
    weight_np = make_array((5, 4), -0.02, 0.15)
    bias_np = make_array((4,), 0.01, -0.02)
    out = paddle.nn.functional.linear(
        make_tensor(x_np, dtype), make_tensor(weight_np, dtype), make_tensor(bias_np, dtype)
    )
    assert_close("paddle.nn.functional.linear", as_float32_numpy(out), x_np @ weight_np + bias_np, dtype)


def is_expected_m100_block(exc):
    message = str(exc)
    return any(token in message for token in EXPECTED_M100_BLOCK_MESSAGES)


def run_case(case_fn, dtype, required):
    try:
        case_fn(dtype)
        return "pass"
    except Exception as exc:
        name = getattr(case_fn, "__name__", str(case_fn)).replace("run_", "")
        if not required and is_expected_m100_block(exc):
            print(f"[EXPECTED_BLOCKED] {name:<22} dtype={dtype}: {exc}")
            return "expected_blocked"
        print(f"[FAIL] {name:<28} dtype={dtype}: {exc}")
        traceback.print_exc()
        return "fail"


def parse_args():
    parser = argparse.ArgumentParser(description="Paddle top-level BLAS/GEMM smoke test for M100")
    parser.add_argument(
        "--optional-dtypes",
        default="float16,bfloat16",
        help="Comma-separated dtypes that may pass or be expected-blocked on M100.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not paddle.device.is_compiled_with_cuda():
        print("[FAIL] installed Paddle is not compiled with CUDA/M100 GPU support")
        return 1

    paddle.disable_static()
    paddle.set_device("gpu:0")
    paddle.seed(20260604)
    np.random.seed(20260604)

    print(f"paddle_version={paddle.__version__}")
    print(f"device={paddle.device.get_device()}")

    cases = [run_matmul, run_matmul_transpose, run_mm, run_bmm, run_linear]
    results = []

    with paddle.no_grad():
        for case_fn in cases:
            results.append(run_case(case_fn, "float32", required=True))

        optional_dtypes = [item.strip() for item in args.optional_dtypes.split(",") if item.strip()]
        for dtype in optional_dtypes:
            for case_fn in cases:
                results.append(run_case(case_fn, dtype, required=False))

    pass_count = results.count("pass")
    expected_blocked_count = results.count("expected_blocked")
    fail_count = results.count("fail")
    print(
        f"Results: pass={pass_count}, expected_blocked={expected_blocked_count}, fail={fail_count}"
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

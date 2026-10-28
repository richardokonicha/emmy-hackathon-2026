#!/usr/bin/env python3
"""Check that every PTX mnemonic ``compile.py`` can emit assembles on the runner.

The runner's ``ptxas`` is older than the one in the CUDA toolkit that the task
was written against: it rejects ``setmaxnreg`` outright.  A mnemonic the
assembler does not know turns a correct kernel into a compile error, so each
mnemonic is probed here in isolation with hand-written operands.

    .venv/bin/python dev/probe_ptx_mnemonics.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

URL = "http://runner.riftstack.ai:48701/compile"

HEADER = """
.version 8.0
.target sm_89
.address_size 64

.visible .entry k(.param .u64 p0, .param .u64 p1)
{
    .reg .pred  %p<5>;
    .reg .b16   %h<5>;
    .reg .b32   %r<33>;
    .reg .b64   %rd<9>;
    .reg .f32   %f<33>;
    .reg .f64   %d<9>;
    ld.param.u64 %rd1, [p0];
    ld.param.u64 %rd2, [p1];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mov.u32 %r1, 0;
    mov.u32 %r2, 1;
    mov.f32 %f1, 0f3F800000;
    mov.f32 %f2, 0f40000000;
    mov.f64 %d1, 0d3FF0000000000000;
    mov.b32 %r3, 0;
    mov.b16 %h1, 0;
    setp.lt.f32 %p1, %f1, %f2;
    setp.ne.b16 %p2, %h1, %h1;
    selp.b32 %r4, %r1, %r2, %p1;
    selp.b32 %r5, %r1, %r2, %p2;
    shfl.sync.bfly.b32 %r6, %r1, 1, 0x1f, 0xffffffff;
    bar.sync 0;
    bar.warp.sync 0;
{body}
    st.global.f32 [%rd3], %f1;
    ret;
}
"""

# (label, instruction text).  Every one is emitted by compile.py somewhere.
BODIES = [
    ("abs.f32", "    abs.f32 %f3, %f1;"),
    ("abs.f64", "    abs.f64 %d2, %d1;"),
    ("abs.s32", "    abs.s32 %r7, %r1;"),
    ("activemask.b32", "    activemask.b32 %r8;"),
    ("add.f32", "    add.f32 %f4, %f1, %f2;"),
    ("add.s32", "    add.s32 %r9, %r1, %r2;"),
    ("add.s64", "    add.s64 %rd5, %rd1, %rd2;"),
    ("add.u32", "    add.u32 %r10, %r1, %r2;"),
    ("and.b32", "    and.b32 %r11, %r1, %r2;"),
    ("and.pred", "    and.pred %p3, %p1, %p2;"),
    ("cp.async", "    cp.async.ca.shared.global [%rd1], [%rd2], 4;"),
    ("cp.async.commit", "    cp.async.commit_group;"),
    ("cp.async.wait", "    cp.async.wait_group 0;"),
    ("cvt.rmi.f32.f32", "    cvt.rmi.f32.f32 %f5, %f1;"),
    ("cvt.rn.f32.s32", "    cvt.rn.f32.s32 %f6, %r1;"),
    ("cvt.rzi.s32.f32", "    cvt.rzi.s32.f32 %r12, %f1;"),
    ("cvt.u32.u64", "    cvt.u32.u64 %r13, %rd1;"),
    ("div.rn.f32", "    div.rn.f32 %f7, %f1, %f2;"),
    ("div.rn.f64", "    div.rn.f64 %d3, %d1, %d1;"),
    ("div.s32", "    div.s32 %r14, %r2, %r2;"),
    ("div.u32", "    div.u32 %r15, %r2, %r2;"),
    ("ex2.approx.f32", "    ex2.approx.f32 %f8, %f1;"),
    ("fma.rn.f32", "    fma.rn.f32 %f9, %f1, %f2, %f1;"),
    ("lg2.approx.f32", "    lg2.approx.f32 %f10, %f1;"),
    ("mad.lo.u32", "    mad.lo.u32 %r16, %r1, %r2, %r1;"),
    ("max.f32", "    max.f32 %f11, %f1, %f2;"),
    ("min.u32", "    min.u32 %r17, %r1, %r2;"),
    ("mul.f32", "    mul.f32 %f12, %f1, %f2;"),
    ("mul.lo.s32", "    mul.lo.s32 %r18, %r1, %r2;"),
    ("mul.wide.s32", "    mul.wide.s32 %rd6, %r1, %r2;"),
    ("neg.f32", "    neg.f32 %f13, %f1;"),
    ("neg.f64", "    neg.f64 %d4, %d1;"),
    ("neg.s32", "    neg.s32 %r19, %r1;"),
    ("not.pred", "    not.pred %p4, %p1;"),
    ("rcp.rn.f32", "    rcp.rn.f32 %f14, %f1;"),
    ("rem.s32", "    rem.s32 %r20, %r2, %r2;"),
    ("rem.u32", "    rem.u32 %r21, %r2, %r2;"),
    ("selp.b32", "    selp.b32 %r22, %r1, %r2, %p1;"),
    ("selp.f32", "    selp.f32 %f15, %f1, %f2, %p1;"),
    ("setp.eq.f32", "    setp.eq.f32 %p1, %f1, %f2;"),
    ("setp.ge.s32", "    setp.ge.s32 %p1, %r1, %r2;"),
    ("setp.ge.u32", "    setp.ge.u32 %p1, %r1, %r2;"),
    ("setp.gt.f32", "    setp.gt.f32 %p1, %f1, %f2;"),
    ("setp.lt.f32", "    setp.lt.f32 %p1, %f1, %f2;"),
    ("setp.lt.s32", "    setp.lt.s32 %p1, %r1, %r2;"),
    ("setp.ne.b32", "    setp.ne.b32 %p1, %r1, %r2;"),
    ("setp.ne.f32", "    setp.ne.f32 %p1, %f1, %f2;"),
    ("setp.ne.f64", "    setp.ne.f64 %p1, %d1, %d1;"),
    ("setp.ne.s32", "    setp.ne.s32 %p1, %r1, %r2;"),
    ("setp.ne.u32", "    setp.ne.u32 %p1, %r1, %r2;"),
    ("shfl.sync.idx.b32", "    shfl.sync.idx.b32 %r23, %r1, 0, 0x1f, 0xffffffff;"),
    ("shr.s32", "    shr.s32 %r24, %r1, 1;"),
    ("shr.u32", "    shr.u32 %r25, %r1, 1;"),
    ("shr.u64", "    shr.u64 %rd7, %rd1, 32;"),
    ("sin.approx.f32", "    sin.approx.f32 %f16, %f1;"),
    ("cos.approx.f32", "    cos.approx.f32 %f17, %f1;"),
    ("st.global.u32", "    st.global.u32 [%rd3], %r1;"),
    ("sub.s32", "    sub.s32 %r26, %r2, %r1;"),
    ("sub.u32", "    sub.u32 %r27, %r2, %r1;"),
    ("st.global.v2.f32", "    st.global.v2.f32 [%rd3], {%f1, %f2};"),
    ("st.global.v4.f32", "    st.global.v4.f32 [%rd3], {%f1, %f2, %f1, %f2};"),
    ("ld.global.v2.f32", "    ld.global.v2.f32 {%f18, %f19}, [%rd3];"),
    ("ld.global.v4.f32", "    ld.global.v4.f32 {%f20, %f21, %f22, %f23}, [%rd3];"),
    ("ld.param.b16", "    ld.param.b16 %h2, [p0];"),
    ("mov.b64 pair (known bad)", "    mov.b64 %rd8, {%r1, %r2};"),
    ("ld.global.nc.v4.f32", "    ld.global.nc.v4.f32 {%f24, %f25, %f26, %f27}, [%rd3];"),
    ("st.global.wt.v2.f32", "    st.global.wt.v2.f32 [%rd3], {%f1, %f2};"),
    ("red.global.add.f32", "    red.global.add.f32 [%rd3], %f1;"),
    ("atom.global.add.f32", "    atom.global.add.f32 %f28, [%rd3], %f1;"),
    # The float<->float conversions: widening f32->f64 is exact, so ptxas
    # rejects a rounding modifier there, and narrowing f64->f32 demands one.
    ("cvt.f64.f32 (widen)", "    cvt.f64.f32 %d2, %f1;"),
    ("cvt.rn.f32.f64 (narrow)", "    cvt.rn.f32.f64 %f2, %d1;"),
    ("cvt.rn.f32.s32", "    cvt.rn.f32.s32 %f2, %r1;"),
    ("cvt.rn.f64.s32", "    cvt.rn.f64.s32 %d2, %r1;"),
    ("cvt.rn.f32.s64", "    cvt.rn.f32.s64 %f2, %rd1;"),
    ("cvt.rn.f64.s64", "    cvt.rn.f64.s64 %d2, %rd1;"),
    ("cvt.rzi.s32.f32", "    cvt.rzi.s32.f32 %r2, %f1;"),
    ("cvt.rzi.s32.f64", "    cvt.rzi.s32.f64 %r2, %d1;"),
    ("cvt.s32.s64", "    cvt.s32.s64 %r2, %rd1;"),
    ("cvt.u64.s32", "    cvt.u64.s32 %rd2, %r1;"),
    # Known-unsupported on this runner: ptxas reports "Not a name of any known
    # instruction".  compile.py drops SetMaxNReg instead of emitting it.
    ("setmaxnreg (expected FAIL)", "    setmaxnreg.increase.sync.aligned.u32 40;"),
]


def post(ptx: str) -> dict:
    body = json.dumps({"ptx": ptx}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=180).read())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "stderr": exc.read().decode()[:600]}


def main() -> int:
    failures = []
    for label, body in BODIES:
        result = post(HEADER.replace("{body}", body))
        if result.get("ok"):
            print(f"  ok    {label}")
            continue
        stderr = (result.get("stderr") or "").strip()
        first = stderr.splitlines()[0] if stderr else json.dumps(result)[:160]
        print(f"  FAIL  {label}: {first}")
        failures.append(label)
    print(f"\n{len(BODIES) - len(failures)}/{len(BODIES)} mnemonics assemble")
    if failures:
        print("not supported by this ptxas: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

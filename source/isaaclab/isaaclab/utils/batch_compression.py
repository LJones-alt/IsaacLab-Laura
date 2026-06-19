#!/usr/bin/env python3
"""
batch_convert.py - Convert demo_XXX recordings (RGB + depth + normals
[+ object COMs]) into reduced, depth-seeded Gaussian-splat sequences.

INPUTS: per demo, either
  .mp4 videos  (8-bit; depth brighter=closer, 0=invalid; normals (n*0.5+0.5)*255)
  .h5 files    (native arrays; float depth in metres is BEST - no quantisation
                floor, no compression halos; uint8/uint16 also handled)
plus an optional objects file giving each object's centre-of-mass per frame.

Schema discovery for .h5 is automatic (largest dataset matching the expected
shape); use --inspect ID to print the tree of a demo's files, and the
--h5-*-key flags to pin dataset names explicitly.

OUTPUT per demo: demo_XXX_splats.npz
  xyz, log_scale, quat, rgb        per-splat Gaussian parameters
  obj_id                           per-splat nearest-object label (-1 = none)
  frame_offsets, frame_indices     O(1) per-frame slicing
  obj_com (F, n_obj, 3)            object COMs per processed frame
  meta                             json (intrinsics, knobs, frame convention)

THE KNOB: --tau (mm). A patch becomes ONE splat only if flat to within tau.
With float depth there is no quantisation floor; with 8-bit depth the floor is
(far-near)/255 and is printed at startup.

Anti-interference: flying-pixel filter (auto: ON for 8-bit sources, OFF for
float), depth-jump split, normal-coherence split.

Frames: splats are in the CAMERA frame by default (x right, y down, z forward).
--world-frame bakes the camera extrinsics (--cam-pos/--cam-quat, defaults from
the Isaac Lab table_cam config) so splats and COMs land in the env frame.
"""

import argparse, json, os, time
import numpy as np
import cv2

C0 = 0.28209479177387814
MIN_THICK = 1.5e-3


# ------------------------------------------------------------------ small math
def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]], np.float32)


def quat_mul(q1, q2):
    """Hamilton product, (w,x,y,z). q2 may be (N,4)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1).astype(np.float32)


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2,1]-R[1,2])/s; y = (R[0,2]-R[2,0])/s; z = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=0.25*s
        y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s
        x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s
        x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    q = np.array([w, x, y, z], np.float32)
    return q / (np.linalg.norm(q) + 1e-12)


# ------------------------------------------------------------------ decoding (8-bit)
def decode_depth_u8(frame_gray_u8, a):
    if a.near is None or a.far is None:
        raise SystemExit("uint8 depth requires --near and --far (metres at values 255 and 0)")
    v = frame_gray_u8.astype(np.float32)
    valid = v > 0
    c = v / 255.0
    depth = (a.near + c*(a.far-a.near)) if a.no_invert else (a.far - c*(a.far-a.near))
    depth[~valid] = 0.0
    return depth, valid


def decode_normals_u8(frame_rgb_u8):
    n = frame_rgb_u8.astype(np.float32)/127.5 - 1.0
    ln = np.linalg.norm(n, axis=2)
    nvalid = ln > 0.5
    n[nvalid] /= ln[nvalid][:, None]
    return n, nvalid


def flying_pixel_filter(depth, valid, jump):
    big = 1e6
    dmax = cv2.dilate(np.where(valid, depth, -big).astype(np.float32), np.ones((3,3), np.uint8))
    dmin = -cv2.dilate(np.where(valid, -depth, -big).astype(np.float32), np.ones((3,3), np.uint8))
    return valid & ((dmax - dmin) < jump)


def backproject(depth, fx, fy, cx, cy):
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth
    return np.stack([(uu-cx)*z/fx, (vv-cy)*z/fy, z], axis=-1)


# ------------------------------------------------------------------ input loaders
def _demo_paths(d, pid, a):
    base = os.path.join(d, f"{a.prefix}{pid:03d}")
    p = [base + a.rgb_suffix, base + a.depth_suffix, base + a.normals_suffix]
    if a.objects_suffix:
        p.append(base + a.objects_suffix)
    return p


def _h5_find(f, pred, key, what):
    if key:
        return f[key][:]
    hits = []
    f.visititems(lambda name, obj: hits.append((name, obj))
                 if hasattr(obj, "shape") and pred(obj.shape) else None)
    if not hits:
        raise SystemExit(f"no dataset matching {what} found; use --inspect / --h5-{what}-key")
    name, ds = max(hits, key=lambda h: int(np.prod(h[1].shape)))
    return ds[:]


def _maybe_K(f):
    hits = []
    def v(n, o):
        sh = getattr(o, "shape", None)
        if sh and len(sh) >= 2 and tuple(sh[-2:]) == (3, 3):
            hits.append((n, o))
        elif sh and "intrinsic" in n.lower():
            hits.append((n, o))
    f.visititems(v)
    if not hits:
        return None, None
    n, o = hits[0]
    K = np.asarray(o[:], np.float32).reshape(-1, 3, 3)[0]
    return n, K


def load_h5_demo(paths, a):
    import h5py
    out = {"K": None, "K_src": ""}
    with h5py.File(paths[0], "r") as f:
        out["rgb"] = _h5_find(f, lambda s: len(s) == 4 and s[-1] == 3, a.h5_rgb_key, "rgb")
        out["K_src"], out["K"] = _maybe_K(f)
    with h5py.File(paths[1], "r") as f:
        out["depth"] = _h5_find(f, lambda s: len(s) in (3, 4), a.h5_depth_key, "depth")
        if out["K"] is None:
            out["K_src"], out["K"] = _maybe_K(f)
        if out["depth"].ndim == 4:
            out["depth"] = out["depth"][..., 0]
    with h5py.File(paths[2], "r") as f:
        out["normals"] = _h5_find(f, lambda s: len(s) == 4 and s[-1] == 3, a.h5_normals_key, "normals")
    out["com"], out["obj_names"] = None, []
    if len(paths) > 3 and os.path.exists(paths[3]):
        with h5py.File(paths[3], "r") as f:
            if a.h5_objects_key:
                out["com"] = f[a.h5_objects_key][:]
            else:
                hits = []
                f.visititems(lambda n, o: hits.append((n, o)) if hasattr(o, "shape") else None)
                tn3 = [(n, o) for n, o in hits if len(o.shape) == 3 and o.shape[-1] == 3]
                t3  = [(n, o) for n, o in hits if len(o.shape) == 2 and o.shape[-1] == 3]
                if tn3:
                    out["com"] = tn3[0][1][:]
                elif t3:                                  # group of (T,3) per object
                    out["com"] = np.stack([o[:] for _, o in t3], axis=1)
                    out["obj_names"] = [n for n, _ in t3]
    return out


def frame_iter_h5(data, a):
    """Yield (rgb u8, depth m, valid, nrm, nvalid, com) at native resolution."""
    rgb_a, dep_a, nrm_a = data["rgb"], data["depth"], data["normals"]
    T = min(len(rgb_a), len(dep_a), len(nrm_a))
    depth_is_u8 = dep_a.dtype == np.uint8
    for t in range(T):
        rgb = rgb_a[t]
        rgb = (np.clip(rgb, 0, 1)*255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        d = dep_a[t]
        if depth_is_u8:
            depth, valid = decode_depth_u8(d, a)
        elif d.dtype == np.uint16:
            depth = d.astype(np.float32)/a.depth_scale
            hi = a.far if a.far else np.inf
            valid = (depth > 1e-3) & (depth < hi)
        else:
            depth = d.astype(np.float32)
            hi = a.far if a.far else np.inf
            valid = np.isfinite(depth) & (depth > 1e-3) & (depth < hi)
            depth = np.where(valid, depth, 0.0)
        n = nrm_a[t]
        if n.dtype == np.uint8:
            nrm, nvalid = decode_normals_u8(n)
        else:
            nrm = n.astype(np.float32)
            if nrm.min() >= -0.01:          # stored encoded as (n*0.5+0.5), even as float
                nrm = nrm*2.0 - 1.0
            ln = np.linalg.norm(nrm, axis=2)
            nvalid = ln > 0.5
            nrm[nvalid] /= ln[nvalid][:, None]
        com = data["com"][t] if data["com"] is not None else None
        yield rgb, depth, valid, nrm, nvalid, com


def frame_iter_mp4(paths, a, W, H):
    caps = [cv2.VideoCapture(p) for p in paths[:3]]
    while True:
        ok1, fr = caps[0].read(); ok2, fd = caps[1].read(); ok3, fn = caps[2].read()
        if not (ok1 and ok2 and ok3):
            break
        if a.native:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            fd = cv2.resize(fd, (W, H), interpolation=cv2.INTER_NEAREST)
            fn = cv2.resize(fn, (W, H), interpolation=cv2.INTER_NEAREST)
        depth, valid = decode_depth_u8(fd[:, :, 0], a)
        nrm, nvalid = decode_normals_u8(cv2.cvtColor(fn, cv2.COLOR_BGR2RGB))
        yield cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), depth, valid, nrm, nvalid, None
    for c in caps:
        c.release()


# ------------------------------------------------------------------ the reduction
def fit_frame(P, valid, rgb, nrm, nvalid, a, jump):
    H, W, _ = P.shape
    tau = a.tau / 1000.0
    cos_ang = np.cos(np.radians(a.angle))
    mu, scale, quat, col = [], [], [], []
    stack = [(0, 0, H, W)]
    while stack:
        r0, c0, r1, c1 = stack.pop()
        m = valid[r0:r1, c0:c1]
        k = int(m.sum())
        if k == 0:
            continue
        ch, cw = r1 - r0, c1 - c0
        too_small = ch <= a.min_cell or cw <= a.min_cell
        pts = P[r0:r1, c0:c1][m]
        if k < 6:
            mu.append(pts.mean(0)); scale.append([MIN_THICK]*3)
            quat.append([1, 0, 0, 0]); col.append(rgb[r0:r1, c0:c1][m].mean(0))
            continue
        z = pts[:, 2]
        jump_ok = (z.max() - z.min()) < jump if not a.no_jump else True
        if a.no_normals:
            ang_ok = True
        else:
            nm = nvalid[r0:r1, c0:c1] & m
            ang_ok = (np.linalg.norm(nrm[r0:r1, c0:c1][nm].mean(0)) >= cos_ang
                      if nm.sum() > 0.5*k else True)
        c = pts.mean(0); X = pts - c
        w, V = np.linalg.eigh((X.T @ X) / k)
        flat_ok = np.sqrt(max(w[0], 0.0)) <= tau
        if (flat_ok and jump_ok and ang_ok) or too_small:
            s = np.sqrt(np.clip(w, 0, None))
            sc = [max(s[i]*a.fill, MIN_THICK) for i in range(3)]
            R = V.copy()
            if np.linalg.det(R) < 0:
                R[:, 0] = -R[:, 0]
            mu.append(c); scale.append(sc); quat.append(rotmat_to_quat(R))
            col.append(rgb[r0:r1, c0:c1][m].mean(0))
        else:
            mr, mc = (r0+r1)//2, (c0+c1)//2
            stack += [(r0,c0,mr,mc),(r0,mc,mr,c1),(mr,c0,r1,mc),(mr,mc,r1,c1)]
    mu = np.asarray(mu, np.float32)
    return (mu, np.log(np.asarray(scale, np.float32)),
            np.asarray(quat, np.float32), np.asarray(col, np.float32)/255.0)


# ------------------------------------------------------------------ ply
def write_ply(path, xyz, rgb01, log_scale, quat, opacity=0.1):
    names = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2",
             "opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    N = len(xyz)
    f_dc = (rgb01 - 0.5)/C0
    op = np.full((N,1), np.log(opacity/(1-opacity)), np.float32)
    data = np.concatenate([xyz, np.zeros((N,3),np.float32), f_dc, op, log_scale, quat], 1).astype(np.float32)
    arr = np.empty(N, dtype=[(n,"<f4") for n in names])
    for i, n in enumerate(names):
        arr[n] = data[:, i]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {N}\n" +
                 "".join(f"property float {n}\n" for n in names) + "end_header\n").encode())
        arr.tofile(f)


# ------------------------------------------------------------------ per-demo job
def convert_demo(pid, a, data=None, tag=None):
    cv2.setNumThreads(1)
    name = tag if tag is not None else f"{a.prefix}{pid:03d}"
    out_npz = os.path.join(a.out, f"{name}_splats.npz")
    if os.path.exists(out_npz) and not a.overwrite:
        return pid, "skip (exists)"
    if data is None:
        paths = _demo_paths(a.dir, pid, a)
        if not all(os.path.exists(p) for p in paths[:3]):
            return pid, f"MISSING ({os.path.basename(paths[0])} ...)"
        is_h5 = paths[0].endswith((".h5", ".hdf5"))
    else:
        is_h5 = True
    obj_names = []
    if is_h5:
        if data is None:
            data = load_h5_demo(paths, a)
        obj_names = data["obj_names"]
        H, W = data["depth"].shape[1:3]
        depth_u8 = data["depth"].dtype == np.uint8
        frames = frame_iter_h5(data, a)
    else:
        cap = cv2.VideoCapture(paths[0])
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if a.native:
            W, H = a.native
        depth_u8 = True
        frames = frame_iter_mp4(paths, a, W, H)

    K = data.get("K") if is_h5 else None
    if a.fx:
        fx, fy = a.fx, (a.fy or a.fx); cx, cy = W/2.0, H/2.0; intr = "flag"
    elif K is not None:
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        intr = f"h5:{data['K_src']}"
    elif a.fov:
        fx = W/(2*np.tan(np.radians(a.fov)/2)); fy = a.fy or fx
        cx, cy = W/2.0, H/2.0; intr = "fov"
    else:
        fx = W*a.focal_length/a.aperture; fy = a.fy or fx
        cx, cy = W/2.0, H/2.0; intr = "focal/aperture"
    quant = ((a.far - a.near)/255.0) if (depth_u8 and a.near is not None and a.far is not None) else 0.0
    jump = (a.jump_mm/1000.0) if a.jump_mm else max(3*a.tau/1000.0, 2.5*quant, 0.004)
    use_flying = a.flying or (depth_u8 and not a.no_flying)

    R_cam = quat_to_rotmat(np.array(a.cam_quat, np.float32))     # cam axes in env
    t_cam = np.array(a.cam_pos, np.float32)
    q_cam = np.array(a.cam_quat, np.float32)

    XYZ, SC, QT, RGB, OID, COMS = [], [], [], [], [], []
    offsets, fidx, dense = [0], [], []
    t0 = time.perf_counter()
    for fi, (rgb, depth, valid, nrm, nvalid, com) in enumerate(frames):
        if fi % a.stride:
            continue
        if use_flying:
            valid = flying_pixel_filter(depth, valid, jump)
        P = backproject(depth, fx, fy, cx, cy)
        xyz, lsc, qt, colr = fit_frame(P, valid, rgb, nrm, nvalid, a, jump)

        if com is not None:
            com = np.asarray(com, np.float32).reshape(-1, 3)
            com_cam = (com - t_cam) @ R_cam if a.com_frame == "env" else com
            d2 = ((xyz[:, None, :] - com_cam[None, :, :])**2).sum(-1)
            oid = np.where(d2.min(1) < a.obj_radius**2, d2.argmin(1), -1).astype(np.int16)
        else:
            com_cam = np.zeros((0, 3), np.float32)
            oid = np.full(len(xyz), -1, np.int16)

        if a.world_frame:
            xyz = xyz @ R_cam.T + t_cam
            qt = quat_mul(q_cam, qt)
            com_store = com if com is not None else com_cam
        else:
            com_store = com_cam

        XYZ.append(xyz); SC.append(lsc); QT.append(qt); RGB.append(colr); OID.append(oid)
        COMS.append(com_store)
        offsets.append(offsets[-1] + len(xyz)); fidx.append(fi); dense.append(int(valid.sum()))
        if a.ply_first and len(fidx) == 1:
            write_ply(os.path.join(a.out, f"{name}_f{fi:04d}.ply"), xyz, colr, lsc, qt)

    if not fidx:
        return pid, "no frames"
    mean_splats = float(np.mean(np.diff(offsets)))
    n_obj = max((len(c) for c in COMS), default=0)
    com_arr = (np.stack([np.pad(c, ((0, n_obj-len(c)), (0, 0)), constant_values=np.nan)
                         for c in COMS]) if n_obj else np.zeros((len(fidx), 0, 3), np.float32))
    meta = dict(tau_mm=a.tau, angle_deg=a.angle, near=a.near, far=a.far,
                intrinsics_source=intr, fx=fx, fy=fy,
                cx=cx, cy=cy, width=W, height=H, stride=a.stride, quant_step_m=quant,
                depth_dtype=("uint8" if depth_u8 else "float"), source=("h5" if is_h5 else "mp4"),
                flying=bool(use_flying), frame=("env" if a.world_frame else "camera"),
                obj_names=obj_names, obj_radius=a.obj_radius, opacity=0.1,
                frames=len(fidx), mean_splats=mean_splats,
                mean_reduction=float(np.mean(dense)/max(mean_splats, 1)))
    np.savez_compressed(out_npz, xyz=np.concatenate(XYZ), log_scale=np.concatenate(SC),
                        quat=np.concatenate(QT), rgb=np.concatenate(RGB),
                        obj_id=np.concatenate(OID), obj_com=com_arr.astype(np.float32),
                        frame_offsets=np.asarray(offsets, np.int64),
                        frame_indices=np.asarray(fidx, np.int32), meta=json.dumps(meta))
    dt = time.perf_counter() - t0
    return pid, (f"{len(fidx)} frames, mean {mean_splats:.0f} splats "
                 f"({meta['mean_reduction']:.0f}x), {dt:.1f}s ({dt/len(fidx)*1000:.0f} ms/frame), "
                 f"{W}x{H} fx={fx:.1f} [{intr}]"
                 + (f", {n_obj} objects" if n_obj else ""))



def load_robomimic_demo(f, demo, a):
    """Converter data-dict from a robomimic-style file: data/<demo>/obs/*."""
    g = f[f"data/{demo}/obs"]
    rgb = g[a.obs_rgb_key][:]
    depth = g[a.obs_depth_key][:]
    if depth.ndim == 4:
        depth = depth[..., 0]
    nrm = g[a.obs_normals_key][:]
    com, names = None, []
    if a.obs_com_key:
        try:
            com = np.asarray(g[a.obs_com_key][:], np.float32).reshape(len(rgb), -1, 3)
            names = [a.obs_com_key]
        except Exception:
            pass
    return dict(rgb=rgb, depth=depth, normals=nrm, com=com,
                obj_names=names, K=None, K_src="")


def run_single_file(a):
    import h5py
    with h5py.File(a.single_file, "r") as f:
        names = list(f["data"].keys())
        idx_of = lambda n: int(n.split("_")[-1]) if n.split("_")[-1].isdigit() else -1
        names.sort(key=idx_of)
        for nm in names:
            i = idx_of(nm)
            if (a.start, a.end) != (1, 999) and not (a.start <= i <= a.end):
                continue
            data = load_robomimic_demo(f, nm, a)
            _, msg = convert_demo(max(i, 0), a, data=data, tag=nm)
            print(f"{nm}: {msg}")


def inspect_demo(pid, a):
    import h5py
    for p in _demo_paths(a.dir, pid, a):
        print(f"=== {os.path.basename(p)} ===")
        if not os.path.exists(p):
            print("  (missing)"); continue
        with h5py.File(p, "r") as f:
            f.visititems(lambda n, o: print(f"  {n:30s} {getattr(o,'shape','-')} {getattr(o,'dtype','')}"))


def build_parser():
    p = argparse.ArgumentParser(description="Batch RGB-D(+normals,+COMs) -> reduced Gaussian splats",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dir", default=None); p.add_argument("--out", default="splats")
    p.add_argument("--single-file", default=None,
                   help="robomimic-style HDF5 (data/demo_N/obs/*) instead of --dir")
    p.add_argument("--obs-rgb-key", default="table_cam")
    p.add_argument("--obs-depth-key", default="table_cam_depth")
    p.add_argument("--obs-normals-key", default="table_cam_normals")
    p.add_argument("--obs-com-key", default="object_position",
                   help="empty string disables COM labels")
    p.add_argument("--start", type=int, default=1); p.add_argument("--end", type=int, default=999)
    p.add_argument("--tau", type=float, default=8.0, help="THE compression knob, mm")
    p.add_argument("--angle", type=float, default=12.0)
    p.add_argument("--min-cell", type=int, default=3)
    p.add_argument("--fill", type=float, default=2.0)
    p.add_argument("--near", type=float, default=None,
                   help="8-bit depth only: metres at value 255 (required for uint8 sources)")
    p.add_argument("--far", type=float, default=None,
                   help="8-bit depth: metres at value 0 (required for uint8); optional validity clip for float")
    p.add_argument("--no-invert", action="store_true")
    p.add_argument("--depth-scale", type=float, default=1000.0, help="uint16 depth divisor")
    p.add_argument("--focal-length", type=float, default=24.0,
                   help="camera focal length (Isaac PinholeCameraCfg units)")
    p.add_argument("--aperture", type=float, default=20.955,
                   help="horizontal aperture (same units); fx = W * focal / aperture at any W")
    p.add_argument("--fov", type=float, default=None, help="override: horizontal FOV in degrees")
    p.add_argument("--fx", type=float, default=None, help="override: fx in pixels")
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--native", type=int, nargs=2, default=None, metavar=("W","H"))
    p.add_argument("--jump-mm", type=float, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--overwrite", action="store_true"); p.add_argument("--ply-first", action="store_true")
    p.add_argument("--no-normals", action="store_true"); p.add_argument("--no-jump", action="store_true")
    p.add_argument("--no-flying", action="store_true"); p.add_argument("--flying", action="store_true",
                   help="force flying-pixel filter ON (default: auto by depth dtype)")
    p.add_argument("--prefix", default="demo_")
    p.add_argument("--rgb-suffix", default="_rgb.h5"); p.add_argument("--depth-suffix", default="_depth.h5")
    p.add_argument("--normals-suffix", default="_normals.h5")
    p.add_argument("--objects-suffix", default="_objects.h5", help="'' to disable")
    p.add_argument("--h5-rgb-key", default=None); p.add_argument("--h5-depth-key", default=None)
    p.add_argument("--h5-normals-key", default=None); p.add_argument("--h5-objects-key", default=None)
    p.add_argument("--cam-pos", type=float, nargs=3, default=[1.4, 0.0, 0.5])
    p.add_argument("--cam-quat", type=float, nargs=4, default=[0.35355, -0.61237, -0.61237, 0.35355],
                   help="(w x y z), ROS convention, camera->env")
    p.add_argument("--com-frame", choices=["env", "camera"], default="env")
    p.add_argument("--obj-radius", type=float, default=0.15, help="COM labelling radius, m")
    p.add_argument("--world-frame", action="store_true", help="output splats in env frame")
    p.add_argument("--inspect", type=int, default=None, metavar="ID",
                   help="print the HDF5 tree of one demo and exit")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.inspect is not None:
        inspect_demo(a.inspect, a); raise SystemExit
    os.makedirs(a.out, exist_ok=True)
    if a.single_file:
        run_single_file(a); raise SystemExit
    if not a.dir:
        raise SystemExit("provide --dir (per-demo files) or --single-file (robomimic layout)")
    ids = list(range(a.start, a.end + 1))
    if a.rgb_suffix.endswith((".h5", ".hdf5")):
        print("h5 mode: W/H per demo from the arrays; intrinsics: --fx > h5 matrix > --fov > focal/aperture")
    elif a.near is not None and a.far is not None:
        print(f"8-bit depth quantisation step ~{(a.far-a.near)/255*1000:.1f} mm")
    if a.workers > 1 and len(ids) > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for pid, msg in pool.starmap(convert_demo, [(i, a) for i in ids]):
                print(f"demo_{pid:03d}: {msg}")
    else:
        for i in ids:
            pid, msg = convert_demo(i, a)
            print(f"demo_{pid:03d}: {msg}")
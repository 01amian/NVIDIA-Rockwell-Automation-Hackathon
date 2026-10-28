import argparse, json, os, random, subprocess, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cam_info = json.load(open(os.path.join(HERE, "cameras", "CapSealCam.json")))
stage_path = os.path.join(HERE, "stations", "tray_inspection", "tray_inspection.usdc")
rw, rh = cam_info["render_px"]

out_dir = os.path.join(HERE, "synthetic_100")
crops_dir = os.path.join(HERE, "synthetic_100", "crops_calibrated")
os.makedirs(os.path.join(crops_dir, "damaged_vial"), exist_ok=True)

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh})

try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import omni.usd, omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_path):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(5):
        kit.update()

    stage = ctx.get_stage()
    prim_path = cam_info["prim_in_station_file"]

    clip = UsdGeom.Camera(stage.GetPrimAtPath(prim_path)).GetClippingRangeAttr()
    near_far = clip.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
    kit.update()
    clip.Set(near_far)
    kit.update()

    rp = rep.create.render_product(prim_path, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    print("Warming up RTX sensor...", flush=True)
    for _ in range(12):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        d = np.asarray(ann.get_data())
        if d.size > 0 and d.ndim == 3 and d.mean() > 5.0:
            print("Sensor warm-up complete!", flush=True)
            break

    coords = {
        'G01': (377, 162), 'G02': (378, 198), 'G03': (380, 233), 'G04': (380, 268), 'G05': (382, 305),
        'G06': (348, 163), 'G07': (349, 199), 'G08': (350, 234), 'G09': (351, 270), 'G10': (352, 306),
        'G11': (318, 164), 'G12': (319, 200), 'G13': (320, 236), 'G14': (322, 271), 'G15': (322, 307),
        'G16': (287, 164), 'G17': (288, 201), 'G18': (290, 237), 'G19': (291, 273), 'G20': (294, 309),
        'G21': (256, 166), 'G22': (258, 202), 'G23': (259, 239), 'G24': (261, 274), 'G25': (262, 311),
        'G26': (225, 167), 'G27': (227, 204), 'G28': (229, 240), 'G29': (230, 276), 'G30': (232, 311)
    }
    good_slots = list(coords.keys())
    random.seed(1337)

    total_scenes = 15
    damaged_crops_saved = 0

    for sc_idx in range(1, total_scenes + 1):
        sc_name = f"damaged_scene_{sc_idx:02d}"
        num_damaged = random.randint(3, 6)
        dv_slots = random.sample(good_slots, num_damaged)
        
        defects = {s: "damaged_vial" for s in dv_slots}
        remaining = [s for s in good_slots if s not in defects]
        for s in random.sample(remaining, random.randint(1, 2)):
            defects[s] = random.choice(["crooked_cap", "missing_cap", "no_vial"])

        modified_prims = []
        for slot, cond in defects.items():
            cap_prim = stage.GetPrimAtPath(f"/TrayInspection/{slot}_cap")
            tube_prim = stage.GetPrimAtPath(f"/TrayInspection/{slot}_tube")

            if cond == "no_vial":
                if cap_prim.IsValid():
                    UsdGeom.Imageable(cap_prim).MakeInvisible()
                    modified_prims.append((cap_prim, "vis"))
                if tube_prim.IsValid():
                    UsdGeom.Imageable(tube_prim).MakeInvisible()
                    modified_prims.append((tube_prim, "vis"))
            elif cond == "missing_cap":
                if cap_prim.IsValid():
                    UsdGeom.Imageable(cap_prim).MakeInvisible()
                    modified_prims.append((cap_prim, "vis"))
                if tube_prim.IsValid():
                    UsdGeom.Imageable(tube_prim).MakeVisible()
            elif cond == "crooked_cap":
                if tube_prim.IsValid():
                    UsdGeom.Imageable(tube_prim).MakeVisible()
                if cap_prim.IsValid():
                    UsdGeom.Imageable(cap_prim).MakeVisible()
                    xform = UsdGeom.Xformable(cap_prim)
                    rot_op = next((op for op in xform.GetOrderedXformOps() if "rotate" in op.GetOpName().lower()), None)
                    if not rot_op:
                        rot_op = xform.AddRotateXYZOp()
                    rot_op.Set(Gf.Vec3f(random.uniform(16.0, 22.0) * random.choice([1, -1]),
                                          random.uniform(6.0, 14.0) * random.choice([1, -1]), 0.0))
                    modified_prims.append((rot_op, "rot"))
            elif cond == "damaged_vial":
                if tube_prim.IsValid():
                    UsdGeom.Imageable(tube_prim).MakeVisible()
                    xform_tube = UsdGeom.Xformable(tube_prim)
                    rot_t = next((op for op in xform_tube.GetOrderedXformOps() if "rotate" in op.GetOpName().lower()), None)
                    if not rot_t:
                        rot_t = xform_tube.AddRotateXYZOp()
                    rot_t.Set(Gf.Vec3f(random.uniform(14.0, 22.0) * random.choice([1, -1]),
                                       random.uniform(10.0, 18.0) * random.choice([1, -1]), -5.0))
                    modified_prims.append((rot_t, "rot"))
                    scale_t = next((op for op in xform_tube.GetOrderedXformOps() if "scale" in op.GetOpName().lower()), None)
                    if not scale_t:
                        scale_t = xform_tube.AddScaleOp()
                    scale_t.Set(Gf.Vec3f(random.uniform(1.3, 1.55), random.uniform(0.6, 0.75), 0.85))
                    modified_prims.append((scale_t, "scale"))
                if cap_prim.IsValid():
                    UsdGeom.Imageable(cap_prim).MakeVisible()
                    xform_cap = UsdGeom.Xformable(cap_prim)
                    rot_c = next((op for op in xform_cap.GetOrderedXformOps() if "rotate" in op.GetOpName().lower()), None)
                    if not rot_c:
                        rot_c = xform_cap.AddRotateXYZOp()
                    rot_c.Set(Gf.Vec3f(random.uniform(10.0, 16.0) * random.choice([1, -1]),
                                       random.uniform(8.0, 15.0) * random.choice([1, -1]), 0.0))
                    modified_prims.append((rot_c, "rot"))

        # Render frame
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        data = np.asarray(ann.get_data())
        if data.size == 0 or data.mean() <= 5.0:
            for _ in range(5):
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
                data = np.asarray(ann.get_data())
                if data.size > 0 and data.mean() > 5.0:
                    break

        img = data[..., :3].copy()
        raw_png = os.path.join(out_dir, f"{sc_name}_raw.png")
        final_png = os.path.join(out_dir, f"{sc_name}.png")
        cv2.imwrite(raw_png, img[..., ::-1])

        subprocess.run([sys.executable, os.path.join(HERE, "tools", "apply_lens.py"),
                        raw_png, os.path.join(HERE, "cameras", "CapSealCam.json"), final_png], check=True)
        if os.path.exists(raw_png):
            os.remove(raw_png)

        # Extract damaged crops
        distorted = cv2.imread(final_png)
        half = 16
        for slot_id in dv_slots:
            cx, cy = coords[slot_id]
            crop = distorted[cy - half:cy + half, cx - half:cx + half]
            if crop.shape[:2] == (32, 32):
                crop_path = os.path.join(crops_dir, "damaged_vial", f"{sc_name}_{slot_id}.jpg")
                cv2.imwrite(crop_path, crop)
                damaged_crops_saved += 1

        # Reset prims
        for target, mod_type in modified_prims:
            if mod_type == "vis":
                UsdGeom.Imageable(target).MakeVisible()
            elif mod_type == "rot":
                target.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            elif mod_type == "scale":
                target.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        kit.update()
        print(f"[{sc_idx}/{total_scenes}] Rendered {sc_name} with {len(dv_slots)} damaged vials. Total crops so far: {damaged_crops_saved}", flush=True)

    print(f"\nSUCCESS: Generated {total_scenes} scenes and saved {damaged_crops_saved} damaged_vial crops!", flush=True)

finally:
    kit.close()

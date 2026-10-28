import argparse, json, os, random, subprocess, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cam_info = json.load(open(os.path.join(HERE, "cameras", "CapSealCam.json")))
stage_path = os.path.join(HERE, "stations", "tray_inspection", "tray_inspection.usdc")
rw, rh = cam_info["render_px"]

out_dir = os.path.join(HERE, "synthetic_100")
crops_dir = os.path.join(HERE, "synthetic_100", "crops_calibrated")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(crops_dir, exist_ok=True)

for c in ["pass", "no_vial", "missing_cap", "crooked_cap", "damaged_vial"]:
    os.makedirs(os.path.join(crops_dir, c), exist_ok=True)

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

    print("Warming up RTX sensor & compiling shaders...", flush=True)
    for _ in range(25):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        d = np.asarray(ann.get_data())
        if d.size > 0 and d.ndim == 3 and d.mean() > 5.0:
            print(f"RTX sensor ready! Warmup mean brightness = {d.mean():.1f}", flush=True)
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
    random.seed(42)

    total_scenes = 100
    crop_counts = {c: 0 for c in ["pass", "no_vial", "missing_cap", "crooked_cap", "damaged_vial"]}

    print(f"\n==========================================", flush=True)
    print(f"STARTING 100-SCENE BENCHMARK DATASET GENERATION", flush=True)
    print(f"==========================================\n", flush=True)

    for sc_idx in range(1, total_scenes + 1):
        scene_name = f"scene_{sc_idx:03d}"
        defects = {}

        if sc_idx <= 20:
            # 20 Pristine Baseline Scenes (All Pass)
            pass
        elif sc_idx <= 40:
            # 20 Missing Cap Scenes (1 to 3 missing caps)
            for s in random.sample(good_slots, random.randint(1, 3)):
                defects[s] = "missing_cap"
        elif sc_idx <= 60:
            # 20 Empty Position Scenes (1 to 4 empty slots)
            for s in random.sample(good_slots, random.randint(1, 4)):
                defects[s] = "no_vial"
        elif sc_idx <= 80:
            # 20 Crooked Cap Scenes (1 to 3 crooked caps)
            for s in random.sample(good_slots, random.randint(1, 3)):
                defects[s] = "crooked_cap"
        else:
            # 20 Damaged Vial & Multi-Defect Edge Cases
            for s in random.sample(good_slots, random.randint(2, 4)):
                defects[s] = "damaged_vial"
            rem = [s for s in good_slots if s not in defects]
            for s in random.sample(rem, random.randint(1, 2)):
                defects[s] = random.choice(["crooked_cap", "missing_cap", "no_vial"])

        ground_truth = {"slots": []}
        for s in good_slots:
            cls_name = defects.get(s, "pass")
            ground_truth["slots"].append({"id": s, "slot_id": s, "class": cls_name, "classification": cls_name})

        # Apply USD prim overrides
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
            for _ in range(8):
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
                data = np.asarray(ann.get_data())
                if data.size > 0 and data.mean() > 5.0:
                    break

        img = data[..., :3].copy()
        raw_png = os.path.join(out_dir, f"{scene_name}_raw.png")
        final_png = os.path.join(out_dir, f"{scene_name}.png")
        gt_json = os.path.join(out_dir, f"{scene_name}_gt.json")
        cv2.imwrite(raw_png, img[..., ::-1])

        subprocess.run([sys.executable, os.path.join(HERE, "tools", "apply_lens.py"),
                        raw_png, os.path.join(HERE, "cameras", "CapSealCam.json"), final_png], check=True)
        if os.path.exists(raw_png):
            os.remove(raw_png)

        with open(gt_json, "w") as f:
            json.dump(ground_truth, f, indent=2)

        # Extract calibrated crops directly from the distorted image
        distorted = cv2.imread(final_png)
        half = 16
        for slot_id, (cx, cy) in coords.items():
            cls_name = defects.get(slot_id, "pass")
            crop = distorted[cy - half:cy + half, cx - half:cx + half]
            if crop.shape[:2] == (32, 32):
                crop_path = os.path.join(crops_dir, cls_name, f"{scene_name}_{slot_id}.jpg")
                cv2.imwrite(crop_path, crop)
                crop_counts[cls_name] += 1

        # Reset prims
        for target, mod_type in modified_prims:
            if mod_type == "vis":
                UsdGeom.Imageable(target).MakeVisible()
            elif mod_type == "rot":
                target.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            elif mod_type == "scale":
                target.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        kit.update()

        if sc_idx % 10 == 0 or sc_idx == total_scenes:
            print(f"[*] Generated {sc_idx:03d}/{total_scenes} scenes... Current crop totals: {crop_counts}", flush=True)

    print(f"\n==========================================", flush=True)
    print(f"SUCCESS: All {total_scenes} benchmark scenes generated cleanly!", flush=True)
    print(f"Final Crop Totals: {crop_counts}", flush=True)
    print(f"==========================================\n", flush=True)

finally:
    kit.close()

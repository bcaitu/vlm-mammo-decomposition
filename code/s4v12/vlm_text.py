"""Training/inference text for the VLM study (paper 4).
Target report = the templated report of cbmammo.reports.render (style 0) built from REFERENCE labels, with a fixed slot order. Slots without a reference label are written as the
placeholder UNK and are excluded from the loss (segment flag False), so the model is never taught to omit a slot because a dataset lacks that label.
build_segments(values, side) -> list of (text, supervised)."""
from cbmammo.reports import DENSITY_TXT, MARGIN_TXT, MORPH_TXT, BIRADS_TXT

UNK = "UNK"

def _merge(segs):
    out = []
    for t, s in segs:
        if not t: continue
        if out and out[-1][1] == s: out[-1] = (out[-1][0] + t, s)
        else: out.append((t, s))
    return out

def build_segments(v, side=None):
    sb = {"L": "left breast", "R": "right breast"}.get(side or "", "breast")
    segs = []
    d = v.get("density")
    segs.append((DENSITY_TXT[d], True) if d else (f"Density: {UNK}.", False))
    mass, shape, margin = v.get("mass"), v.get("mass_shape"), v.get("mass_margin")
    if mass is None and (shape or margin): mass = "present"
    segs.append((" ", True))
    if mass == "present":
        art = ("an" if shape[0] in "aeiou" else "a") if shape else "a"
        segs += [("There is ", True), (art, bool(shape)), (" ", True), (shape if shape else UNK, bool(shape)), (" mass", True)]
        segs += [(" with ", True), (MARGIN_TXT[margin] if margin else UNK, bool(margin)), (" margins", True)]
        segs += [(f" in the {sb}.", True)]
    elif mass == "absent": segs.append(("No suspicious mass is seen.", True))
    else: segs.append((f"Mass: {UNK}.", False))
    calc, morph, dist = v.get("calc"), v.get("calc_morphology"), v.get("calc_distribution")
    if calc is None and (morph or dist): calc = "present"
    segs.append((" ", True))
    if calc == "present":
        mt = MORPH_TXT[morph] if morph else UNK
        mt = mt[0].upper() + mt[1:]
        segs += [(mt, bool(morph)), (" calcifications in a ", True), (dist if dist else UNK, bool(dist)), (" distribution are present.", True)]
    elif calc == "absent": segs.append(("No calcifications are seen.", True))
    else: segs.append((f"Calcifications: {UNK}.", False))
    segs.append((" ", True))
    a = v.get("asymmetry")
    segs.append(("An asymmetry is noted.", True) if a == "present" else ("No asymmetry.", True) if a == "absent" else (f"Asymmetry: {UNK}.", False))
    segs.append((" ", True))
    x = v.get("distortion")
    segs.append(("Architectural distortion is present.", True) if x == "present" else ("No architectural distortion.", True) if x == "absent" else (f"Distortion: {UNK}.", False))
    segs.append((" ", True))
    b = v.get("birads")
    segs.append(("Impression: " + BIRADS_TXT[b], True) if b else (f"Impression: BI-RADS {UNK}.", False))
    return _merge(segs)

def text_of(segs): return "".join(t for t, _ in segs)

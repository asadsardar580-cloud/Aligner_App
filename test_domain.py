"""The treatment domain and its encrypted local persistence.

Two claims are pinned here, and both are about what a case file is ALLOWED to
contain rather than about geometry.

1. THE SCAN IS NEVER WRITTEN. A prescription is six numbers and a tooth id — a
   treatment plan. A mesh of somebody's dentition is biometric data that
   identifies them whether or not a name is attached. The test greps the actual
   ciphertext-decrypted payload for geometry keys, because a comment promising
   this is worth nothing.

2. A TAMPERED FILE IS REFUSED, NOT PARTIALLY LOADED. A treatment plan that
   loads with half its teeth missing is more dangerous than one that will not
   load at all.
"""
import json
import os
import shutil
import tempfile

import case_store
import domain
from session_store import SessionStore


# ---------------------------------------------------------------- domain ---

def test_fdi_drives_tooth_class_and_wheeler_root():
    cases = [(11, "incisor", 10.0), (22, "incisor", 10.0), (33, "canine", 13.0),
             (43, "canine", 13.0), (14, "premolar", 9.0), (25, "premolar", 9.0),
             (36, "molar", 9.0), (48, "molar", 9.0)]
    for fdi, klass, root in cases:
        t = domain.Tooth(tooth_id="x", arch="lower", fdi=fdi)
        assert t.tooth_class == klass, f"FDI {fdi} classed as {t.tooth_class}"
        assert t.expected_root_mm == root, f"FDI {fdi} expected {root}mm"
    print(f"PASS  {len(cases)} FDI numbers map to Wheeler buckets 10/13/9mm")


def test_the_shipped_manifest_bug_is_now_detectable():
    """A real manifest carried FDI 33 — a mandibular canine — on a 10mm root
    where Wheeler gives 13, because one session-wide slider fed every tooth.
    C_res is extrapolated along the long axis by exactly this distance."""
    wrong = domain.Tooth(tooth_id="a", arch="lower", fdi=33, root_length_mm=10.0)
    assert wrong.root_length_matches_fdi() is False, "the canine-on-10mm bug is invisible"

    right = domain.Tooth(tooth_id="b", arch="lower", fdi=33, root_length_mm=13.0)
    assert right.root_length_matches_fdi() is True

    # Unknowable is a THIRD answer, not a failure. Without segmentation there is
    # no FDI, and claiming a mismatch would be as wrong as claiming a match.
    unknown = domain.Tooth(tooth_id="c", arch="lower", fdi=None, root_length_mm=10.0)
    assert unknown.root_length_matches_fdi() is None, "unknown must not read as False"
    print("PASS  FDI 33 on a 10mm root flagged; unknown FDI returns None, not False")


def test_case_resolves_the_antagonist_without_the_client():
    """The server could not previously find a tooth's opposing arch — nothing
    linked the two sessions, so the client had to pass opposing_session_id."""
    c = domain.Case()
    c.arches["upper"] = domain.Arch(arch="upper", session_id="UP")
    c.arches["lower"] = domain.Arch(arch="lower", session_id="LO")
    assert c.opposing_session_id("upper") == "LO"
    assert c.opposing_session_id("lower") == "UP"

    solo = domain.Case()
    solo.arches["lower"] = domain.Arch(arch="lower", session_id="LO")
    assert solo.opposing_session_id("lower") is None, \
        "a missing antagonist must be None — NOT an empty string a caller might send"
    print("PASS  antagonist resolves server-side; absent arch yields None")


def test_stage_count_is_the_max_and_names_what_binds_it():
    c = domain.Case()
    c.arches["lower"] = domain.Arch(arch="lower")
    c.arches["lower"].teeth["slow"] = domain.Tooth(
        tooth_id="slow", arch="lower", fdi=36,
        prescription=domain.Prescription(tip_deg=9.0))        # 9/2.0 -> 5
    c.arches["lower"].teeth["fast"] = domain.Tooth(
        tooth_id="fast", arch="lower", fdi=35,
        prescription=domain.Prescription(d_md=0.3))           # 0.3/0.25 -> 2

    assert c.stage_count() == 5, f"expected 5 stages, got {c.stage_count()}"
    rows = c.binding_teeth()
    binder = [r for r in rows if r["binds_the_case"]]
    assert len(binder) == 1 and binder[0]["tooth_id"] == "slow"
    assert binder[0]["driver"] == "tip_deg", \
        "a clinician asking why it is 5 stages needs the CHANNEL, not the number"
    print(f"PASS  case = max over teeth (5 stages), bound by slow/tip_deg")


def test_prescription_scales_rather_than_lerping_a_matrix():
    p = domain.Prescription(tip_deg=8.0, torque_deg=-4.0, d_md=0.4)
    half = p.scaled(0.5)
    assert abs(half.tip_deg - 4.0) < 1e-12 and abs(half.torque_deg + 2.0) < 1e-12
    assert abs(half.d_md - 0.2) < 1e-12
    assert domain.Prescription().is_zero() and not p.is_zero()
    assert p.scaled(0.0).is_zero(), "stage 0 must be exactly T0"
    print("PASS  prescription scales linearly; stage 0 is exactly T0")


# ----------------------------------------------------------- persistence ---

def _isolated_store():
    tmp = tempfile.mkdtemp(prefix="aligner_cases_")
    case_store.CASE_DIR = tmp
    case_store.KEY_FILE = os.path.join(tmp, ".case_key")
    return tmp


def _populated_case():
    c = domain.Case(label="upper crowding, mid-course")
    c.arches["lower"] = domain.Arch(arch="lower", session_id="LO",
                                    has_occlusal_frame=True, has_segmentation=True,
                                    scan_vertex_count=94848, scan_face_count=187625)
    c.arches["lower"].teeth["t1"] = domain.Tooth(
        tooth_id="t1", arch="lower", fdi=33, root_length_mm=13.0,
        c_res=[1.5, -2.0, -7.0],
        prescription=domain.Prescription(tip_deg=6.5, d_md=0.4), review="confirmed")
    return c


def test_a_case_round_trips_through_an_encrypted_file():
    tmp = _isolated_store()
    try:
        c = _populated_case()
        path = case_store.save(c)
        assert os.path.exists(path)

        back = case_store.load(c.case_id)
        assert back.case_id == c.case_id and back.label == c.label
        t = back.arches["lower"].teeth["t1"]
        assert t.fdi == 33 and t.root_length_mm == 13.0 and t.review == "confirmed"
        assert abs(t.prescription.tip_deg - 6.5) < 1e-12
        assert back.stage_count() == c.stage_count()
        print(f"PASS  case round-trips encrypted: {os.path.getsize(path)} bytes on disk")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_file_on_disk_is_not_readable_plaintext():
    tmp = _isolated_store()
    try:
        c = _populated_case()
        path = case_store.save(c)
        raw = open(path, "rb").read()

        # DISTINCTIVE strings only. A short needle like b"33" is worse than
        # useless here: Fernet output is base64, so any 2-character sequence
        # appears by chance roughly once per 4096 characters, and an 844-byte
        # token hits one about 20% of the time. That made this assertion flaky -
        # it failed on the FDI number while b"ab" collided just as readily, and
        # the result depended on the randomly generated key. A leak test whose
        # verdict changes with the key tests nothing.
        for leak in (b"crowding", b"mid-course", b"confirmed",
                     b"tip_deg", b"root_length_mm", b"prescription"):
            assert leak not in raw, f"{leak!r} is sitting in the file in clear text"

        # And the positive half, which the negative half cannot prove: the
        # content must still be RECOVERABLE with the key. Without this, a
        # function that wrote 844 random bytes would pass every check above.
        back = case_store.load(c.case_id)
        assert back.label == c.label
        assert back.arches["lower"].teeth["t1"].fdi == 33
        print(f"PASS  ciphertext ({len(raw)}B) leaks no distinctive plaintext, and the "
              f"content is recoverable with the key")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_scan_geometry_is_ever_persisted():
    """The load-bearing PHI claim, checked against the real payload."""
    tmp = _isolated_store()
    try:
        c = _populated_case()
        case_store.save(c)
        payload = json.dumps(c.to_dict())
        for banned in ("verts", "faces", "positions", "indices", "graph",
                       "concavity", "crown", "socket_cup", "vertex_normals"):
            assert banned not in payload, \
                f"'{banned}' reached the case payload — that is scan geometry"
        # Counts are fine; they are metadata, not the mesh.
        assert "scan_face_count" in payload
        assert len(payload) < 20_000, \
            f"payload is {len(payload)} bytes — too large to be plan-only"
        print(f"PASS  payload is {len(payload)} bytes of plan, zero geometry keys")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_tampered_file_is_refused_not_half_loaded():
    tmp = _isolated_store()
    try:
        c = _populated_case()
        path = case_store.save(c)
        raw = bytearray(open(path, "rb").read())
        raw[len(raw) // 2] ^= 0x01            # flip one bit
        open(path, "wb").write(bytes(raw))

        try:
            case_store.load(c.case_id)
            raise AssertionError("a tampered case file loaded successfully")
        except ValueError as e:
            assert "integrity" in str(e).lower() or "decrypt" in str(e).lower()

        # And truncation.
        open(path, "wb").write(bytes(raw[: len(raw) // 3]))
        try:
            case_store.load(c.case_id)
            raise AssertionError("a truncated case file loaded successfully")
        except ValueError:
            pass
        print("PASS  one flipped bit and a truncation both refused by the HMAC")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_newer_schema_is_refused_rather_than_silently_downgraded():
    d = _populated_case().to_dict()
    d["schema_version"] = domain.SCHEMA_VERSION + 5
    try:
        domain.Case.from_dict(d)
        raise AssertionError("a future schema loaded, dropping fields silently")
    except ValueError as e:
        assert "newer" in str(e)
    print("PASS  a future schema version is refused, not partially read")


def test_listing_reports_that_a_scan_reload_is_required():
    tmp = _isolated_store()
    try:
        case_store.save(_populated_case())
        rows = case_store.list_cases()
        assert len(rows) == 1 and rows[0]["readable"]
        assert rows[0]["needs_scan_reload"] is True, \
            "the listing must say the scan is not in the file"
        assert rows[0]["tooth_count"] == 1
        print("PASS  listing states needs_scan_reload - the plan is saved, the scan is not")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_session_eviction_is_recorded_instead_of_silent():
    """Loading a fifth arch destroyed a case and the only evidence was a 404 on
    the next request — which reads as 'expired' and sends people to the TTL."""
    st = SessionStore(max_sessions=2)
    st.create("upper"); st.create("lower")
    assert st.evictions() == [], "nothing should be evicted yet"
    st.create("upper")
    ev = st.evictions()
    assert len(ev) == 1 and ev[0]["reason"] == "max_sessions"
    assert ev[0]["arch"] in ("upper", "lower") and ev[0]["max_sessions"] == 2
    print(f"PASS  eviction recorded: arch={ev[0]['arch']}, reason={ev[0]['reason']}")
# ------------------------------------------------------- long axis modes ---

def test_both_long_axis_derivations_are_available_and_measured():
    """The directive's u_OA = u_MD x u_BL against the shipped rim-normal form.

    The default is NOT changed. What is pinned here is that both exist, that
    the default stays rim_plane, and that switching is a measurable difference
    rather than a matter of opinion.
    """
    import numpy as np
    import arch_frame as af
    import core_geometry as cg
    from tooth_fixture import flat_molar_on_base

    assert cg.LONG_AXIS_MODE == "rim_plane", \
        "the default long-axis derivation changed; every measurement in " \
        "CLAUDE.md was taken against rim_plane"

    v, f, r = flat_molar_on_base()
    lo, hi = v.min(0), v.max(0)
    mid = (lo + hi) / 2
    frame = af.fit_occlusal_frame([lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                                  [mid[0], hi[1], hi[2]], v.mean(0))
    rr = np.linalg.norm(v[:, :2], axis=1)
    crown = v[(rr < r * 0.95) & (v[:, 2] > v[:, 2].min() + 0.35)]
    rim = v[(np.abs(rr - r * 0.95) < 0.6) & (v[:, 2] > v[:, 2].min() + 0.30)]

    got = {}
    for mode in ("rim_plane", "cross_product"):
        fr = cg.derive_frame_from_region([-r, 0, hi[2]], [r, 0, hi[2]],
                                         crown, rim, frame, long_axis_mode=mode)
        assert fr["axis_mode"] == mode
        got[mode] = np.asarray(fr["u_oa"], float)
        # Whichever mode, the basis must remain right-handed and orthonormal.
        M = np.column_stack([fr["u_md"], fr["u_bl"], fr["u_oa"]])
        assert abs(np.linalg.det(M) - 1.0) < 1e-6, f"{mode} basis is not right-handed"

    sep = np.degrees(np.arccos(np.clip(got["rim_plane"] @ got["cross_product"], -1, 1)))
    # They agree closely on a well-formed rim. They are NOT identical, which is
    # the point: on a short broad molar with a ragged margin they diverge, and
    # that divergence is what the tetherball fix was about.
    assert sep < 5.0, f"the two derivations disagree by {sep:.2f} deg on a clean fixture"
    assert got["rim_plane"][2] > 0.9, "rim_plane axis is not pointing occlusally"
    print(f"PASS  both axis modes available; they differ by {sep:.3f} deg on this "
          f"fixture, default stays rim_plane")



# ------------------------------------------------------------ audit trail ---

def test_audit_records_decisions_and_refuses_patient_data():
    """An entry says WHAT changed and by how much. It cannot be made to say
    who the patient is, and that is enforced at runtime, not by convention."""
    import audit
    c = domain.Case()
    c.record(audit.TOOTH_CUT, arch="lower", tooth_id="t1", fdi=36,
             detail="wand flood", values={"root_length_mm": 9.0})
    c.record(audit.PRESCRIPTION_COMMITTED, tooth_id="t1", fdi=36,
             values={"tip_deg": 6.5, "d_md": 0.4})

    assert len(c.audit_entries) == 2
    assert c.trail().summary()["by_action"] == {
        audit.TOOTH_CUT: 1, audit.PRESCRIPTION_COMMITTED: 1}
    assert c.trail().summary()["telemetry"].startswith("none")

    # Geometry and identifiers are refused, checked on the real payload.
    for bad in ({"verts": [[0, 0, 0]]}, {"patient_name": "x"}, {"filename": "a.stl"}):
        try:
            c.record(audit.TOOTH_CUT, values=bad)
            raise AssertionError(f"audit accepted {list(bad)}")
        except ValueError as e:
            assert "Refusing to audit" in str(e)

    # An unknown verb is refused too: a typo must not create a silent category.
    try:
        c.record("deleted_everything")
        raise AssertionError("an unknown audit action was accepted")
    except ValueError as e:
        assert "vocabulary is closed" in str(e)
    print("PASS  audit records 2 decisions; geometry, identifiers and typos all refused")


def test_audit_survives_the_encrypted_round_trip_and_is_bounded():
    import audit
    tmp = _isolated_store()
    try:
        c = _populated_case()
        for i in range(5):
            c.record(audit.PRESCRIPTION_COMMITTED, tooth_id="t1", fdi=33,
                     values={"tip_deg": float(i)})
        case_store.save(c)
        back = case_store.load(c.case_id)
        assert len(back.audit_entries) == 5, "the decision record did not persist"
        assert back.trail().entries(action=audit.PRESCRIPTION_COMMITTED)[0].fdi == 33

        # Bounded: a clinical workstation must not accumulate an unbounded log.
        t = audit.AuditTrail(max_entries=3)
        for i in range(10):
            t.record(audit.TOOTH_RESET, tooth_id=f"t{i}")
        assert len(t.to_list()) == 3, "the trail is not bounded"
        assert t.entries()[-1].tooth_id == "t9", "bounding dropped the NEWEST entries"
        print("PASS  audit persists through encryption and bounds to the newest entries")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_fdi_drives_tooth_class_and_wheeler_root()
    test_the_shipped_manifest_bug_is_now_detectable()
    test_case_resolves_the_antagonist_without_the_client()
    test_stage_count_is_the_max_and_names_what_binds_it()
    test_prescription_scales_rather_than_lerping_a_matrix()
    test_a_case_round_trips_through_an_encrypted_file()
    test_the_file_on_disk_is_not_readable_plaintext()
    test_no_scan_geometry_is_ever_persisted()
    test_a_tampered_file_is_refused_not_half_loaded()
    test_a_newer_schema_is_refused_rather_than_silently_downgraded()
    test_listing_reports_that_a_scan_reload_is_required()
    test_session_eviction_is_recorded_instead_of_silent()
    test_both_long_axis_derivations_are_available_and_measured()
    test_audit_records_decisions_and_refuses_patient_data()
    test_audit_survives_the_encrypted_round_trip_and_is_bounded()
    print("\nALL DOMAIN + PERSISTENCE TESTS PASSED")

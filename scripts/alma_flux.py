"""Measure the integrated flux density of a resolved asteroid in an ALMA continuum image (Stokes I, pbcor) and
write it, with the observing geometry from JPL Horizons, to a JSON file that `ir_to_mm.py` and
`act_vs_neatm.py` can overlay on the SED."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons

AU_KM = 1.495978707e8


def measure(
    fitsPath: Path, aperture: float, annulus: tuple[float, float], half: int
) -> dict:
    """
    Integrated flux of the source at the centre of a primary-beam-corrected ALMA image.

    The flux is the sum of the pixels inside a circular aperture centred on the flux-weighted centroid of the
    pixels above 5 sigma, divided by the clean-beam area in pixels. The noise is the rms in an annulus around
    the field centre (outside the source). The aperture systematic is half the spread of the growth curve
    between 0.7 and 1.5 times the aperture radius (plateau of the curve; includes the clean bowl).

    Parameters
    ----------
    fitsPath : `~pathlib.Path`
        ALMA continuum FITS image (Stokes I is the first plane of the Stokes axis).
    aperture : float
        Aperture radius in arcsec.
    annulus : tuple of float
        Inner and outer radius [arcsec] of the noise annulus, measured from the image centre.
    half : int
        Half-width in pixels of the cutout used for the measurement.

    Returns
    -------
    dict
        flux_mJy, stat_mJy, aperture_sys_mJy, peak_mJy_beam, rms_mJy_beam, centroid_offset_arcsec, growth
        (radius [arcsec] and flux [mJy] lists), freq_GHz, date_obs, beam_arcsec and the image cutout.
    """
    with fits.open(fitsPath, memmap=True) as hdul:
        hd = hdul[0].header
        n = hdul[0].data.shape[-1]
        c0 = n // 2
        box = np.array(hdul[0].data[0, 0][c0 - half : c0 + half, c0 - half : c0 + half])
        pix = abs(hd["CDELT2"]) * 3600  # arcsec
        beamPix = np.pi * hd["BMAJ"] * hd["BMIN"] / (4 * np.log(2)) / (pix / 3600) ** 2
        info = {
            "freq_GHz": hd["CRVAL3"] / 1e9,
            "bandwidth_GHz": abs(hd["CDELT3"]) / 1e9,
            "date_obs": hd["DATE-OBS"],
            "object": hd.get("OBJECT"),
            "beam_arcsec": [hd["BMAJ"] * 3600, hd["BMIN"] * 3600],
            "bunit": hd["BUNIT"],
        }
    yy, xx = np.indices(box.shape)
    r0 = np.hypot(xx - half, yy - half) * pix
    ann = (r0 > annulus[0]) & (r0 < annulus[1]) & np.isfinite(box)
    rms = float(np.std(box[ann]))
    m = (box > 5 * rms) & (r0 < 0.5)
    cy, cx = (
        (yy[m] * box[m]).sum() / box[m].sum(),
        (xx[m] * box[m]).sum() / box[m].sum(),
    )
    rr = np.hypot(xx - cx, yy - cy) * pix

    def flux(radius: float) -> float:
        return float(np.nansum(box[rr < radius]) / beamPix * 1e3)

    radii = np.array(
        [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 0.80]
    )
    growth = [flux(r) for r in radii]
    plateau = [flux(r) for r in np.linspace(0.7 * aperture, 1.5 * aperture, 9)]
    sel = rr < aperture
    info |= {
        "flux_mJy": flux(aperture),
        "stat_mJy": float(rms * np.sqrt(sel.sum() / beamPix) * 1e3),
        "aperture_sys_mJy": float((max(plateau) - min(plateau)) / 2),
        "aperture_arcsec": aperture,
        "peak_mJy_beam": float(np.nanmax(box) * 1e3),
        "rms_mJy_beam": rms * 1e3,
        "centroid_offset_arcsec": [float((cx - half) * pix), float((cy - half) * pix)],
        "growth": {"radius_arcsec": radii.tolist(), "flux_mJy": growth},
    }
    return info | {"_cutout": box, "_pix": pix, "_centroid": (cx, cy)}


def main() -> None:
    """Parse the CLI, measure the flux, query Horizons for the geometry and write the JSON and a diagnostic plot."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", type=Path, required=True)
    ap.add_argument("--horizons-id", default="8", help="Horizons small-body id")
    ap.add_argument("--aperture", type=float, default=0.20, help="arcsec")
    ap.add_argument(
        "--annulus",
        type=float,
        nargs=2,
        default=[1.0, 3.5],
        help="noise annulus radii [arcsec]",
    )
    ap.add_argument(
        "--cal-frac",
        type=float,
        default=0.05,
        help="absolute flux calibration uncertainty (fraction)",
    )
    ap.add_argument("--out", type=Path, default=Path("reports/flora/alma_flora.json"))
    ap.add_argument("--debug", action="store_true", help="smaller cutout")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    m = measure(a.fits, a.aperture, tuple(a.annulus), 300 if a.debug else 900)
    cutout, pix, (cx, cy) = m.pop("_cutout"), m.pop("_pix"), m.pop("_centroid")
    t = Time(m["date_obs"], scale="utc").jd
    e = Horizons(
        id=a.horizons_id, id_type="smallbody", location="500", epochs=[t]
    ).ephemerides(quantities="1,19,20,24")
    r, delta, alpha = (float(e[k][0]) for k in ["r", "delta", "alpha"])
    err = float(
        np.sqrt(
            m["stat_mJy"] ** 2
            + m["aperture_sys_mJy"] ** 2
            + (a.cal_frac * m["flux_mJy"]) ** 2
        )
    )
    point = {
        "label": f"ALMA {m['freq_GHz']:.0f} GHz",
        "GHz": m["freq_GHz"],
        "F_mJy": m["flux_mJy"],
        "err_mJy": err,
        "date_obs": m["date_obs"],
        "r_au": r,
        "delta_au": delta,
        "alpha_rad": float(np.radians(alpha)),
        "calibration_fraction_assumed": a.cal_frac,
        "angular_diameter_arcsec_per_km": 3600 * np.degrees(1 / (delta * AU_KM)),
        "measurement": m,
    }
    a.out.write_text(json.dumps([point], indent=1))
    print(json.dumps({k: v for k, v in point.items() if k != "measurement"}, indent=1))
    print(
        {
            k: m[k]
            for k in [
                "flux_mJy",
                "stat_mJy",
                "aperture_sys_mJy",
                "rms_mJy_beam",
                "peak_mJy_beam",
            ]
        }
    )

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ext = cutout.shape[0] / 2 * pix
    ax[0].imshow(
        cutout * 1e3, origin="lower", extent=[-ext, ext, -ext, ext], cmap="inferno"
    )
    ax[0].add_patch(
        plt.Circle(
            ((cx - cutout.shape[0] / 2) * pix, (cy - cutout.shape[0] / 2) * pix),
            a.aperture,
            fill=False,
            color="c",
        )
    )
    ax[0].set(
        xlim=(-0.6, 0.6),
        ylim=(-0.6, 0.6),
        xlabel="arcsec",
        ylabel="arcsec",
        title="Stokes I [mJy/beam]",
    )
    ax[1].plot(m["growth"]["radius_arcsec"], m["growth"]["flux_mJy"], "o-")
    ax[1].axvline(a.aperture, c="k", ls=":")
    ax[1].set(
        xlabel="aperture radius [arcsec]",
        ylabel="integrated flux [mJy]",
        title=f"growth curve; F = {m['flux_mJy']:.1f} +/- {err:.1f} mJy",
    )
    fig.tight_layout()
    fig.savefig(a.out.with_suffix(".png"), dpi=100)


if __name__ == "__main__":
    main()

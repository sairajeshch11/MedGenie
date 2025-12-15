
from typing import List, Dict, Any, Optional
import requests

NOMINATIM_BASE = "https://nominatim.openstreetmap.org"

def geocode_location(location: str, limit: int = 1) -> Optional[Dict[str, Any]]:
    """
    Returns {lat, lon, display_name} or None
    """
    params = {"q": location, "format": "json", "limit": limit}
    headers = {"User-Agent": "ir-llm-project/1.0 (edu demo)"}
    r = requests.get(f"{NOMINATIM_BASE}/search", params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    hit = data[0]
    return {"lat": float(hit["lat"]), "lon": float(hit["lon"]), "display_name": hit.get("display_name","")}

def find_hospitals_near(location: str, radius_m: int = 8000, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Uses Nominatim (Overpass-style features exposed) via /search with 'amenity=hospital' around the geocoded point.
    Note: Nominatim isn't a perfect POI search; but it's enough for a class demo.
    """
    geo = geocode_location(location)
    if not geo:
        return []

    lat, lon = geo["lat"], geo["lon"]

    # Nominatim supports 'q' queries; use a bounded viewbox to approximate radius.
    # Rough conversion: 1 degree lat ~ 111km. For lon scale with cos(lat).
    dlat = radius_m / 111000.0
    dlon = radius_m / (111000.0 * max(0.2, __import__("math").cos(__import__("math").radians(lat))))
    left, right = lon - dlon, lon + dlon
    top, bottom = lat + dlat, lat - dlat

    params = {
        "q": "hospital",
        "format": "json",
        "limit": limit,
        "viewbox": f"{left},{top},{right},{bottom}",
        "bounded": 1,
    }
    headers = {"User-Agent": "ir-llm-project/1.0 (edu demo)"}
    r = requests.get(f"{NOMINATIM_BASE}/search", params=params, headers=headers, timeout=25)
    r.raise_for_status()
    data = r.json()

    out = []
    for it in data:
        out.append({
            "name": it.get("display_name","").split(",")[0].strip() or "Hospital",
            "display_name": it.get("display_name",""),
            "lat": float(it["lat"]),
            "lon": float(it["lon"]),
            "source": "OpenStreetMap (Nominatim)"
        })
    return out
# ---- Backward-compatible alias (app.py imports find_nearby_hospitals) ----
def find_nearby_hospitals(location: str, radius_m: int = 8000, limit: int = 8):
    return find_hospitals_near(location, radius_m=radius_m, limit=limit)
# ---- Backward/forward-compatible wrapper (app.py imports find_nearby_hospitals) ----
def find_nearby_hospitals(
    location: str,
    radius_m: int = None,
    radius_km: float = None,
    max_results: int = None,
    limit: int = None,
):
    # Supports calls like:
    #   find_nearby_hospitals(loc, radius_km=R, max_results=k)
    #   find_nearby_hospitals(loc, radius_km, k)
    #   find_nearby_hospitals(loc, radius_m=R, limit=k)

    # Decide number of results
    k = max_results if max_results is not None else limit
    if k is None:
        k = 8

    # Decide radius in meters
    if radius_m is not None:
        r_m = int(radius_m)
    elif radius_km is not None:
        r_m = int(float(radius_km) * 1000.0)
    else:
        r_m = 8000

    return find_hospitals_near(location, radius_m=r_m, limit=int(k))


import math
import requests
from typing import Any, Dict, List, Tuple

UA = {"User-Agent": "MedRAG-StudentProject/1.0 (educational)"}

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def geocode_location(location: str) -> Tuple[float, float, str]:
    """
    Returns (lat, lon, display_name)
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers=UA, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError("Location not found.")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    name = data[0].get("display_name", location)
    return lat, lon, name

def find_nearby_hospitals(lat: float, lon: float, radius_km: float = 10.0, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Uses Overpass to find nearby hospitals.
    """
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    r = requests.post(url, data=query.encode("utf-8"), headers=UA, timeout=40)
    r.raise_for_status()
    data = r.json()

    results = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Hospital")
        # coordinates
        if el.get("type") == "node":
            hlat, hlon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center") or {}
            hlat, hlon = center.get("lat"), center.get("lon")
        if hlat is None or hlon is None:
            continue

        dist = _haversine_km(lat, lon, float(hlat), float(hlon))
        addr = " ".join(filter(None, [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:city"),
            tags.get("addr:state"),
            tags.get("addr:postcode")
        ])).strip()

        results.append({
            "name": name,
            "distance_km": round(dist, 2),
            "address": addr if addr else None,
            "map_url": f"https://www.openstreetmap.org/?mlat={hlat}&mlon={hlon}#map=16/{hlat}/{hlon}"
        })

    results.sort(key=lambda x: x["distance_km"])
    return results[:limit]

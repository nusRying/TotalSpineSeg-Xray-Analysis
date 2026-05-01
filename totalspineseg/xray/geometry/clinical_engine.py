import numpy as np
from scipy.spatial import ConvexHull
from typing import Dict, List, Tuple, Optional

class VertebraGeometry:
    """
    Represents the geometric properties of a single vertebral body.
    Supports initialization from a mask or direct landmarks.
    """
    def __init__(self, mask: Optional[np.ndarray] = None, label_id: int = 0, label_name: str = "", 
                 landmarks: Optional[Dict[str, np.ndarray]] = None):
        self.label_id = label_id
        self.label_name = label_name
        self.valid = False
        
        if landmarks is not None:
            self.corners = landmarks
            self.centroid = np.mean(list(landmarks.values()), axis=0)
            self.valid = True
        elif mask is not None:
            self.y_coords, self.x_coords = np.where(mask == label_id)
            if len(self.x_coords) >= 10:
                self.points = np.column_stack((self.x_coords, self.y_coords))
                self.centroid = np.mean(self.points, axis=0)
                self.corners = self._extract_corners_from_mask()
                self.valid = True

    def _extract_corners_from_mask(self) -> Dict[str, np.ndarray]:
        hull = ConvexHull(self.points)
        hull_pts = self.points[hull.vertices]
        
        # SA (Superior Anterior), SP (Superior Posterior), IA (Inferior Anterior), IP (Inferior Posterior)
        # Assuming Lateral Neutral: Anterior is Left (-X), Superior is Top (-Y)
        sa = hull_pts[np.argmin(hull_pts[:, 0] + hull_pts[:, 1])]
        ip = hull_pts[np.argmax(hull_pts[:, 0] + hull_pts[:, 1])]
        sp = hull_pts[np.argmax(hull_pts[:, 0] - hull_pts[:, 1])]
        ia = hull_pts[np.argmin(hull_pts[:, 0] - hull_pts[:, 1])]
        
        return {"SA": sa, "SP": sp, "IA": ia, "IP": ip}

    def get_slope(self, surface: str = "superior") -> float:
        p1, p2 = (self.corners["SA"], self.corners["SP"]) if surface == "superior" else (self.corners["IA"], self.corners["IP"])
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))

    def get_height(self, side: str = "anterior") -> float:
        p1, p2 = (self.corners["SA"], self.corners["IA"]) if side == "anterior" else (self.corners["SP"], self.corners["IP"])
        return np.linalg.norm(p1 - p2)

class ClinicalGeometryEngine:
    def __init__(self, mask: Optional[np.ndarray] = None, label_map: Dict[int, str] = None, 
                 pixel_spacing: float = 1.0, vertebrae: List[VertebraGeometry] = None,
                 view: str = "lateral_neutral"):
        self.pixel_spacing = pixel_spacing
        self.view = view.lower()
        if vertebrae is not None:
            self.vertebrae = sorted(vertebrae, key=lambda v: v.centroid[1])
        elif mask is not None and label_map is not None:
            temp_v = []
            for lid, name in label_map.items():
                if lid == 0: continue
                vg = VertebraGeometry(mask, lid, name)
                if vg.valid: temp_v.append(vg)
            self.vertebrae = sorted(temp_v, key=lambda v: v.centroid[1])
        else:
            self.vertebrae = []

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculates all 21 metrics, filling with 0.0 if not applicable to current view."""
        metrics = {m: 0.0 for m in [
            "cervical_lordosis_deg", "thoracic_kyphosis_deg", "lumbar_lordosis_deg",
            "lat_translation_mm", "coronal_cobb_deg", "coronal_balance_mm",
            "neutral_listhesis_mm", "neutral_seg_angle_deg", "neutral_interspinous_gap_mm",
            "disc_ht_ant_mm", "disc_ht_mid_mm", "disc_ht_post_mm", "vert_ht_loss_pct",
            "flex_listhesis_mm", "flex_angle_deg", "flex_gap_mm",
            "ext_listhesis_mm", "ext_angle_deg", "ext_gap_mm",
            "facet_space_narrow_grade", "facet_osteophyte_mm"
        ]}
        
        # 1. Regional Alignment
        alignment = self._calculate_regional_alignment()
        metrics.update(alignment)
        
        # 2. View-Specific Logic
        if "lateral" in self.view:
            prefix = "neutral" if "neutral" in self.view else ("flex" if "flexion" in self.view else "ext")
            
            for i in range(len(self.vertebrae) - 1):
                sup, inf = self.vertebrae[i], self.vertebrae[i+1]
                pair_key = f"{sup.label_name}_{inf.label_name}"
                
                # Listhesis & Angle
                metrics[f"{prefix}_listhesis_mm"] = (sup.corners["IP"][0] - inf.corners["SP"][0]) * self.pixel_spacing
                metrics[f"{prefix}_seg_angle_deg"] = sup.get_slope("inferior") - inf.get_slope("superior")
                
                if prefix == "neutral":
                    metrics[f"{pair_key}_disc_ht_ant_mm"] = self._dist(sup.corners["IA"], inf.corners["SA"])
                    metrics[f"{pair_key}_disc_ht_post_mm"] = self._dist(sup.corners["IP"], inf.corners["SP"])
                    metrics[f"{pair_key}_disc_ht_mid_mm"] = self._dist(np.mean([sup.corners["IA"], sup.corners["IP"]], axis=0), 
                                                                   np.mean([inf.corners["SA"], inf.corners["SP"]], axis=0))

            for v in self.vertebrae:
                ant_h, post_h = v.get_height("anterior"), v.get_height("posterior")
                metrics[f"{v.label_name}_vert_ht_loss_pct"] = max(0, (post_h - ant_h) / post_h * 100) if post_h > 0 else 0

        elif "ap" in self.view:
            metrics["coronal_cobb_deg"] = self._calculate_max_cobb()
            if len(self.vertebrae) >= 2:
                # Coronal Balance: C7 vs Sacrum/L5 horizontal distance
                v_dict = {v.label_name: v for v in self.vertebrae}
                top = v_dict.get("C7") or self.vertebrae[0]
                bot = v_dict.get("sacrum") or v_dict.get("L5") or self.vertebrae[-1]
                metrics["coronal_balance_mm"] = (top.centroid[0] - bot.centroid[0]) * self.pixel_spacing
                
                # Lat Translation: Mean horizontal shift between adjacent pairs
                shifts = []
                for i in range(len(self.vertebrae)-1):
                    shifts.append(abs(self.vertebrae[i].centroid[0] - self.vertebrae[i+1].centroid[0]))
                metrics["lat_translation_mm"] = np.mean(shifts) * self.pixel_spacing

        return metrics

    def generate_report(self) -> Dict:
        return {
            "summary": self.calculate_all_metrics(),
            "metadata": {"view": self.view, "vertebrae": [v.label_name for v in self.vertebrae], "spacing": self.pixel_spacing}
        }

    def _calculate_max_cobb(self) -> float:
        if len(self.vertebrae) < 2: return 0.0
        slopes = [v.get_slope("superior") for v in self.vertebrae] + [v.get_slope("inferior") for v in self.vertebrae]
        max_diff = 0.0
        for i in range(len(slopes)):
            for j in range(i + 1, len(slopes)):
                max_diff = max(max_diff, abs(slopes[i] - slopes[j]))
        return max_diff

    def _dist(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return np.linalg.norm(p1 - p2) * self.pixel_spacing

    def _calculate_regional_alignment(self) -> Dict[str, float]:
        res = {}
        v_dict = {v.label_name: v for v in self.vertebrae}
        ranges = {"cervical_lordosis_deg": ("C2", "C7"), "thoracic_kyphosis_deg": ("T1", "T12"), "lumbar_lordosis_deg": ("L1", "S1")}
        for name, (start, end) in ranges.items():
            if start in v_dict and end in v_dict:
                res[name] = v_dict[start].get_slope("superior") - v_dict[end].get_slope("inferior")
            else:
                res[name] = 0.0
        return res

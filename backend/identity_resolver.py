"""Identity resolution layer between ByteTrack and SubjectManager.

Maps raw track_ids (which reset on re-entry) to stable subject_ids that
persist across disappearances and re-appearances, using appearance
embeddings and body proportions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from multi_person_tracker import TrackedPerson
from pipeline_interface import PipelineResult
from embedding_extractor import EmbeddingExtractor
from body_proportions import compute_limb_lengths, ProportionAccumulator
from brace.core.pose import coco_keypoints_to_landmarks


@dataclass
class ResolvedPerson:
    """A tracked person with stable identity information."""
    person: TrackedPerson
    subject_id: int
    identity_status: str  # "unknown" | "tentative" | "confirmed"
    identity_confidence: float
    label: str  # "S1", "S3", etc. (always S#)


@dataclass
class ResolvedPipelineResult:
    """A PipelineResult with stable identity information."""
    pipeline_result: PipelineResult
    subject_id: int
    identity_status: str  # "unknown" | "tentative" | "confirmed"
    identity_confidence: float
    label: str  # "S1", "S3", etc. (always S#)


@dataclass
class _IdentityGallery:
    """Internal gallery of embeddings and proportions for one subject."""
    subject_id: int
    embeddings: list[np.ndarray] = field(default_factory=list)
    proportions: ProportionAccumulator = field(default_factory=ProportionAccumulator)
    status: str = "unknown"  # "unknown" | "tentative" | "confirmed"
    gallery_size: int = 20

    def add_embedding(self, emb: np.ndarray) -> None:
        self.embeddings.append(emb)
        if len(self.embeddings) > self.gallery_size:
            self.embeddings.pop(0)

    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        mean = np.mean(np.stack(self.embeddings), axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean

    def gallery_consistency(self) -> float:
        """Average pairwise cosine similarity within the gallery."""
        if len(self.embeddings) < 2:
            return 0.0
        mean = self.mean_embedding()
        if mean is None:
            return 0.0
        sims = []
        for emb in self.embeddings:
            sims.append(float(np.dot(emb, mean)))
        return float(np.mean(sims))


class IdentityResolver:
    """Resolves ByteTrack track_ids to stable subject identities."""

    def __init__(
        self,
        extractor: EmbeddingExtractor,
        match_threshold: float = 0.40,
        confirm_threshold: float = 0.65,
        gallery_size: int = 20,
        extraction_interval: int = 2,
        proportion_weight: float = 0.2,
        spatial_iou_threshold: float = 0.1,
        spatial_max_age: int = 90,
        cross_cut_extractor: EmbeddingExtractor | None = None,
        post_cut_frames: int = 5,
        centroid_distance_threshold: float = 0.15,
        centroid_max_age: int = 30,
        single_person_threshold: float = 0.30,
    ):
        self._extractor = extractor
        self._match_threshold = match_threshold
        self._confirm_threshold = confirm_threshold
        self._gallery_size = gallery_size
        self._extraction_interval = extraction_interval
        self._proportion_weight = proportion_weight
        self._spatial_iou_threshold = spatial_iou_threshold
        self._spatial_max_age = spatial_max_age

        # Centroid-distance spatial fallback parameters
        self._centroid_distance_threshold = centroid_distance_threshold
        self._centroid_max_age = centroid_max_age

        # Single-person bias threshold
        self._single_person_threshold = single_person_threshold

        # Optional CLIP-based extractor for cross-cut re-identification
        self._cross_cut_extractor = cross_cut_extractor
        self._post_cut_max_frames = post_cut_frames

        # track_id -> subject_id
        self._track_to_subject: dict[int, int] = {}

        # subject_id -> gallery
        self._galleries: dict[int, _IdentityGallery] = {}

        # subject_id -> is currently tracked (has active track)
        self._active_subjects: set[int] = set()

        # subject_id -> (last_bbox_normalized, last_seen_frame) for spatial fallback
        self._last_known_bboxes: dict[int, tuple[tuple[float, float, float, float], int]] = {}

        self._next_subject_id = 1
        self._frame_count = 0

        # Cross-cut re-identification state
        self._in_post_cut_mode = False
        self._post_cut_frames_remaining = 0

    def on_scene_cut(self) -> None:
        """Handle a detected scene cut.

        Marks all currently tracked subjects as lost (clears track->subject
        mappings) but preserves all galleries for future cross-cut matching.
        Activates post-cut mode for the next N frames where CLIP-based
        matching is used against ALL galleries (not just inactive ones).
        """
        # Clear all track->subject mappings (tracks will get new IDs after cut)
        self._track_to_subject.clear()

        # All subjects become inactive (no active tracks after cut)
        self._active_subjects.clear()

        # Enable post-cut matching mode
        if self._cross_cut_extractor is not None:
            self._in_post_cut_mode = True
            self._post_cut_frames_remaining = self._post_cut_max_frames

    @property
    def in_post_cut_mode(self) -> bool:
        """Whether the resolver is in post-cut CLIP matching mode."""
        return self._in_post_cut_mode

    def resolve(
        self,
        persons: list[TrackedPerson],
        rgb_frame: np.ndarray,
        w: int,
        h: int,
    ) -> list[ResolvedPerson]:
        """Resolve track_ids to stable subject_ids.

        Args:
            persons: detected/tracked persons from ByteTrack.
            rgb_frame: (H, W, 3) RGB frame for embedding extraction.
            w, h: frame dimensions.

        Returns:
            List of ResolvedPerson with stable subject_ids.
        """
        self._frame_count += 1
        should_extract = (self._frame_count % self._extraction_interval) == 0

        # Identify new track_ids that need forced embedding extraction
        new_track_indices = [
            i for i, p in enumerate(persons)
            if p.track_id not in self._track_to_subject
        ]

        # Extract embeddings: on interval frames for all, otherwise only for new tracks
        embeddings: list[np.ndarray | None] = [None] * len(persons)
        if persons:
            if should_extract:
                bboxes = [p.bbox_pixel for p in persons]
                all_embs = self._extractor.extract(rgb_frame, bboxes)
                for j, emb in enumerate(all_embs):
                    embeddings[j] = emb
            elif new_track_indices:
                # Force extraction only for new tracks (avoids creating blind subjects)
                new_bboxes = [persons[i].bbox_pixel for i in new_track_indices]
                new_embs = self._extractor.extract(rgb_frame, new_bboxes)
                for j, idx in enumerate(new_track_indices):
                    if j < len(new_embs):
                        embeddings[idx] = new_embs[j]

        resolved = []
        current_active = set()

        for i, person in enumerate(persons):
            track_id = person.track_id
            emb = embeddings[i]

            if track_id in self._track_to_subject:
                # Known track -> existing subject
                subject_id = self._track_to_subject[track_id]
                gallery = self._galleries[subject_id]

                # Update gallery
                if emb is not None and np.linalg.norm(emb) > 0:
                    gallery.add_embedding(emb)

                # Update proportions
                landmarks = coco_keypoints_to_landmarks(person.keypoints, w, h)
                limbs = compute_limb_lengths(landmarks)
                if limbs is not None:
                    gallery.proportions.add(limbs)

                # Try status promotion
                self._try_promote(gallery)

            else:
                # New track_id -> try to match against inactive galleries
                subject_id = self._match_to_existing(emb, person, w, h, current_active)

                if subject_id is None:
                    # No match -> create new unknown subject
                    subject_id = self._next_subject_id
                    self._next_subject_id += 1
                    print(
                        f"[reid] NEW subject S{subject_id} "
                        f"track={track_id} frame={self._frame_count} "
                        f"bbox={person.bbox_normalized} "
                        f"galleries={len(self._galleries)} "
                        f"emb={'yes' if emb is not None and np.linalg.norm(emb)>1e-8 else 'no'}",
                        flush=True,
                    )
                    gallery = _IdentityGallery(
                        subject_id=subject_id,
                        gallery_size=self._gallery_size,
                    )
                    self._galleries[subject_id] = gallery

                    if emb is not None and np.linalg.norm(emb) > 0:
                        gallery.add_embedding(emb)

                    landmarks = coco_keypoints_to_landmarks(person.keypoints, w, h)
                    limbs = compute_limb_lengths(landmarks)
                    if limbs is not None:
                        gallery.proportions.add(limbs)
                else:
                    gallery = self._galleries[subject_id]
                    if emb is not None and np.linalg.norm(emb) > 0:
                        gallery.add_embedding(emb)
                    landmarks = coco_keypoints_to_landmarks(person.keypoints, w, h)
                    limbs = compute_limb_lengths(landmarks)
                    if limbs is not None:
                        gallery.proportions.add(limbs)

                self._track_to_subject[track_id] = subject_id

            # Store last known bbox for spatial fallback
            self._last_known_bboxes[subject_id] = (
                person.bbox_normalized, self._frame_count
            )
            current_active.add(subject_id)

            resolved.append(ResolvedPerson(
                person=person,
                subject_id=subject_id,
                identity_status=self._galleries[subject_id].status,
                identity_confidence=self._galleries[subject_id].gallery_consistency(),
                label=self._make_label(subject_id, self._galleries[subject_id].status),
            ))

        self._active_subjects = current_active
        return resolved

    def resolve_pipeline_results(
        self,
        results: list[PipelineResult],
        rgb_frame: np.ndarray,
        w: int,
        h: int,
    ) -> list[ResolvedPipelineResult]:
        """Resolve PipelineResult track_ids to stable subject_ids.

        Like resolve() but accepts PipelineResult (which already has
        landmarks_mp in MediaPipe format) instead of TrackedPerson.

        During post-cut mode (first N frames after a scene cut), uses the
        cross-cut extractor (CLIP) and matches against ALL galleries
        (not just inactive ones) for cross-camera re-identification.

        Args:
            results: pipeline results with track_ids and landmarks.
            rgb_frame: (H, W, 3) RGB frame for embedding extraction.
            w, h: frame dimensions.

        Returns:
            List of ResolvedPipelineResult with stable subject_ids.
        """
        self._frame_count += 1
        should_extract = (self._frame_count % self._extraction_interval) == 0

        # Determine which extractor to use
        in_post_cut = self._in_post_cut_mode and self._post_cut_frames_remaining > 0
        extractor = self._cross_cut_extractor if in_post_cut else self._extractor

        # In post-cut mode, force extraction every frame for all detections
        force_all_extraction = in_post_cut

        new_track_indices = [
            i for i, r in enumerate(results)
            if r.track_id not in self._track_to_subject
        ]

        # Extract embeddings
        embeddings: list[np.ndarray | None] = [None] * len(results)
        if results:
            if should_extract or force_all_extraction:
                bboxes = [r.bbox_pixel for r in results]
                all_embs = extractor.extract(rgb_frame, bboxes)
                for j, emb in enumerate(all_embs):
                    embeddings[j] = emb
            elif new_track_indices:
                new_bboxes = [results[i].bbox_pixel for i in new_track_indices]
                new_embs = extractor.extract(rgb_frame, new_bboxes)
                for j, idx in enumerate(new_track_indices):
                    if j < len(new_embs):
                        embeddings[idx] = new_embs[j]

        resolved = []
        current_active = set()

        for i, pr in enumerate(results):
            track_id = pr.track_id
            emb = embeddings[i]
            landmarks = pr.landmarks_mp  # already MediaPipe (33, 4)

            if track_id in self._track_to_subject:
                subject_id = self._track_to_subject[track_id]
                gallery = self._galleries[subject_id]

                if emb is not None and np.linalg.norm(emb) > 0:
                    gallery.add_embedding(emb)

                limbs = compute_limb_lengths(landmarks)
                if limbs is not None:
                    gallery.proportions.add(limbs)

                self._try_promote(gallery)
            else:
                # In post-cut mode, match against ALL galleries (skip nothing)
                if in_post_cut:
                    subject_id = self._match_cross_cut(emb, landmarks, current_active)
                else:
                    subject_id = self._match_to_existing_by_landmarks(
                        emb, landmarks, pr.bbox_normalized, current_active
                    )

                if subject_id is None:
                    subject_id = self._next_subject_id
                    self._next_subject_id += 1
                    gallery = _IdentityGallery(
                        subject_id=subject_id,
                        gallery_size=self._gallery_size,
                    )
                    self._galleries[subject_id] = gallery

                    if emb is not None and np.linalg.norm(emb) > 0:
                        gallery.add_embedding(emb)

                    limbs = compute_limb_lengths(landmarks)
                    if limbs is not None:
                        gallery.proportions.add(limbs)
                else:
                    gallery = self._galleries[subject_id]
                    if emb is not None and np.linalg.norm(emb) > 0:
                        gallery.add_embedding(emb)
                    limbs = compute_limb_lengths(landmarks)
                    if limbs is not None:
                        gallery.proportions.add(limbs)

                self._track_to_subject[track_id] = subject_id

            self._last_known_bboxes[subject_id] = (
                pr.bbox_normalized, self._frame_count
            )
            current_active.add(subject_id)

            resolved.append(ResolvedPipelineResult(
                pipeline_result=pr,
                subject_id=subject_id,
                identity_status=self._galleries[subject_id].status,
                identity_confidence=self._galleries[subject_id].gallery_consistency(),
                label=self._make_label(subject_id, self._galleries[subject_id].status),
            ))

        self._active_subjects = current_active

        # Decrement post-cut counter
        if in_post_cut:
            self._post_cut_frames_remaining -= 1
            if self._post_cut_frames_remaining <= 0:
                self._in_post_cut_mode = False

        return resolved

    def cleanup_stale_tracks(self, active_track_ids: set[int] | list[int]) -> None:
        """Remove track->subject mappings for gone tracks.

        Keeps gallery/identity for future re-matching.
        """
        active = set(active_track_ids)
        stale = [tid for tid in self._track_to_subject if tid not in active]
        for tid in stale:
            sid = self._track_to_subject.pop(tid)
            self._active_subjects.discard(sid)

    def _match_to_existing(
        self,
        emb: np.ndarray | None,
        person: TrackedPerson,
        w: int,
        h: int,
        current_active: set[int] | None = None,
    ) -> int | None:
        """Try to match a new track against inactive galleries.

        Uses appearance embeddings + body proportions first, with bbox IoU
        as a fallback for recently-disappeared subjects.
        """
        # Skip subjects that are active (from prev frame) or already assigned this frame
        skip = self._active_subjects | (current_active or set())

        # Collect inactive galleries for single-person bias check
        inactive_galleries = [
            sid for sid in self._galleries if sid not in skip
        ]

        best_score = -1.0
        best_id = None
        has_valid_emb = emb is not None and np.linalg.norm(emb) >= 1e-8

        # Single-person bias: if exactly one inactive gallery and one new track,
        # use a lower threshold for strong re-match bias
        effective_threshold = self._match_threshold
        if len(inactive_galleries) == 1 and has_valid_emb:
            effective_threshold = self._single_person_threshold

        if has_valid_emb:
            for sid, gallery in self._galleries.items():
                if sid in skip:
                    continue

                mean_emb = gallery.mean_embedding()
                if mean_emb is None:
                    continue

                # Cosine similarity (both L2-normalized)
                appearance_sim = float(np.dot(emb, mean_emb))

                # Body proportion similarity (if available)
                prop_sim = 0.0
                landmarks = coco_keypoints_to_landmarks(person.keypoints, w, h)
                limbs = compute_limb_lengths(landmarks)
                if limbs is not None:
                    prop_vec = gallery.proportions.get_proportion_vector()
                    if prop_vec is not None:
                        n1 = np.linalg.norm(limbs)
                        n2 = np.linalg.norm(prop_vec)
                        if n1 > 1e-8 and n2 > 1e-8:
                            prop_sim = float(np.dot(limbs, prop_vec) / (n1 * n2))

                score = (1 - self._proportion_weight) * appearance_sim + self._proportion_weight * prop_sim

                if score > best_score:
                    best_score = score
                    best_id = sid

            if best_score >= effective_threshold and best_id is not None:
                return best_id

        # Fallback 1: spatial IoU for recently-disappeared subjects
        person_bbox = person.bbox_normalized
        best_iou = 0.0
        best_spatial_id = None

        for sid, (last_bbox, last_frame) in self._last_known_bboxes.items():
            if sid in skip:
                continue
            if self._frame_count - last_frame > self._spatial_max_age:
                continue

            iou = _bbox_iou(person_bbox, last_bbox)
            if iou > best_iou:
                best_iou = iou
                best_spatial_id = sid

        if best_iou >= self._spatial_iou_threshold and best_spatial_id is not None:
            return best_spatial_id

        # Fallback 2: centroid distance for recently-disappeared subjects
        best_dist = float("inf")
        best_centroid_id = None

        person_cx = (person_bbox[0] + person_bbox[2]) / 2
        person_cy = (person_bbox[1] + person_bbox[3]) / 2

        for sid, (last_bbox, last_frame) in self._last_known_bboxes.items():
            if sid in skip:
                continue
            if self._frame_count - last_frame > self._centroid_max_age:
                continue

            last_cx = (last_bbox[0] + last_bbox[2]) / 2
            last_cy = (last_bbox[1] + last_bbox[3]) / 2
            dist = ((person_cx - last_cx) ** 2 + (person_cy - last_cy) ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_centroid_id = sid

        if best_dist <= self._centroid_distance_threshold and best_centroid_id is not None:
            return best_centroid_id

        return None

    def _match_to_existing_by_landmarks(
        self,
        emb: np.ndarray | None,
        landmarks: np.ndarray,
        bbox_normalized: tuple[float, float, float, float],
        current_active: set[int] | None = None,
    ) -> int | None:
        """Try to match a new track against inactive galleries using pre-computed landmarks.

        Same logic as _match_to_existing but takes MediaPipe landmarks directly
        instead of a TrackedPerson with COCO keypoints.
        """
        skip = self._active_subjects | (current_active or set())

        # Collect inactive galleries for single-person bias check
        inactive_galleries = [
            sid for sid in self._galleries if sid not in skip
        ]

        best_score = -1.0
        best_id = None
        has_valid_emb = emb is not None and np.linalg.norm(emb) >= 1e-8

        # Single-person bias: if exactly one inactive gallery and one new track,
        # use a lower threshold for strong re-match bias
        effective_threshold = self._match_threshold
        if len(inactive_galleries) == 1 and has_valid_emb:
            effective_threshold = self._single_person_threshold

        if has_valid_emb:
            limbs = compute_limb_lengths(landmarks)
            for sid, gallery in self._galleries.items():
                if sid in skip:
                    continue

                mean_emb = gallery.mean_embedding()
                if mean_emb is None:
                    continue

                appearance_sim = float(np.dot(emb, mean_emb))

                prop_sim = 0.0
                if limbs is not None:
                    prop_vec = gallery.proportions.get_proportion_vector()
                    if prop_vec is not None:
                        n1 = np.linalg.norm(limbs)
                        n2 = np.linalg.norm(prop_vec)
                        if n1 > 1e-8 and n2 > 1e-8:
                            prop_sim = float(np.dot(limbs, prop_vec) / (n1 * n2))

                score = (1 - self._proportion_weight) * appearance_sim + self._proportion_weight * prop_sim

                if score > best_score:
                    best_score = score
                    best_id = sid

            if best_score >= effective_threshold and best_id is not None:
                return best_id

        # Fallback 1: spatial IoU
        best_iou = 0.0
        best_spatial_id = None

        for sid, (last_bbox, last_frame) in self._last_known_bboxes.items():
            if sid in skip:
                continue
            if self._frame_count - last_frame > self._spatial_max_age:
                continue

            iou = _bbox_iou(bbox_normalized, last_bbox)
            if iou > best_iou:
                best_iou = iou
                best_spatial_id = sid

        if best_iou >= self._spatial_iou_threshold and best_spatial_id is not None:
            return best_spatial_id

        # Fallback 2: centroid distance for recently-disappeared subjects
        best_dist = float("inf")
        best_centroid_id = None

        person_cx = (bbox_normalized[0] + bbox_normalized[2]) / 2
        person_cy = (bbox_normalized[1] + bbox_normalized[3]) / 2

        for sid, (last_bbox, last_frame) in self._last_known_bboxes.items():
            if sid in skip:
                continue
            if self._frame_count - last_frame > self._centroid_max_age:
                continue

            last_cx = (last_bbox[0] + last_bbox[2]) / 2
            last_cy = (last_bbox[1] + last_bbox[3]) / 2
            dist = ((person_cx - last_cx) ** 2 + (person_cy - last_cy) ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_centroid_id = sid

        if best_dist <= self._centroid_distance_threshold and best_centroid_id is not None:
            return best_centroid_id

        return None

    def _match_cross_cut(
        self,
        emb: np.ndarray | None,
        landmarks: np.ndarray,
        current_active: set[int] | None = None,
    ) -> int | None:
        """Match a new detection against ALL galleries after a scene cut.

        Unlike normal matching, this does NOT skip active subjects -- after
        a scene cut, all tracks are new and all galleries are candidates.
        Only skips subjects already assigned in this frame (current_active).
        """
        skip = current_active or set()

        best_score = -1.0
        best_id = None
        has_valid_emb = emb is not None and np.linalg.norm(emb) >= 1e-8

        if has_valid_emb:
            limbs = compute_limb_lengths(landmarks)
            for sid, gallery in self._galleries.items():
                if sid in skip:
                    continue

                mean_emb = gallery.mean_embedding()
                if mean_emb is None:
                    continue

                appearance_sim = float(np.dot(emb, mean_emb))

                prop_sim = 0.0
                if limbs is not None:
                    prop_vec = gallery.proportions.get_proportion_vector()
                    if prop_vec is not None:
                        n1 = np.linalg.norm(limbs)
                        n2 = np.linalg.norm(prop_vec)
                        if n1 > 1e-8 and n2 > 1e-8:
                            prop_sim = float(np.dot(limbs, prop_vec) / (n1 * n2))

                score = (1 - self._proportion_weight) * appearance_sim + self._proportion_weight * prop_sim

                if score > best_score:
                    best_score = score
                    best_id = sid

            if best_score >= self._match_threshold and best_id is not None:
                return best_id

        return None

    def _try_promote(self, gallery: _IdentityGallery) -> None:
        """Try to promote identity status based on gallery consistency."""
        consistency = gallery.gallery_consistency()
        if gallery.status == "unknown" and consistency >= self._match_threshold:
            gallery.status = "tentative"
        elif gallery.status == "tentative" and consistency >= self._confirm_threshold:
            gallery.status = "confirmed"

    @staticmethod
    def _make_label(subject_id: int, status: str) -> str:
        """Generate label: always S# (status is internal only)."""
        return f"S{subject_id}"


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two normalized bboxes (x1, y1, x2, y2)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union < 1e-8:
        return 0.0
    return inter / union

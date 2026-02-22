"use client";

import { useRef, useMemo, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import type { SubjectState } from "@/lib/types";
import { CLUSTER_COLORS, PHASE_COLORS } from "@/lib/colors";
import { jerseyDisplayColor } from "@/lib/colors";
import Card from "./ui/Card";

interface EmbeddingGraphProps {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  highlightedClusterRef: React.MutableRefObject<number | null>;
  nSegments?: number;
  nClusters?: number;
  showJerseyColors?: boolean;
}

const MAX_POINTS = 4000;
const POINT_RADIUS = 0.04;
const CURRENT_RADIUS = 0.1;
const GRAY = new THREE.Color("#CDCDCD");

/** Parse hex color to THREE.Color, cached. */
const colorCache = new Map<string, THREE.Color>();
function getColor(hex: string): THREE.Color {
  let c = colorCache.get(hex);
  if (!c) {
    c = new THREE.Color(hex);
    colorCache.set(hex, c);
  }
  return c;
}

/** Darken a THREE.Color by a factor (0-1). */
function darkenColor(color: THREE.Color, factor: number): THREE.Color {
  const c = color.clone();
  c.r *= factor;
  c.g *= factor;
  c.b *= factor;
  return c;
}

/** Instanced point cloud for all embedding points. */
function PointCloud({
  subjectsRef,
  selectedSubjectRef,
  highlightedClusterRef,
  showJerseyColors,
}: EmbeddingGraphProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tempColor = useMemo(() => new THREE.Color(), []);
  const boundsRef = useRef({ cx: 0, cy: 0, cz: 0, scale: 1 });

  // Current position sphere refs
  const currentRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef(0);
  const targetPos = useMemo(() => new THREE.Vector3(), []);

  // Hover state for popover
  const { raycaster, pointer } = useThree();
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const hoveredPosRef = useRef(new THREE.Vector3());

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const selectedId = selectedSubjectRef.current;
    if (selectedId === null) {
      mesh.count = 0;
      if (currentRef.current) currentRef.current.visible = false;
      if (ringRef.current) ringRef.current.visible = false;
      setHoveredIdx(null);
      return;
    }

    const subject = subjectsRef.current.get(selectedId);
    if (!subject) {
      mesh.count = 0;
      if (currentRef.current) currentRef.current.visible = false;
      if (ringRef.current) ringRef.current.visible = false;
      setHoveredIdx(null);
      return;
    }

    const { points, clusterIds, currentIdx } = subject.embedding;

    if (points.length < 20) {
      mesh.count = 0;
      if (currentRef.current) currentRef.current.visible = false;
      if (ringRef.current) ringRef.current.visible = false;
      setHoveredIdx(null);
      return;
    }

    // Compute bounds and normalize to [-2, 2] cube
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (const p of points) {
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
      if (p[2] < minZ) minZ = p[2];
      if (p[2] > maxZ) maxZ = p[2];
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const rangeZ = maxZ - minZ || 1;
    const maxRange = Math.max(rangeX, rangeY, rangeZ);
    const scale = 4.0 / maxRange;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    boundsRef.current = { cx, cy, cz, scale };

    const highlightCid = highlightedClusterRef.current;
    const count = Math.min(points.length, MAX_POINTS);
    mesh.count = count;

    // Jersey color mode: prefer visual team color (K-Means) over Gemini string
    const jerseyHex = subject.teamColor ?? (subject.jerseyColor ? jerseyDisplayColor(subject.jerseyColor) : null);
    const useJersey = showJerseyColors && jerseyHex;
    const jerseyBase = useJersey ? getColor(jerseyHex) : null;
    const jerseyDark = jerseyBase ? darkenColor(jerseyBase, 0.7) : null;

    for (let i = 0; i < count; i++) {
      const p = points[i];
      const x = (p[0] - cx) * scale;
      const y = (p[1] - cy) * scale;
      const z = (p[2] - cz) * scale;

      const cid = clusterIds[i];
      let radius = POINT_RADIUS;

      if (useJersey && jerseyDark) {
        // Jersey color mode
        if (i === currentIdx) {
          tempColor.copy(jerseyBase!);
          radius = POINT_RADIUS * 1.5;
        } else {
          tempColor.copy(jerseyDark);
        }
      } else if (highlightCid !== null) {
        if (cid === highlightCid) {
          tempColor.copy(
            cid !== null && cid !== undefined
              ? getColor(CLUSTER_COLORS[cid % CLUSTER_COLORS.length])
              : GRAY
          );
          radius = POINT_RADIUS * 1.3;
        } else {
          tempColor.set("#E0E0E0");
          radius = POINT_RADIUS * 0.7;
        }
      } else {
        tempColor.copy(
          cid !== null && cid !== undefined
            ? getColor(CLUSTER_COLORS[cid % CLUSTER_COLORS.length])
            : GRAY
        );
      }

      dummy.position.set(x, y, z);
      dummy.scale.setScalar(radius / POINT_RADIUS);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    // Hover detection via raycasting (only in debug/jersey mode)
    if (showJerseyColors) {
      raycaster.setFromCamera(pointer, raycaster.camera);
      const intersects = raycaster.intersectObject(mesh);
      if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
        const idx = intersects[0].instanceId;
        setHoveredIdx(idx);
        const p = points[idx];
        if (p) {
          hoveredPosRef.current.set(
            (p[0] - cx) * scale,
            (p[1] - cy) * scale,
            (p[2] - cz) * scale,
          );
        }
      } else {
        setHoveredIdx(null);
      }
    } else {
      if (hoveredIdx !== null) setHoveredIdx(null);
    }

    // Current position
    if (currentIdx >= 0 && currentIdx < points.length) {
      const cp = points[currentIdx];
      const cx2 = (cp[0] - cx) * scale;
      const cy2 = (cp[1] - cy) * scale;
      const cz2 = (cp[2] - cz) * scale;
      const phaseColor = getColor(PHASE_COLORS[subject.phase]);

      if (currentRef.current) {
        currentRef.current.visible = true;
        targetPos.set(cx2, cy2, cz2);
        currentRef.current.position.lerp(targetPos, 0.15);
        const mat = currentRef.current.material as THREE.MeshStandardMaterial;
        mat.color.copy(useJersey && jerseyBase ? jerseyBase : phaseColor);
      }

      if (ringRef.current) {
        ringRef.current.visible = true;
        ringRef.current.position.lerp(targetPos, 0.15);
        pulseRef.current += 0.05;
        const pulse = Math.sin(pulseRef.current) * 0.3 + 0.7;
        ringRef.current.scale.setScalar(pulse);
        const ringMat = ringRef.current.material as THREE.MeshBasicMaterial;
        ringMat.color.copy(useJersey && jerseyBase ? jerseyBase : phaseColor);
        ringMat.opacity = pulse * 0.5;
      }
    } else {
      if (currentRef.current) currentRef.current.visible = false;
      if (ringRef.current) ringRef.current.visible = false;
    }
  });

  const sphereGeo = useMemo(() => new THREE.SphereGeometry(POINT_RADIUS, 8, 8), []);
  const pointMat = useMemo(() => new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.1 }), []);

  // Get hover popover data
  const selectedId = selectedSubjectRef.current;
  const subject = selectedId !== null ? subjectsRef.current.get(selectedId) : undefined;

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[sphereGeo, pointMat, MAX_POINTS]}
        frustumCulled={false}
      />
      {/* Current position: inner solid sphere */}
      <mesh ref={currentRef} visible={false}>
        <sphereGeometry args={[CURRENT_RADIUS, 16, 16]} />
        <meshStandardMaterial roughness={0.3} metalness={0.2} />
      </mesh>
      {/* Current position: outer pulsing ring */}
      <mesh ref={ringRef} visible={false}>
        <sphereGeometry args={[CURRENT_RADIUS * 1.8, 16, 16]} />
        <meshBasicMaterial transparent opacity={0.3} wireframe />
      </mesh>
      {/* Hover popover (debug mode only) */}
      {showJerseyColors && hoveredIdx !== null && subject && (
        <Html
          position={[hoveredPosRef.current.x, hoveredPosRef.current.y + 0.3, hoveredPosRef.current.z]}
          style={{ pointerEvents: "none" }}
        >
          <div className="bg-black/85 text-white text-[10px] rounded-lg px-2 py-1.5 whitespace-nowrap max-w-[200px]">
            <div className="font-bold">Point {hoveredIdx}</div>
            {subject.embedding.clusterIds[hoveredIdx] != null && (
              <div>Cluster {subject.embedding.clusterIds[hoveredIdx]}</div>
            )}
            {subject.jerseyNumber != null && (
              <div>
                Jersey #{subject.jerseyNumber}
                {subject.jerseyColor && (
                  <span
                    className="inline-block w-2.5 h-2.5 rounded-full ml-1 align-middle"
                    style={{ backgroundColor: jerseyDisplayColor(subject.jerseyColor) }}
                  />
                )}
              </div>
            )}
            {subject.jerseyCropBase64 && (
              <img
                src={`data:image/jpeg;base64,${subject.jerseyCropBase64}`}
                alt="crop"
                className="mt-1 rounded max-w-[120px]"
              />
            )}
            {subject.jerseyGeminiResponse && (
              <div className="mt-0.5 text-[9px] text-gray-400 break-all">
                {subject.jerseyGeminiResponse}
              </div>
            )}
          </div>
        </Html>
      )}
    </>
  );
}

function SceneContent(props: EmbeddingGraphProps) {
  return (
    <>
      <PointCloud {...props} />
    </>
  );
}

export default function EmbeddingGraph({
  subjectsRef,
  selectedSubjectRef,
  highlightedClusterRef,
  nSegments,
  nClusters,
  showJerseyColors,
}: EmbeddingGraphProps) {
  return (
    <Card>
      <h3 className="text-sm font-bold text-[#3C3C3C] uppercase tracking-[0.03em] mb-0.5">
        Movement Embedding
      </h3>
      {(nSegments !== undefined || nClusters !== undefined) && (
        <div className="text-[11px] text-[#AFAFAF] mb-1.5">
          {nSegments ?? 0} motions &middot; {nClusters ?? 0} clusters
        </div>
      )}
      <div className="relative w-full bg-[#FAFAFA] rounded-[12px] overflow-hidden" style={{ aspectRatio: "1/1" }}>
        <Canvas
          camera={{ position: [0, 0, 6], fov: 50 }}
          style={{ background: "#FAFAFA" }}
        >
          <ambientLight intensity={0.7} />
          <directionalLight position={[5, 5, 5]} intensity={0.5} />
          <SceneContent
            subjectsRef={subjectsRef}
            selectedSubjectRef={selectedSubjectRef}
            highlightedClusterRef={highlightedClusterRef}
            showJerseyColors={showJerseyColors}
          />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            minDistance={2}
            maxDistance={12}
            autoRotate
            autoRotateSpeed={0.5}
          />
        </Canvas>
      </div>
    </Card>
  );
}

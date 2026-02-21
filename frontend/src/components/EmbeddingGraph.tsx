"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { SubjectState } from "@/lib/types";
import { CLUSTER_COLORS, PHASE_COLORS } from "@/lib/colors";
import Card from "./ui/Card";

interface EmbeddingGraphProps {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  highlightedClusterRef: React.MutableRefObject<number | null>;
  nSegments?: number;
  nClusters?: number;
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

/** Instanced point cloud for all embedding points. */
function PointCloud({
  subjectsRef,
  selectedSubjectRef,
  highlightedClusterRef,
}: EmbeddingGraphProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tempColor = useMemo(() => new THREE.Color(), []);
  const boundsRef = useRef({ cx: 0, cy: 0, cz: 0, scale: 1 });

  // Current position sphere refs
  const currentRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef(0);

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const selectedId = selectedSubjectRef.current;
    if (selectedId === null) {
      mesh.count = 0;
      if (currentRef.current) currentRef.current.visible = false;
      if (ringRef.current) ringRef.current.visible = false;
      return;
    }

    const subject = subjectsRef.current.get(selectedId);
    if (!subject) {
      mesh.count = 0;
      if (currentRef.current) currentRef.current.visible = false;
      if (ringRef.current) ringRef.current.visible = false;
      return;
    }

    const { points, clusterIds, currentIdx } = subject.embedding;

    if (points.length < 20) {
      mesh.count = 0;
      if (currentRef.current) currentRef.current.visible = false;
      if (ringRef.current) ringRef.current.visible = false;
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

    for (let i = 0; i < count; i++) {
      const p = points[i];
      const x = (p[0] - cx) * scale;
      const y = (p[1] - cy) * scale;
      const z = (p[2] - cz) * scale;

      const cid = clusterIds[i];
      let radius = POINT_RADIUS;

      if (highlightCid !== null) {
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

    // Current position
    if (currentIdx >= 0 && currentIdx < points.length) {
      const cp = points[currentIdx];
      const cx2 = (cp[0] - cx) * scale;
      const cy2 = (cp[1] - cy) * scale;
      const cz2 = (cp[2] - cz) * scale;
      const phaseColor = getColor(PHASE_COLORS[subject.phase]);

      if (currentRef.current) {
        currentRef.current.visible = true;
        currentRef.current.position.set(cx2, cy2, cz2);
        const mat = currentRef.current.material as THREE.MeshStandardMaterial;
        mat.color.copy(phaseColor);
      }

      if (ringRef.current) {
        ringRef.current.visible = true;
        ringRef.current.position.set(cx2, cy2, cz2);
        pulseRef.current += 0.05;
        const pulse = Math.sin(pulseRef.current) * 0.3 + 0.7;
        ringRef.current.scale.setScalar(pulse);
        const ringMat = ringRef.current.material as THREE.MeshBasicMaterial;
        ringMat.color.copy(phaseColor);
        ringMat.opacity = pulse * 0.5;
      }
    } else {
      if (currentRef.current) currentRef.current.visible = false;
      if (ringRef.current) ringRef.current.visible = false;
    }
  });

  const sphereGeo = useMemo(() => new THREE.SphereGeometry(POINT_RADIUS, 8, 8), []);
  const pointMat = useMemo(() => new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.1 }), []);

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
    </>
  );
}

/** Placeholder shown when no data. */
function PlaceholderSphere() {
  return (
    <mesh>
      <sphereGeometry args={[0.05, 8, 8]} />
      <meshStandardMaterial color="#AFAFAF" transparent opacity={0.3} />
    </mesh>
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

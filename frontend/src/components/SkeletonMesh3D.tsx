"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { FEATURE_BONES } from "@/lib/skeleton";
import { interpolateSmplParams, axisAngleToQuat } from "@/lib/smpl";
import type { SubjectState, SmplParams } from "@/lib/types";

interface SkeletonMesh3DProps {
  joints: [number, number, number][] | null;
  color: string;
  subjectsRef?: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef?: React.MutableRefObject<number | null>;
}

/** Single joint sphere in the wireframe skeleton. */
function JointSphere({
  position,
  color,
}: {
  position: [number, number, number];
  color: string;
}) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.06, 16, 16]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

/** Single bone cylinder connecting two joints. */
function BoneCylinder({
  start,
  end,
  color,
}: {
  start: [number, number, number];
  end: [number, number, number];
  color: string;
}) {
  const ref = useRef<THREE.Mesh>(null);

  // Compute position, rotation, and scale for a cylinder between two points
  const { position, quaternion, length } = useMemo(() => {
    const s = new THREE.Vector3(...start);
    const e = new THREE.Vector3(...end);
    const mid = new THREE.Vector3().addVectors(s, e).multiplyScalar(0.5);
    const dir = new THREE.Vector3().subVectors(e, s);
    const len = dir.length();
    dir.normalize();

    // CylinderGeometry is along Y-axis by default
    const up = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);

    return {
      position: [mid.x, mid.y, mid.z] as [number, number, number],
      quaternion: quat,
      length: len,
    };
  }, [start, end]);

  return (
    <mesh ref={ref} position={position} quaternion={quaternion}>
      <cylinderGeometry args={[0.03, 0.03, length, 8]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

/** Wireframe skeleton rendered with spheres and cylinders. */
function WireframeSkeleton({
  joints,
  color,
}: {
  joints: [number, number, number][];
  color: string;
}) {
  return (
    <group>
      {joints.map((pos, i) => (
        <JointSphere key={`j-${i}`} position={pos} color={color} />
      ))}
      {FEATURE_BONES.filter(
        ([a, b]) => a < joints.length && b < joints.length
      ).map(([a, b], i) => (
        <BoneCylinder
          key={`b-${i}`}
          start={joints[a]}
          end={joints[b]}
          color={color}
        />
      ))}
    </group>
  );
}

/** SMPL mesh driven by backend pose parameters at 60fps. */
function SmplMesh({
  subjectsRef,
  selectedSubjectRef,
  color,
}: {
  subjectsRef: React.MutableRefObject<Map<number, SubjectState>>;
  selectedSubjectRef: React.MutableRefObject<number | null>;
  color: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);

  // 60fps update: read SMPL params from refs, interpolate, apply
  useFrame(() => {
    const selectedId = selectedSubjectRef.current;
    if (selectedId === null) return;
    const subject = subjectsRef.current.get(selectedId);
    if (!subject) return;

    const { smplFrame } = subject;
    if (!smplFrame.current) return;

    // Compute interpolation factor
    let params: SmplParams;
    if (smplFrame.prev && smplFrame.prevTime > 0) {
      const elapsed = performance.now() - smplFrame.currentTime;
      const interval = smplFrame.currentTime - smplFrame.prevTime;
      const t = interval > 0 ? Math.min(elapsed / interval, 1) : 1;
      params = interpolateSmplParams(smplFrame.prev, smplFrame.current, t);
    } else {
      params = smplFrame.current;
    }

    // Apply translation
    if (meshRef.current) {
      meshRef.current.position.set(
        params.trans[0],
        params.trans[1],
        params.trans[2]
      );

      // Apply root rotation from first 3 pose params
      const [qx, qy, qz, qw] = axisAngleToQuat(
        params.pose[0],
        params.pose[1],
        params.pose[2]
      );
      meshRef.current.quaternion.set(qx, qy, qz, qw);
    }
  });

  // Placeholder sphere mesh -- will be replaced with actual SMPL .glb when available
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshStandardMaterial
        ref={materialRef}
        color={color}
        transparent
        opacity={0.8}
        wireframe
      />
    </mesh>
  );
}

/** Placeholder text when no data is available. */
function PlaceholderText({ text }: { text: string }) {
  return (
    <mesh position={[0, 0, 0]}>
      <sphereGeometry args={[0.05, 8, 8]} />
      <meshStandardMaterial color="#AFAFAF" transparent opacity={0.3} />
    </mesh>
  );
}

/** Inner scene content -- must be inside Canvas. */
function SceneContent({
  joints,
  color,
  subjectsRef,
  selectedSubjectRef,
}: SkeletonMesh3DProps) {
  // Determine if SMPL data is available
  const hasSmpl = useMemo(() => {
    if (!subjectsRef || !selectedSubjectRef) return false;
    const selectedId = selectedSubjectRef.current;
    if (selectedId === null) return false;
    const subject = subjectsRef.current.get(selectedId);
    return subject?.smplFrame?.current !== null && subject?.smplFrame?.current !== undefined;
  }, [subjectsRef, selectedSubjectRef]);

  if (hasSmpl && subjectsRef && selectedSubjectRef) {
    return (
      <>
        <SmplMesh
          subjectsRef={subjectsRef}
          selectedSubjectRef={selectedSubjectRef}
          color={color}
        />
        {/* Also show wireframe joints overlaid for reference */}
        {joints && <WireframeSkeleton joints={joints} color={color} />}
      </>
    );
  }

  if (!joints) {
    return <PlaceholderText text="No pose detected" />;
  }

  return <WireframeSkeleton joints={joints} color={color} />;
}

export default function SkeletonMesh3D({
  joints,
  color,
  subjectsRef,
  selectedSubjectRef,
}: SkeletonMesh3DProps) {
  return (
    <div className="absolute inset-0 w-full h-full">
      <Canvas
        camera={{ position: [0, 0, 4], fov: 50 }}
        style={{ background: "#FAFAFA" }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-3, -3, 2]} intensity={0.3} />
        <SceneContent
          joints={joints}
          color={color}
          subjectsRef={subjectsRef}
          selectedSubjectRef={selectedSubjectRef}
        />
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          minDistance={2}
          maxDistance={8}
        />
      </Canvas>
    </div>
  );
}

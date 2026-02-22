"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion } from "framer-motion";

interface Vector3DVisualizerProps {
  vectorData: number[];
  playerName: string;
}

export function Vector3DVisualizer({ vectorData, playerName }: Vector3DVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const autoRotateAngleRef = useRef(0);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [isRotating, setIsRotating] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [lastTouch, setLastTouch] = useState({ x: 0, y: 0 });
  const [lastPinchDistance, setLastPinchDistance] = useState(0);
  const [autoRotate, setAutoRotate] = useState(true);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#16213e33';
    ctx.lineWidth = 1;
    for (let i = 0; i < 10; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i * height) / 10);
      ctx.lineTo(width, (i * height) / 10);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo((i * width) / 10, 0);
      ctx.lineTo((i * width) / 10, height);
      ctx.stroke();
    }

    const cx = width / 2 + offset.x;
    const cy = height / 2 + offset.y;

    const currentRotY = autoRotate ? autoRotateAngleRef.current : rotation.y;
    const currentRotX = autoRotate ? 0 : rotation.x;

    const points: Array<{ x: number; y: number; z: number; intensity: number }> = [];

    for (let i = 0; i < 10; i++) {
      const startIdx = i * 5;
      const dimCluster = vectorData.slice(startIdx, startIdx + 5);
      const avgVal = dimCluster.reduce((a, b) => a + b, 0) / dimCluster.length;

      const angle = (i / 10) * Math.PI * 2 + currentRotY;
      const radius = avgVal * 80 * zoom;

      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = (vectorData[i] - 0.5) * 100 * zoom;

      const cosX = Math.cos(currentRotX);
      const sinX = Math.sin(currentRotX);
      const newY = y * cosX - z * sinX;
      const newZ = y * sinX + z * cosX;

      points.push({ x, y: newY, z: newZ, intensity: avgVal });
    }

    points.sort((a, b) => a.z - b.z);

    // Draw connections
    ctx.strokeStyle = '#0ea5e9';
    ctx.lineWidth = 2;
    for (let i = 0; i < points.length - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];

      const perspective1 = 300 / (300 + p1.z);
      const perspective2 = 300 / (300 + p2.z);

      const sx1 = cx + p1.x * perspective1;
      const sy1 = cy + p1.y * perspective1;
      const sx2 = cx + p2.x * perspective2;
      const sy2 = cy + p2.y * perspective2;

      const alpha = Math.min(perspective1, perspective2);
      ctx.globalAlpha = alpha * 0.5;

      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.stroke();
    }

    // Draw points
    points.forEach((point) => {
      const perspective = 300 / (300 + point.z);
      const sx = cx + point.x * perspective;
      const sy = cy + point.y * perspective;
      const size = perspective * 8;

      ctx.globalAlpha = perspective;

      const gradient = ctx.createRadialGradient(sx, sy, 0, sx, sy, size * 2);
      gradient.addColorStop(0, `rgba(59, 130, 246, ${point.intensity})`);
      gradient.addColorStop(0.5, `rgba(14, 165, 233, ${point.intensity * 0.5})`);
      gradient.addColorStop(1, 'rgba(6, 182, 212, 0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(sx, sy, size * 2, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = point.intensity > 0.7 ? '#ef4444' : point.intensity > 0.4 ? '#eab308' : '#22c55e';
      ctx.beginPath();
      ctx.arc(sx, sy, size, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.globalAlpha = 1;

    ctx.fillStyle = '#0ea5e9';
    ctx.font = '10px "Press Start 2P"';
    ctx.fillText('50D VECTOR', 10, 20);
    ctx.fillText('SPACE', 10, 35);

    if (autoRotate) {
      autoRotateAngleRef.current += 0.01;
    }
    animationRef.current = requestAnimationFrame(draw);
  }, [vectorData, rotation, zoom, offset, autoRotate]);

  useEffect(() => {
    draw();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [draw]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.shiftKey) {
      setIsDragging(true);
    } else {
      setIsRotating(true);
      setAutoRotate(false);
    }
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging && !isRotating) return;

    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;

    if (isDragging) {
      setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
    } else if (isRotating) {
      setRotation(prev => ({ x: prev.x + dy * 0.01, y: prev.y + dx * 0.01 }));
    }

    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsRotating(false);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.3, Math.min(3, prev * delta)));
  };

  const handleReset = () => {
    setRotation({ x: 0, y: 0 });
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setAutoRotate(true);
  };

  // Touch event handlers for mobile/tablet interaction
  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (e.touches.length === 1) {
      const touch = e.touches[0];
      setIsRotating(true);
      setAutoRotate(false);
      setLastTouch({ x: touch.clientX, y: touch.clientY });
    } else if (e.touches.length === 2) {
      const touch = e.touches[0];
      setIsDragging(true);
      setIsRotating(false);
      setLastTouch({ x: touch.clientX, y: touch.clientY });
      const distance = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      setLastPinchDistance(distance);
    }
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();

    if (e.touches.length === 1 && isRotating) {
      const touch = e.touches[0];
      const dx = touch.clientX - lastTouch.x;
      const dy = touch.clientY - lastTouch.y;

      setRotation(prev => ({
        x: prev.x + dy * 0.01,
        y: prev.y + dx * 0.01
      }));

      setLastTouch({ x: touch.clientX, y: touch.clientY });
    } else if (e.touches.length === 2 && isDragging) {
      const touch = e.touches[0];
      const dx = touch.clientX - lastTouch.x;
      const dy = touch.clientY - lastTouch.y;

      setOffset(prev => ({
        x: prev.x + dx,
        y: prev.y + dy
      }));

      setLastTouch({ x: touch.clientX, y: touch.clientY });
    }

    if (e.touches.length === 2) {
      const distance = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      if (lastPinchDistance > 0) {
        const delta = distance / lastPinchDistance;
        setZoom(prev => Math.max(0.3, Math.min(3, prev * delta)));
      }
      setLastPinchDistance(distance);
    }
  };

  const handleTouchEnd = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    setIsDragging(false);
    setIsRotating(false);
  };

  return (
    <motion.div
      className="bg-gray-950 rounded-lg border-4 border-white overflow-hidden shadow-2xl"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="bg-blue-600 px-4 py-2 border-b-4 border-white">
        <div className="flex justify-between items-center">
          <h3 className="text-white text-xs" style={{ fontFamily: "'Press Start 2P', cursive" }}>
            {playerName} - VECTOR ANALYSIS
          </h3>
          <button
            onClick={handleReset}
            className="text-white text-xs px-2 py-1 bg-blue-700 hover:bg-blue-800 rounded border-2 border-white transition-colors"
            style={{ fontFamily: "'Press Start 2P', cursive" }}
          >
            RESET
          </button>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="w-full cursor-move"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      />
      <div className="bg-gray-900 px-4 py-3 border-t-4 border-gray-700">
        <div className="text-xs text-gray-400 space-y-2" style={{ fontFamily: "'Press Start 2P', cursive" }}>
          <div className="flex justify-between">
            <span>DIMS: 50</span>
            <span>ZOOM: {zoom.toFixed(1)}x</span>
          </div>
          <div className="text-gray-500 text-xs" style={{ fontSize: '8px' }}>
            TOUCH: ROTATE | 2-FINGER: PAN/ZOOM | MOUSE: DRAG/SCROLL
          </div>
          <div className="flex gap-2 mt-2">
            <button
              onClick={() => setAutoRotate(!autoRotate)}
              className={`flex-1 px-2 py-1 rounded border-2 transition-colors ${
                autoRotate
                  ? 'bg-green-700 border-green-500 text-white'
                  : 'bg-gray-800 border-gray-600 text-gray-400'
              }`}
              style={{ fontSize: '8px' }}
            >
              {autoRotate ? 'AUTO: ON' : 'AUTO: OFF'}
            </button>
          </div>
          <div className="mt-3 pt-3 border-t border-gray-700 space-y-1 text-gray-500" style={{ fontSize: '8px' }}>
            <div>DIMS 0-9: HEAD IMPACT SENSORS</div>
            <div>DIMS 10-19: EXERTION LEVELS</div>
            <div>DIMS 20-29: HEART RATE VARIABILITY</div>
            <div>DIMS 30-39: MOVEMENT QUALITY</div>
            <div>DIMS 40-49: PERFORMANCE METRICS</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

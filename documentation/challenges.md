# BRACE - Challenges

## VR Integration (Meta Quest 3)

### Challenge: Bandwidth over Tailscale VPN
The web frontend sends ~600KB/s per subject (landmarks, SRP joints, UMAP embeddings, cluster representatives). With 5 subjects at 30fps, this exceeds reasonable bandwidth for VR over Tailscale. We solved this by adding a `client=vr` query parameter that triggers a stripped-down response format — unselected subjects get only bbox + label (~80 bytes), and even selected subjects omit landmarks, SRP joints, UMAP, and cluster representatives. This achieved ~10x bandwidth reduction.

### Challenge: Subject Selection in VR
The web frontend handles subject selection entirely client-side (click on canvas → identify closest skeleton). VR can't do this the same way since it renders bounding boxes in 3D, not skeleton overlays. We added a `select_subject` WebSocket message type so the VR client can tell the server which subject it cares about, and the server only sends detailed analysis data for that subject.

### Challenge: Coordinate System Mismatch
BRACE uses image-space normalized coordinates [0,1] with y=0 at the top (standard for computer vision). Unity's viewport has y=0 at the bottom. The VR client must flip Y when converting bounding boxes to 3D world space.

### Challenge: Frame Capture on Quest 3
The Passthrough Camera API requires specific Android API levels (32+), Vulkan graphics, and has inherent 40-60ms latency from camera to texture. GPU-to-CPU readback for JPEG encoding adds additional overhead. We recommend on-demand capture and downscaling to 480p before encoding.

### Challenge: Latency Budget
End-to-end latency (camera capture → network → GPU inference → response) is 130-260ms. This is acceptable for bounding box overlays but too high for precise skeleton-to-body alignment. We chose bounding boxes + floating stats panels rather than skeleton overlays for the VR visualization.

## Basketball Game Analysis

### Challenge: Player Identity Fragmentation Across Scene Cuts
During end-to-end testing of the basketball game processor, a 70-second NBA clip produced 30+ subject IDs for only ~10 actual players. Each camera angle change (scene cut) resets the tracker, creating entirely new track IDs. Without working CLIP-ReID (disabled due to a bug), every scene cut created duplicate subjects for every visible player. We solved this by using jersey numbers as a strong identity signal: after Gemini Pro detects a player's jersey number and team color, this information is fed back into the IdentityResolver. On subsequent scene cuts, players are re-identified by their jersey number + color combination before falling back to appearance/spatial matching. A merge mechanism retroactively combines fragment subjects when jersey detection reveals duplicates, and short-lived fragments (< 1 second) are filtered from final results.

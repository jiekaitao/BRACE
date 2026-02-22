export interface Sport {
  id: string;
  name: string;
  /** Demo video filename in backend/data/sports_videos/, served via /api/demo-videos */
  demoVideo?: string;
}

export const SPORTS: Sport[] = [
  {
    id: "basketball",
    name: "Basketball",
    demoVideo: "basketball.mp4",
  },
  {
    id: "football",
    name: "Football",
  },
  {
    id: "soccer",
    name: "Soccer",
  },
  {
    id: "large-class",
    name: "Large Class",
  },
  {
    id: "boxing",
    name: "Boxing",
  },
];

export interface SportVideo {
  id: string;
  title: string;
  thumbnail: string;
}

export interface Sport {
  id: string;
  name: string;
  image: string;
  gradient: string;
  videos: SportVideo[];
}

export const SPORTS: Sport[] = [
  {
    id: "basketball",
    name: "Basketball",
    image: "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=600&q=80",
    gradient: "from-[#FF4B4B] to-[#FF9600]",
    videos: [
      {
        id: "bball-1",
        title: "Full Court Scrimmage",
        thumbnail: "https://images.unsplash.com/photo-1574623452334-1e0ac2b3ccb4?w=300&q=80",
      },
      {
        id: "bball-2",
        title: "Free Throw Practice",
        thumbnail: "https://images.unsplash.com/photo-1559692048-79a3f837883d?w=300&q=80",
      },
      {
        id: "bball-3",
        title: "Layup Drills",
        thumbnail: "https://images.unsplash.com/photo-1608245449230-4ac19066d2d0?w=300&q=80",
      },
    ],
  },
  {
    id: "football",
    name: "Football",
    image: "https://images.unsplash.com/photo-1508098682722-e99c43a406b2?w=600&q=80",
    gradient: "from-[#58CC02] to-[#1CB0F6]",
    videos: [
      {
        id: "fb-1",
        title: "Route Running",
        thumbnail: "https://images.unsplash.com/photo-1612872087720-bb876e2e67d1?w=300&q=80",
      },
      {
        id: "fb-2",
        title: "Tackle Form Review",
        thumbnail: "https://images.unsplash.com/photo-1560272564-c83b66b1ad12?w=300&q=80",
      },
      {
        id: "fb-3",
        title: "Sprint Conditioning",
        thumbnail: "https://images.unsplash.com/photo-1575361204480-aadea25e6e68?w=300&q=80",
      },
    ],
  },
  {
    id: "soccer",
    name: "Soccer",
    image: "https://images.unsplash.com/photo-1431324155629-1a6deb1dec8d?w=600&q=80",
    gradient: "from-[#CE82FF] to-[#1CB0F6]",
    videos: [
      {
        id: "soc-1",
        title: "Passing Accuracy Drill",
        thumbnail: "https://images.unsplash.com/photo-1579952363873-27f3bade9f55?w=300&q=80",
      },
      {
        id: "soc-2",
        title: "Dribbling Through Cones",
        thumbnail: "https://images.unsplash.com/photo-1551958219-acbc608c6377?w=300&q=80",
      },
      {
        id: "soc-3",
        title: "Match Highlights",
        thumbnail: "https://images.unsplash.com/photo-1522778119026-d647f0596c20?w=300&q=80",
      },
    ],
  },
];
